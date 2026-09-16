"""Sample the existing dense feature grid before per-pixel normalization."""
import numpy as np
import torch
import torch.nn.functional as F


def sampled_bilinear_grid(features, size, ys, xs):
    """Compute retained FP32 vectors with the CPU channel-last interpolation order.

    The production 1152-channel grid needs only 80 x 107 output locations.
    Keeping channels contiguous and accumulating the four weighted corners
    avoids materializing the unused 240 x 320 feature image. Other layouts
    and channel counts retain PyTorch's original interpolation path.
    """
    batch, channels, patch_h, patch_w = features.shape
    height, width = size
    if (features.device.type != "cpu" or features.dtype != torch.float32
            or channels % 16 or not features.is_contiguous(memory_format=torch.channels_last)
            or patch_h == height or patch_w == width):
        return F.interpolate(features, size, mode="bilinear", align_corners=True)[:, :, ys[:, None], xs[None, :]]
    y = torch.as_tensor(ys, dtype=torch.float32) * ((patch_h - 1) / max(1, height - 1) if height > 1 else 0.)
    x = torch.as_tensor(xs, dtype=torch.float32) * ((patch_w - 1) / max(1, width - 1) if width > 1 else 0.)
    y0 = y.floor().long().clamp_max(patch_h - 1)
    x0 = x.floor().long().clamp_max(patch_w - 1)
    y1 = (y0 + 1).clamp_max(patch_h - 1)
    x1 = (x0 + 1).clamp_max(patch_w - 1)
    dy = (y - y0).clamp(0., 1.)
    dx = (x - x0).clamp(0., 1.)
    indices = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
    weights = [(1-dy)[:, None]*(1-dx)[None, :], (1-dy)[:, None]*dx[None, :],
               dy[:, None]*(1-dx)[None, :], dy[:, None]*dx[None, :]]
    values = features.permute(0, 2, 3, 1)
    output = None
    for index in (2, 3, 1, 0):
        iy, ix = indices[index]
        corner = values[:, iy[:, None], ix[None, :]]
        weight = weights[index][None, :, :, None]
        if output is None:
            output = corner * weight
        else:
            output.addcmul_(corner, weight)
    return output.permute(0, 3, 1, 2)


def sampled_mask_siglip(encoder, rgb, stride=6):
    height, width = rgb.shape[:2]
    scale = min(1., 320 / max(height, width))
    fh, fw = round(height * scale), round(width * scale)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    inputs = encoder.processor(images=image, padding="max_length", return_tensors="pt")
    inputs = {key: value.to(encoder.device) for key, value in inputs.items()}
    # These are the unchanged production MaskSiglipEncoder projection stages.
    output = encoder.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
    features = encoder.forward_one_block_(encoder.model.vision_model.head.attention,
                                          output.last_hidden_state)
    features = encoder.model.vision_model.head.layernorm(features)
    features = features + encoder.model.vision_model.head.mlp(features)
    features = features.detach().cpu()
    batch, channels, patch_h, patch_w = encoder.model.vision_model.embeddings.patch_embedding(
        inputs["pixel_values"]).shape
    features = features.reshape(batch, patch_h, patch_w, channels).permute(0, 3, 1, 2)
    ys = np.rint(np.arange(0, height, stride) * (fh - 1) / max(1, height - 1)).astype(int)
    xs = np.rint(np.arange(0, width, stride) * (fw - 1) / max(1, width - 1)).astype(int)
    # Every retained vector is identical; unused pixels never enter the memory.
    features = sampled_bilinear_grid(features, (fh, fw), ys, xs)
    # The production interpolation retains contiguous channels. Keep that
    # reduction layout after indexing, including for the 1152-channel model.
    features = features.contiguous(memory_format=torch.channels_last)
    return F.normalize(features, dim=1).permute(0, 2, 3, 1)[0]
