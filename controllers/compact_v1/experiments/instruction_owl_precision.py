"""Bound vision compute precision while keeping OWL weights and heads FP32."""
from types import MethodType

import torch


def image_embedder(self, pixel_values):
    if self.training or pixel_values.device.type != "cuda":
        return self._dream_original_image_embedder(pixel_values)
    # Autocast is scoped to the vision tower. Text embeddings, class scores,
    # and continuous bounding boxes are computed outside this context.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feature_map, vision_output = self._dream_original_image_embedder(pixel_values)
    return feature_map.float(), vision_output


def configure_owl_vision_precision(model):
    if any(parameter.dtype != torch.float32
           for parameter in model.owlv2.vision_model.parameters()):
        raise ValueError("Vision autocast requires the original FP32 weights")
    supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if supported and not hasattr(model, "_dream_original_image_embedder"):
        model._dream_original_image_embedder = model.image_embedder
        model.image_embedder = MethodType(image_embedder, model)
    return dict(enabled=supported, weights_precision="float32",
                cuda_vision_compute="bfloat16_autocast" if supported else "float32",
                text_class_and_box_heads="float32",
                cpu_and_training_use_original=True,
                numerical_identity="Not bitwise identical; separately measured and evaluated.")
