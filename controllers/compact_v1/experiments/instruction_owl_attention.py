"""Evaluate OWL's unchanged attention equations with bounded GPU memory."""
from types import MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def efficient_attention(query, key, value):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0., scale=1.)


def vision_attention(self, hidden_states, attention_mask=None,
                     causal_attention_mask=None, output_attentions=False):
    if (self.training or output_attentions or attention_mask is not None
            or causal_attention_mask is not None or hidden_states.device.type != "cuda"):
        return self._dream_original_forward(hidden_states, attention_mask,
                                            causal_attention_mask, output_attentions)
    return projected_attention(self, hidden_states)


def projected_attention(self, hidden_states):
    batch, length, channels = hidden_states.shape
    # Apply the original scale before the attention product. This helper never
    # casts parameters; an explicitly recorded caller may enable autocast.
    query = self._shape(self.q_proj(hidden_states) * self.scale, length, batch)
    key = self._shape(self.k_proj(hidden_states), length, batch)
    value = self._shape(self.v_proj(hidden_states), length, batch)
    output = efficient_attention(query, key, value)
    output = output.transpose(1, 2).reshape(batch, length, channels)
    return self.out_proj(output), None


def configure_owl_vision_attention(model):
    layers = model.owlv2.vision_model.encoder.layers
    for layer in layers:
        attention = layer.self_attn
        if any(parameter.dtype != torch.float32 for parameter in attention.parameters()):
            raise ValueError("This OWL attention configuration requires unchanged FP32 weights")
        if not hasattr(attention, "_dream_original_forward"):
            attention._dream_original_forward = attention.forward
            attention.forward = MethodType(vision_attention, attention)
    return dict(vision_layers=len(layers), backend="torch_memory_efficient_sdpa",
                weights_unchanged=True, precision="float32",
                masked_training_and_attention_output_use_original=True,
                numerical_identity="Equivalent attention equations; fused arithmetic is not bitwise identical.")
