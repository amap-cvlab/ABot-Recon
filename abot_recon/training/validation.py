"""Streaming validation with the development model's per-sequence semantics."""

from __future__ import annotations

import torch


def _concat_batch(values):
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {key: _concat_batch([value[key] for value in values]) for key in first}
    return None if first is None else values


@torch.no_grad()
def stream_validation_forward(model, images: torch.Tensor):
    """Run each clip with fresh KV/camera state; restore training settings even on error."""
    if images.ndim != 5 or images.shape[0] < 1 or images.shape[1] < 1:
        raise ValueError("stream validation expects nonempty [B,N,C,H,W] images")
    had_mode = hasattr(model, "infer_mode")
    old_mode = getattr(model, "infer_mode", None)
    flex_flags = [(module, module.use_chunk_flex_attention)
                  for module in model.modules() if hasattr(module, "use_chunk_flex_attention")]
    try:
        model.infer_mode = "stream"
        for module, _ in flex_flags:
            module.use_chunk_flex_attention = False
        # Match development validation, including B=1-only streaming backends.
        predictions = [model.inference_stream(images[index:index + 1], causal_global_attn=True)
                       for index in range(images.shape[0])]
        return predictions[0] if len(predictions) == 1 else _concat_batch(predictions)
    finally:
        for module, enabled in flex_flags:
            module.use_chunk_flex_attention = enabled
        if had_mode:
            model.infer_mode = old_mode
        else:
            delattr(model, "infer_mode")
