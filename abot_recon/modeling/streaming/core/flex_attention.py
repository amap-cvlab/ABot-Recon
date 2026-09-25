"""Wenli-compatible FlexAttention backend for causal streaming windows.

The training path keeps all frame tokens in one tensor, but its visibility is
sparse: a query can only see causal reference frames, its local temporal
window, and optional per-frame summary tokens.  ``BlockMask`` represents that
pattern without materializing the quadratic floating-point bias used by SDPA.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the installed PyTorch build
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]
    FLEX_ATTENTION_AVAILABLE = False


# Fixed-shape training only compiles once, while variable-length experiments may
# need several specializations.  This matches the effective Wenli backend.
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.cache_size_limit = max(
        int(torch._dynamo.config.cache_size_limit), 128
    )


_BLOCK_MASK_CACHE_MAX = 512
_BLOCK_MASK_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_COMPILED_MODULES: Dict[Tuple[Any, ...], nn.Module] = {}


def assert_flex_attention_available() -> None:
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "FlexAttention is required for full-clip training. Install PyTorch >= 2.5 "
            "with torch.nn.attention.flex_attention support."
        )


def clear_flex_attention_caches() -> None:
    _BLOCK_MASK_CACHE.clear()
    _COMPILED_MODULES.clear()


def make_streaming_mask_mod(
    t_past: int,
    t_new: int,
    tokens_per_frame: int,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """Return the token mask used by the source FlexAttention implementation."""

    t_p = int(t_past)
    _ = int(t_new)
    tpf = int(tokens_per_frame)
    references = max(int(num_reference_frames), 0)
    window = max(int(local_window_frames), 1)
    summaries = max(int(num_summary_tokens), 0)

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del b, h
        query_frame = q_idx // tpf + t_p
        key_frame = kv_idx // tpf
        key_token = kv_idx % tpf
        causal = key_frame <= query_frame
        in_reference = key_frame < references
        in_window = key_frame >= query_frame - window + 1
        is_summary = key_token < summaries
        return causal & (in_reference | in_window | is_summary)

    return mask_mod


def _block_mask_cache_key(
    q: Tensor,
    k: Tensor,
    *,
    t_past: int,
    t_new: int,
    tokens_per_frame: int,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
) -> Tuple[Any, ...]:
    batch, heads, query_length, _ = q.shape
    return (
        str(q.device),
        int(batch),
        int(heads),
        int(query_length),
        int(k.shape[2]),
        int(t_past),
        int(t_new),
        int(tokens_per_frame),
        int(num_reference_frames),
        int(local_window_frames),
        int(num_summary_tokens),
    )


def _get_or_create_block_mask(
    q: Tensor,
    k: Tensor,
    *,
    t_past: int,
    t_new: int,
    tokens_per_frame: int,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
):
    assert create_block_mask is not None
    key = _block_mask_cache_key(
        q,
        k,
        t_past=t_past,
        t_new=t_new,
        tokens_per_frame=tokens_per_frame,
        num_reference_frames=num_reference_frames,
        local_window_frames=local_window_frames,
        num_summary_tokens=num_summary_tokens,
    )
    if key in _BLOCK_MASK_CACHE:
        _BLOCK_MASK_CACHE.move_to_end(key)
        return _BLOCK_MASK_CACHE[key]

    mask_mod = make_streaming_mask_mod(
        t_past,
        t_new,
        tokens_per_frame,
        num_reference_frames,
        local_window_frames,
        num_summary_tokens,
    )
    batch, heads, query_length, _ = q.shape
    block_mask = create_block_mask(
        mask_mod,
        B=batch,
        H=heads,
        Q_LEN=query_length,
        KV_LEN=k.shape[2],
        device=str(q.device),
        _compile=True,
    )
    _BLOCK_MASK_CACHE[key] = block_mask
    _BLOCK_MASK_CACHE.move_to_end(key)
    while len(_BLOCK_MASK_CACHE) > _BLOCK_MASK_CACHE_MAX:
        _BLOCK_MASK_CACHE.popitem(last=False)
    return block_mask


# ``create_block_mask`` internally inspects the nested mask function.  Keep it
# outside Dynamo capture, exactly as in the source Wenli implementation.
if getattr(torch, "_dynamo", None) is not None:
    _get_or_create_block_mask = torch._dynamo.disable(_get_or_create_block_mask)


def streaming_flex_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    t_past: int,
    t_new: int,
    tokens_per_frame: int,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
    scale: float | None = None,
) -> Tensor:
    assert_flex_attention_available()
    assert flex_attention is not None
    if scale is None:
        scale = float(q.shape[-1] ** -0.5)
    block_mask = _get_or_create_block_mask(
        q,
        k,
        t_past=t_past,
        t_new=t_new,
        tokens_per_frame=tokens_per_frame,
        num_reference_frames=num_reference_frames,
        local_window_frames=local_window_frames,
        num_summary_tokens=num_summary_tokens,
    )
    return flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=float(scale),
        enable_gqa=False,
    )


class StreamingFlexAttention(nn.Module):
    def __init__(
        self,
        t_past: int,
        t_new: int,
        tokens_per_frame: int,
        num_reference_frames: int,
        local_window_frames: int,
        num_summary_tokens: int,
    ) -> None:
        super().__init__()
        self.t_past = int(t_past)
        self.t_new = int(t_new)
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_reference_frames = int(num_reference_frames)
        self.local_window_frames = int(local_window_frames)
        self.num_summary_tokens = int(num_summary_tokens)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return streaming_flex_attention(
            q,
            k,
            v,
            t_past=self.t_past,
            t_new=self.t_new,
            tokens_per_frame=self.tokens_per_frame,
            num_reference_frames=self.num_reference_frames,
            local_window_frames=self.local_window_frames,
            num_summary_tokens=self.num_summary_tokens,
        )


def get_compiled_streaming_flex_attention(
    t_past: int,
    t_new: int,
    tokens_per_frame: int,
    num_reference_frames: int,
    local_window_frames: int,
    num_summary_tokens: int,
) -> nn.Module:
    """Return a concrete-shape compiled module safe for FlexAttention backward."""

    assert_flex_attention_available()
    key = (
        int(t_past),
        int(t_new),
        int(tokens_per_frame),
        int(num_reference_frames),
        int(local_window_frames),
        int(num_summary_tokens),
    )
    if key not in _COMPILED_MODULES:
        module = StreamingFlexAttention(*key)
        _COMPILED_MODULES[key] = torch.compile(
            module, dynamic=False, fullgraph=False
        )
    return _COMPILED_MODULES[key]
