"""Inference-only streaming stream attention (packed ``StreamingKVState``).

Packed carry already stores only visible K/V; SDPA runs without an additive mask.
Training / dense ``(pk, pv)`` chunks still use ``stream_chunk_forward`` + streaming bias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from abot_recon.modeling.pi3.models.layers.attention import _sdpa_math_ctx

from abot_recon.modeling.streaming.kv_state import StreamingKVState
from abot_recon.modeling.streaming.rope3d import apply_rotary_emb

if TYPE_CHECKING:
    from abot_recon.modeling.streaming.attention import StreamingFlashAttention


def absolute_rope3d_visible_freqs(
    rope: Any,
    *,
    stored_past_spans: Sequence[Tuple[int, int]],
    current_frame_idx: int,
    num_new_frames: int,
    tokens_per_frame: int,
    patches_h: int,
    patches_w: int,
    patch_start_idx: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Generate K/Q RoPE only for visible frames, at absolute frame indices."""
    key_parts = []
    for absolute_frame_idx, token_count in stored_past_spans:
        frame_freqs = rope(
            1,
            patches_h,
            patches_w,
            patch_start_idx,
            device,
            f_start=int(absolute_frame_idx),
            f_end=int(absolute_frame_idx) + 1,
        )
        key_parts.append(frame_freqs[:, :, : int(token_count), :])

    current_freqs = rope(
        int(num_new_frames),
        patches_h,
        patches_w,
        patch_start_idx,
        device,
        f_start=int(current_frame_idx),
        f_end=int(current_frame_idx) + int(num_new_frames),
    )
    key_parts.append(current_freqs[:, :, : int(tokens_per_frame), :])
    return torch.cat(key_parts, dim=2), current_freqs


def stream_window_attention(
    attn: StreamingFlashAttention,
    x: Tensor,
    *,
    pos_q: Tensor,
    hs_carry: StreamingKVState,
    num_new_frames: int,
    chunk_key_positions: Tensor,
    image_hw: Optional[Tuple[int, int]],
    patch_start_idx: int,
) -> Tuple[Tensor, StreamingKVState]:
    """One global-block step: past packed carry + ``num_new_frames`` new frame(s), no attn bias."""
    if attn.memory_mode != "streaming":
        raise ValueError("stream_window_attention requires memory_mode=streaming")
    B, seq_len, C = x.shape
    t_new = int(num_new_frames)
    if seq_len % t_new != 0:
        raise ValueError(f"seq_len {seq_len} not divisible by num_new_frames {t_new}")
    tpf = seq_len // t_new

    qkv = attn.qkv(x).reshape(B, seq_len, 3, attn.num_heads, C // attn.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    num_heads = q.shape[1]
    head_dim = q.shape[-1]

    k_new = k.reshape(B, num_heads, t_new, tpf, head_dim)[:, :, 0, :, :].contiguous()
    v_new = v.reshape(B, num_heads, t_new, tpf, head_dim)[:, :, 0, :, :].contiguous()
    k_past, v_past, stored_past_spans = hs_carry.flatten_past_kv_indexed()
    if k_past.shape[2] == 0:
        k_flat = k_new
        v_flat = v_new
    else:
        k_flat = torch.cat([k_past, k_new], dim=2).contiguous()
        v_flat = torch.cat([v_past, v_new], dim=2).contiguous()

    Lk = int(k_flat.shape[2])
    Lq = int(t_new * tpf)
    q, k_flat = attn.q_norm(q).to(v_flat.dtype), attn.k_norm(k_flat).to(v_flat.dtype)
    v_flat = v_flat.to(q.dtype)

    T_p = int(hs_carry.total_frames_seen)
    enc = attn.global_pos_encoding
    if enc == "rope3d":
        if attn.rope3d_embed is None:
            raise RuntimeError("global_pos_encoding=rope3d requires rope3d_embed module")
        if image_hw is None:
            raise ValueError("image_hw (H, W) required for rope3d RoPE stream path")
        H_img, W_img = int(image_hw[0]), int(image_hw[1])
        patch = 14
        pph, ppw = H_img // patch, W_img // patch
        # Build RoPE only for K/V frames that are actually retained in the
        # packed carry.  Crucially, each frame keeps its absolute temporal
        # index: a window containing frames 4989..4999 plus current frame 5000
        # uses those indices, never a renumbered 0..11 coordinate system.
        freqs_k, current_freqs = absolute_rope3d_visible_freqs(
            attn.rope3d_embed,
            stored_past_spans=stored_past_spans,
            current_frame_idx=T_p,
            num_new_frames=t_new,
            tokens_per_frame=tpf,
            patches_h=pph,
            patches_w=ppw,
            patch_start_idx=patch_start_idx,
            device=q.device,
        )
        freqs_q = current_freqs[:, :, :Lq, :]
        if freqs_k.shape[2] != Lk or freqs_q.shape[2] != Lq:
            raise RuntimeError(
                "RoPE3D sparse RoPE length mismatch: "
                f"freqs_k={freqs_k.shape[2]} Lk={Lk} "
                f"freqs_q={freqs_q.shape[2]} Lq={Lq}"
            )
        q = apply_rotary_emb(q, freqs_q)
        k_flat = apply_rotary_emb(k_flat, freqs_k)
    elif enc == "pi3_2d":
        if attn.rope is not None:
            if chunk_key_positions.shape[1] != Lk:
                raise ValueError(
                    f"chunk_key_positions {chunk_key_positions.shape[1]} != Lk {Lk}"
                )
            q = attn.rope(q, pos_q)
            k_flat = attn.rope(k_flat, chunk_key_positions)
    else:
        raise RuntimeError(f"stream packed carry needs pi3_2d or rope3d, got {enc!r}")

    with _sdpa_math_ctx():
        out = scaled_dot_product_attention(q, k_flat, v_flat, attn_mask=None, dropout_p=0.0)

    out = out.transpose(1, 2).reshape(B, seq_len, C)
    out = attn._apply_output_gate(x, out)
    out = attn.proj(out)
    out = attn.proj_drop(out)

    carry_out = hs_carry.clone_buffers()
    carry_out.append_committed_frame(k_new, v_new)
    return out, carry_out
