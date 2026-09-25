"""Chunk-parallel KV attention (streaming; dense window, no sparse keyframes)."""

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from abot_recon.modeling.pi3.models.layers.attention import FlashAttentionRope, _sdpa_math_ctx

from abot_recon.modeling.streaming.core.window_mask import streaming_chunk_attention_bias
from abot_recon.modeling.streaming.core.flex_attention import (
    get_compiled_streaming_flex_attention,
    streaming_flex_attention,
)


def kv_carry_fingerprint_for_bias_cache(
    kv_carry_meta: Optional[Dict[str, Any]],
) -> Any:
    """Dense concat mask ignores ``kv_carry_meta``; signature kept for call-site stability."""
    del kv_carry_meta
    return ()


class ChunkFlashAttentionRope(FlashAttentionRope):
    @classmethod
    def from_flash_attn(cls, old: FlashAttentionRope) -> "ChunkFlashAttentionRope":
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        obj = cls(
            dim=old.qkv.in_features,
            num_heads=old.num_heads,
            qkv_bias=old.qkv.bias is not None,
            proj_bias=old.proj.bias is not None,
            attn_drop=old.attn_drop.p,
            proj_drop=old.proj_drop.p,
            qk_norm=not isinstance(old.q_norm, nn.Identity),
            norm_layer=norm_layer,
            rope=old.rope,
            use_packaged_flash_attn=getattr(old, "use_packaged_flash_attn", False),
        )
        obj.load_state_dict(old.state_dict(), strict=True)
        return obj

    def compute_streaming_attn_bias(
        self,
        *,
        t_past: int,
        num_new_frames: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
        temporal_window_frames: Optional[int] = None,
        kv_carry_meta: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        t_new = int(num_new_frames)
        tpf = int(tokens_per_frame)
        W = (
            int(temporal_window_frames)
            if temporal_window_frames is not None
            else int(getattr(self, "chunk_temporal_window", 12))
        )
        del kv_carry_meta
        return streaming_chunk_attention_bias(
            t_past,
            t_new,
            tpf,
            dtype,
            device,
            temporal_window_frames=W,
            num_reference_frames=0,
        )

    def stream_chunk_forward(
        self,
        x: Tensor,
        pos_q: Tensor,
        past_key_values: Optional[Tuple[Tensor, Tensor]],
        num_new_frames: int,
        chunk_key_positions: Tensor,
        temporal_window_frames: Optional[int] = None,
        kv_carry_meta: Optional[Dict[str, Any]] = None,
        cached_streaming_attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, seq_len, C = x.shape
        t_new = int(num_new_frames)
        if seq_len % t_new != 0:
            raise ValueError(f"seq_len {seq_len} not divisible by num_new_frames {t_new}")
        tpf = seq_len // t_new
        qkv = self.qkv(x).reshape(B, seq_len, 3, self.num_heads, C // self.num_heads).transpose(
            1, 3
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]
        num_heads = q.shape[1]
        head_dim = q.shape[-1]
        k_b = k.reshape(B, num_heads, t_new, tpf, head_dim)
        v_b = v.reshape(B, num_heads, t_new, tpf, head_dim)
        if past_key_values is not None:
            pk, pv = past_key_values
            t_past = pk.shape[2]
            k_b = torch.cat([pk, k_b], dim=2)
            v_b = torch.cat([pv, v_b], dim=2)
        else:
            t_past = 0
        new_kv = (k_b.contiguous(), v_b.contiguous())
        t_k = k_b.shape[2]
        k_flat = k_b.reshape(B, num_heads, t_k * tpf, head_dim)
        v_flat = v_b.reshape(B, num_heads, t_k * tpf, head_dim)
        Lq = t_new * tpf
        Lk = t_k * tpf
        W = (
            int(temporal_window_frames)
            if temporal_window_frames is not None
            else int(getattr(self, "chunk_temporal_window", 12))
        )
        q, k_flat = self.q_norm(q).to(v_flat.dtype), self.k_norm(k_flat).to(v_flat.dtype)
        v_flat = v_flat.to(q.dtype)
        if self.rope is not None:
            q = self.rope(q, pos_q)
            k_flat = self.rope(k_flat, chunk_key_positions)
        dropout_p = self.attn_drop.p if self.training else 0.0
        use_flex = bool(getattr(self, "use_chunk_flex_attention", False)) and q.is_cuda
        if use_flex:
            reference_frames = int(getattr(self, "streaming_num_reference_frames", 0))
            # Validation also needs the compiled, memory-efficient kernel.
            if bool(getattr(self, "use_chunk_flex_compile", False)):
                flex_module = get_compiled_streaming_flex_attention(
                    t_past,
                    t_new,
                    tpf,
                    reference_frames,
                    W,
                    0,
                )
                out = flex_module(q, k_flat, v_flat)
            else:
                out = streaming_flex_attention(
                    q,
                    k_flat,
                    v_flat,
                    t_past=t_past,
                    t_new=t_new,
                    tokens_per_frame=tpf,
                    num_reference_frames=reference_frames,
                    local_window_frames=W,
                    num_summary_tokens=0,
                )
        else:
            if cached_streaming_attn_bias is not None:
                bias_flat = cached_streaming_attn_bias
                if tuple(bias_flat.shape) != (Lq, Lk):
                    raise ValueError(
                        f"cached_streaming_attn_bias shape {tuple(bias_flat.shape)} != ({Lq}, {Lk})"
                    )
                bias = bias_flat.view(1, 1, Lq, Lk)
            else:
                bias = self.compute_streaming_attn_bias(
                    t_past=t_past,
                    num_new_frames=t_new,
                    tokens_per_frame=tpf,
                    dtype=q.dtype,
                    device=q.device,
                    temporal_window_frames=W,
                    kv_carry_meta=kv_carry_meta,
                ).view(1, 1, Lq, Lk)
            with _sdpa_math_ctx():
                out = scaled_dot_product_attention(
                    q, k_flat, v_flat, attn_mask=bias, dropout_p=dropout_p  # type: ignore[arg-type]
                )
        out = out.transpose(1, 2).reshape(B, seq_len, C)
        out = self._apply_output_gate(x, out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, new_kv

    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        xpos=None,
        past_key_values: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        return super().forward(x, attn_bias, xpos, past_key_values, use_cache)
