"""Global-block attention: sliding window (parent) or streaming mask + optional 3D RoPE."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from abot_recon.modeling.pi3.models.layers.attention import FlashAttentionRope, _sdpa_math_ctx

from abot_recon.modeling.streaming.core.chunk_attention import ChunkFlashAttentionRope
from abot_recon.modeling.streaming.core.flex_attention import (
    get_compiled_streaming_flex_attention,
    streaming_flex_attention,
)
from abot_recon.modeling.streaming.core.window_mask import streaming_chunk_attention_bias

from abot_recon.modeling.streaming.kv_state import StreamingKVState
from abot_recon.modeling.streaming.attention_bias import streaming_attention_bias
from abot_recon.modeling.streaming.rope3d import apply_rotary_emb


class StreamingFlashAttention(ChunkFlashAttentionRope):
    """Extends ``ChunkFlashAttentionRope`` with ``memory_mode=streaming`` and global RoPE modes."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer=None,
        rope=None,
        *,
        use_packaged_flash_attn: bool = False,
        memory_mode: str = "streaming",
        num_reference_frames: int = 0,
        local_window_frames: int = 12,
        num_summary_tokens: int = 0,
        global_pos_encoding: str = "pi3_2d",
        rope3d_embed: Optional[nn.Module] = None,
    ):
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            rope=rope,
            use_packaged_flash_attn=use_packaged_flash_attn,
        )
        self.memory_mode = str(memory_mode).lower()
        if self.memory_mode not in ("window", "streaming"):
            raise ValueError(f"memory_mode must be 'window' or 'streaming', got {memory_mode!r}")
        self.num_reference_frames = int(num_reference_frames)
        self.local_window_frames = int(local_window_frames)
        self.num_summary_tokens = int(num_summary_tokens)
        self.global_pos_encoding = str(global_pos_encoding).lower()
        if self.global_pos_encoding not in ("pi3_2d", "rope3d", "none"):
            raise ValueError(
                f"global_pos_encoding must be pi3_2d|rope3d|none, got {global_pos_encoding!r}"
            )
        self.rope3d_embed = rope3d_embed
        # Window-mode streaming mask: first n frame slots stay globally visible (see chunk_mask).
        self.streaming_num_reference_frames = int(num_reference_frames) if self.memory_mode == "window" else 0

    @classmethod
    def from_flash_attn(cls, old: FlashAttentionRope, **streaming_kw) -> "StreamingFlashAttention":
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
            **streaming_kw,
        )
        obj.load_state_dict(old.state_dict(), strict=True)
        return obj

    def compute_memory_attn_bias(
        self,
        *,
        t_past: int,
        num_new_frames: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return streaming_attention_bias(
            t_past,
            num_new_frames,
            tokens_per_frame,
            dtype,
            device,
            num_reference_frames=self.num_reference_frames,
            local_window_frames=self.local_window_frames,
            num_summary_tokens=self.num_summary_tokens,
        )

    def paged_stream_forward(
        self,
        x: Tensor,
        *,
        pos_q: Optional[Tensor],
        manager,
        layer_idx: int,
        frame_idx: int,
        image_hw: Tuple[int, int],
        patch_start_idx: int,
    ) -> Tensor:
        """Single-frame paged streaming attention (B=1, N=1).

        Pipeline per layer:
            qkv proj → q_norm/k_norm → RoPE (pi3_2d or rope3d per-frame)
            → manager.append_frame (POST-RoPE K) → manager.compute_attention
            → proj + proj_drop.

        Args:
            x: [1, tpf, C] -- pre-attn block input AFTER ``blk.norm1``.
            pos_q: [1, tpf, 2] for pi3_2d (ignored for rope3d).
            manager: PagedKVCacheManager.
            layer_idx: global-layer index (0..num_global_layers-1).
            frame_idx: current frame's global index ``g`` (0-based).
            image_hw: (H, W) of the input image for rope3d RoPE.
            patch_start_idx: number of special tokens at the front of each frame.

        Returns:
            [1, tpf, C] attention output (already proj'd; residual added outside).
        """
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"paged_stream_forward requires B=1, got x.shape={tuple(x.shape)}")
        B, tpf, C = x.shape
        if C != self.qkv.in_features:
            raise ValueError(f"x channel {C} != qkv.in_features {self.qkv.in_features}")

        num_heads = self.num_heads
        head_dim = C // num_heads

        # qkv proj → [B, tpf, 3, H, D] → [B, H, 3, tpf, D] → 3 × [B, H, tpf, D]
        qkv = self.qkv(x).reshape(B, tpf, 3, num_heads, head_dim).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]  # each [B=1, H, tpf, D]

        # qk-norm + dtype align (mirrors stream_chunk_forward)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # ── RoPE: applied PER FRAME at append time; Q at current frame ──────
        enc = self.global_pos_encoding
        if enc == "pi3_2d":
            if self.rope is not None:
                if pos_q is None:
                    raise ValueError("pi3_2d requires pos_q for paged_stream_forward")
                q = self.rope(q, pos_q)
                k = self.rope(k, pos_q)  # positions are frame-invariant for pi3_2d
        elif enc == "rope3d":
            if self.rope3d_embed is None:
                raise RuntimeError("rope3d requires rope3d_embed module")
            H_img, W_img = int(image_hw[0]), int(image_hw[1])
            patch = 14
            pph, ppw = H_img // patch, W_img // patch
            freqs_g = self.rope3d_embed(
                1, pph, ppw, patch_start_idx, q.device,
                f_start=int(frame_idx), f_end=int(frame_idx) + 1,
            )
            if freqs_g.shape[2] != tpf:
                raise RuntimeError(
                    f"rope3d freqs len {freqs_g.shape[2]} != tpf {tpf} "
                    f"(pph={pph} ppw={ppw} ps={patch_start_idx})"
                )
            q = apply_rotary_emb(q, freqs_g)
            k = apply_rotary_emb(k, freqs_g)
        elif enc == "none":
            pass
        else:
            raise RuntimeError(f"unsupported global_pos_encoding {enc!r}")

        # BHND [1, H, tpf, D] → NHD [tpf, H, D] for FlashInfer
        q_nhd = q.squeeze(0).transpose(0, 1).contiguous()
        k_nhd = k.squeeze(0).transpose(0, 1).contiguous()
        v_nhd = v.squeeze(0).transpose(0, 1).contiguous()

        # Store POST q_norm/k_norm/RoPE K,V and run paged attention
        manager.append_frame(layer_idx, k_nhd, v_nhd)
        out_nhd = manager.compute_attention(layer_idx, q_nhd)  # [tpf, H, D]

        # NHD [tpf, H, D] → flatten heads → [1, tpf, H*D] → proj
        out = out_nhd.reshape(1, tpf, num_heads * head_dim)
        out = self._apply_output_gate(x, out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def stream_chunk_forward(
        self,
        x: Tensor,
        pos_q: Tensor,
        past_key_values: Optional[Union[Tuple[Tensor, Tensor], StreamingKVState]],
        num_new_frames: int,
        chunk_key_positions: Tensor,
        temporal_window_frames: Optional[int] = None,
        kv_carry_meta: Optional[Dict[str, Any]] = None,
        cached_streaming_attn_bias: Optional[Tensor] = None,
        *,
        image_hw: Optional[Tuple[int, int]] = None,
        patch_start_idx: int = 5,
    ) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], StreamingKVState]]:
        if self.memory_mode == "window":
            tw = (
                int(temporal_window_frames)
                if temporal_window_frames is not None
                else int(self.local_window_frames)
            )
            return super().stream_chunk_forward(
                x,
                pos_q,
                past_key_values,
                num_new_frames,
                chunk_key_positions,
                temporal_window_frames=tw,
                kv_carry_meta=kv_carry_meta,
                cached_streaming_attn_bias=cached_streaming_attn_bias,
            )

        del kv_carry_meta
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

        hs_in = past_key_values if isinstance(past_key_values, StreamingKVState) else None
        if hs_in is not None:
            from abot_recon.modeling.streaming.sdpa import stream_window_attention

            del cached_streaming_attn_bias  # packed carry: visibility is physical, no bias tensor
            return stream_window_attention(
                self,
                x,
                pos_q=pos_q,
                hs_carry=hs_in,
                num_new_frames=int(num_new_frames),
                chunk_key_positions=chunk_key_positions,
                image_hw=image_hw,
                patch_start_idx=patch_start_idx,
            )

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

        q, k_flat = self.q_norm(q).to(v_flat.dtype), self.k_norm(k_flat).to(v_flat.dtype)
        v_flat = v_flat.to(q.dtype)

        enc = self.global_pos_encoding
        if enc == "rope3d":
            if self.rope3d_embed is None:
                raise RuntimeError("global_pos_encoding=rope3d requires rope3d_embed module")
            if image_hw is None:
                raise ValueError("image_hw (H, W) required for rope3d RoPE in stream_chunk_forward")
            H_img, W_img = int(image_hw[0]), int(image_hw[1])
            patch = 14
            pph, ppw = H_img // patch, W_img // patch
            ppf_frames = t_k
            freqs_full = self.rope3d_embed(
                ppf_frames, pph, ppw, patch_start_idx, q.device, f_start=0, f_end=None
            )
            if freqs_full.shape[2] != Lk:
                raise RuntimeError(
                    f"RoPE3D length {freqs_full.shape[2]} != Lk {Lk} "
                    f"(ppf={ppf_frames} tpf={tpf})"
                )
            start_q = t_past * tpf
            freqs_q = freqs_full[:, :, start_q : start_q + Lq, :]
            q = apply_rotary_emb(q, freqs_q)
            k_flat = apply_rotary_emb(k_flat, freqs_full)
        elif enc == "pi3_2d":
            if self.rope is not None:
                q = self.rope(q, pos_q)
                k_flat = self.rope(k_flat, chunk_key_positions)
        else:
            pass

        dropout_p = self.attn_drop.p if self.training else 0.0
        use_flex = bool(getattr(self, "use_chunk_flex_attention", False)) and q.is_cuda
        if use_flex:
            # Eager FlexAttention in PyTorch 2.6 materializes dense scores even
            # under no_grad; keep the compiled kernel during validation too.
            if bool(getattr(self, "use_chunk_flex_compile", False)):
                flex_module = get_compiled_streaming_flex_attention(
                    t_past,
                    t_new,
                    tpf,
                    self.num_reference_frames,
                    self.local_window_frames,
                    self.num_summary_tokens,
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
                    num_reference_frames=self.num_reference_frames,
                    local_window_frames=self.local_window_frames,
                    num_summary_tokens=self.num_summary_tokens,
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
                bias = self.compute_memory_attn_bias(
                    t_past=t_past,
                    num_new_frames=t_new,
                    tokens_per_frame=tpf,
                    dtype=q.dtype,
                    device=q.device,
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
        del kv_carry_meta
        if self.memory_mode == "streaming":
            return self.compute_memory_attn_bias(
                t_past=t_past,
                num_new_frames=num_new_frames,
                tokens_per_frame=tokens_per_frame,
                dtype=dtype,
                device=device,
            )
        t_new = int(num_new_frames)
        tpf = int(tokens_per_frame)
        W = (
            int(temporal_window_frames)
            if temporal_window_frames is not None
            else int(getattr(self, "local_window_frames", 12))
        )
        return streaming_chunk_attention_bias(
            t_past,
            t_new,
            tpf,
            dtype,
            device,
            temporal_window_frames=W,
            num_reference_frames=int(self.num_reference_frames),
        )
