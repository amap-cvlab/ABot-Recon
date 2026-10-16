# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# described in THIRD_PARTY_NOTICES.md.
#
# Adapted from ``lingbot-map/lingbot_map/layers/rope.py`` (RoPE3D, 3D RoPE).

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)]
                / dim
            )
        )
        / linear_factor
    )
    freqs = torch.outer(pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    if use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class RoPE3D(nn.Module):
    """3D RoPE: independent 1D rotary along frame / height / width, concatenated on head dim."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        fhw_dim: Optional[Tuple[int, int, int]] = None,
        *,
        rope_fwd_cache_max: int = 32,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        if fhw_dim is not None:
            t_dim, h_dim, w_dim = fhw_dim
            assert attention_head_dim == sum(fhw_dim), (
                f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}"
            )
            assert t_dim % 2 == 0 and h_dim % 2 == 0 and w_dim % 2 == 0, (
                f"fhw_dim must be all even for RoPE splits, got {fhw_dim}"
            )
        else:
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim

        self.fhw_dim = (t_dim, h_dim, w_dim)
        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.register_buffer("freqs", torch.cat(freqs, dim=1), persistent=False)
        # Per-(shape, device) cache for ``forward`` (many global blocks repeat the same call).
        self.rope_fwd_cache_max = max(0, int(rope_fwd_cache_max))
        self._freqs_fwd_cache: "OrderedDict[Any, torch.Tensor]" = OrderedDict()

    def clear_forward_freqs_cache(self) -> None:
        """Drop cached ``forward`` outputs (e.g. after resolution / device change in same process)."""
        self._freqs_fwd_cache.clear()

    def forward(
        self,
        ppf: int,
        pph: int,
        ppw: int,
        patch_start_idx: int,
        device: torch.device,
        f_start: int = 0,
        f_end: Optional[int] = None,
    ) -> torch.Tensor:
        key = (
            int(ppf),
            int(pph),
            int(ppw),
            int(patch_start_idx),
            str(device),
            int(f_start),
            int(f_end) if f_end is not None else None,
        )
        if self.rope_fwd_cache_max > 0 and key in self._freqs_fwd_cache:
            self._freqs_fwd_cache.move_to_end(key)
            return self._freqs_fwd_cache[key]

        out = self._freqs_fwd_no_cache(ppf, pph, ppw, patch_start_idx, device, f_start, f_end)

        if self.rope_fwd_cache_max > 0:
            self._freqs_fwd_cache[key] = out
            self._freqs_fwd_cache.move_to_end(key)
            while len(self._freqs_fwd_cache) > self.rope_fwd_cache_max:
                self._freqs_fwd_cache.popitem(last=False)

        return out

    def _freqs_fwd_no_cache(
        self,
        ppf: int,
        pph: int,
        ppw: int,
        patch_start_idx: int,
        device: torch.device,
        f_start: int,
        f_end: Optional[int],
    ) -> torch.Tensor:
        freqs_stored = self.freqs.to(device)
        t_dim, h_dim, w_dim = self.fhw_dim
        freqs = freqs_stored.split_with_sizes(
            [t_dim // 2, h_dim // 2, w_dim // 2],
            dim=1,
        )

        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            frame_slice = slice(0, ppf)

        if patch_start_idx > 0:
            freqs_special_f = (
                freqs[0][frame_slice].reshape(ppf, 1, -1).expand(ppf, patch_start_idx, -1)
            )
            freqs_special_h = (
                freqs[1][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)
            )
            freqs_special_w = (
                freqs[2][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)
            )
            freqs_special = torch.cat([freqs_special_f, freqs_special_h, freqs_special_w], dim=-1)
            freqs_special = freqs_special.reshape(ppf, patch_start_idx, -1)

            freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
            freqs_h = (
                freqs[1][patch_start_idx : patch_start_idx + pph]
                .reshape(1, pph, 1, -1)
                .expand(ppf, pph, ppw, -1)
            )
            freqs_w = (
                freqs[2][patch_start_idx : patch_start_idx + ppw]
                .reshape(1, 1, ppw, -1)
                .expand(ppf, pph, ppw, -1)
            )
            freqs_patches = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)
            freqs_patches = freqs_patches.reshape(ppf, pph * ppw, -1)

            out = torch.cat([freqs_special, freqs_patches], dim=1)
            out = out.reshape(ppf * (patch_start_idx + pph * ppw), -1)
            return out.unsqueeze(0).unsqueeze(0)

        freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        out = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return out


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    cos = freqs.real.to(x.dtype)
    sin = freqs.imag.to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).reshape(x.shape)
