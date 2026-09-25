# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

from . import curope as _kernels  # run `python setup.py build_ext --inplace`


class cuRoPE2D_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        _kernels.rope_2d(tokens, positions, base, F0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        # CUDA kernel expects layout (B, N, H, D) with stride(2)==D, stride(3)==1
        g = grad_res.contiguous()
        _kernels.rope_2d(g, positions, base, -F0)
        ctx.mark_dirty(g)
        return g, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0

    def forward(self, tokens, positions):
        # Attention passes (B, nheads, ntokens, dim); kernel expects (B, N, H, D) contiguous.
        # A plain transpose(1, 2) does not satisfy stride(2)==D when ntokens>1.
        x = tokens.transpose(1, 2).contiguous()
        x = cuRoPE2D_func.apply(x, positions, self.base, self.F0)
        return x.transpose(1, 2).contiguous()
