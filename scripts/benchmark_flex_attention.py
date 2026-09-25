#!/usr/bin/env python3
"""Numerical parity and 128-frame training benchmark for the Flex backend."""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.nn.functional import scaled_dot_product_attention

from abot_recon.modeling.streaming.attention_bias import streaming_attention_bias
from abot_recon.modeling.streaming.core.flex_attention import (
    FLEX_ATTENTION_AVAILABLE,
    get_compiled_streaming_flex_attention,
)


def synchronize() -> None:
    torch.cuda.synchronize()


def parity(device: torch.device) -> dict[str, float]:
    # Match the model's BF16/head-dim/token-density kernel family. Very small
    # toy shapes select PyTorch's unrelated flex-decoding path instead.
    t_past, t_new, tpf = 0, 2, 725
    heads, head_dim = 2, 64
    q_shape = (1, heads, t_new * tpf, head_dim)
    k_shape = (1, heads, (t_past + t_new) * tpf, head_dim)
    generator = torch.Generator(device=device).manual_seed(11)
    source = [
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        for shape in (q_shape, k_shape, k_shape)
    ]
    upstream = torch.randn(q_shape, device=device, dtype=torch.bfloat16, generator=generator)

    qd, kd, vd = [value.clone().requires_grad_(True) for value in source]
    bias = streaming_attention_bias(
        t_past,
        t_new,
        tpf,
        qd.dtype,
        device,
        num_reference_frames=0,
        local_window_frames=3,
        num_summary_tokens=0,
    ).view(1, 1, q_shape[2], k_shape[2])
    dense = scaled_dot_product_attention(qd, kd, vd, attn_mask=bias)
    (dense * upstream).sum().backward()

    qf, kf, vf = [value.clone().requires_grad_(True) for value in source]
    module = get_compiled_streaming_flex_attention(t_past, t_new, tpf, 0, 3, 0)
    flex = module(qf, kf, vf)
    (flex * upstream).sum().backward()
    return {
        "forward_max_abs": float((flex - dense).abs().max()),
        "q_grad_max_abs": float((qf.grad - qd.grad).abs().max()),
        "k_grad_max_abs": float((kf.grad - kd.grad).abs().max()),
        "v_grad_max_abs": float((vf.grad - vd.grad).abs().max()),
    }


def benchmark(args, device: torch.device) -> dict[str, float | int | bool]:
    frames = int(args.frames)
    tpf = int(args.tokens_per_frame)
    shape = (1, int(args.heads), frames * tpf, int(args.head_dim))
    module = get_compiled_streaming_flex_attention(
        0, frames, tpf, 0, int(args.window), 0
    )

    def make_inputs():
        return [
            torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_(True)
            for _ in range(3)
        ]

    # Compile/warm up outside the reported steady-state measurement.
    q, k, v = make_inputs()
    warmup_start = time.perf_counter()
    module(q, k, v).float().square().mean().backward()
    synchronize()
    compile_and_warmup_sec = time.perf_counter() - warmup_start
    del q, k, v
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    q, k, v = make_inputs()
    synchronize()
    start = time.perf_counter()
    output = module(q, k, v)
    forward_end = time.perf_counter()
    loss = output.float().square().mean()
    loss.backward()
    synchronize()
    end = time.perf_counter()

    length = frames * tpf
    dense_bias_bytes = length * length * torch.tensor([], dtype=torch.bfloat16).element_size()
    finite = bool(
        torch.isfinite(output).all()
        and torch.isfinite(q.grad).all()
        and torch.isfinite(k.grad).all()
        and torch.isfinite(v.grad).all()
    )
    return {
        "frames": frames,
        "tokens_per_frame": tpf,
        "sequence_tokens": length,
        "window_frames": int(args.window),
        "compile_and_warmup_sec": compile_and_warmup_sec,
        "steady_forward_sec": forward_end - start,
        "steady_forward_backward_sec": end - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "dense_bf16_bias_gib_avoided": dense_bias_bytes / 2**30,
        "finite_output_and_gradients": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--tokens-per-frame", type=int, default=725)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("This benchmark requires CUDA FlexAttention")
    device = torch.device("cuda")
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "parity": parity(device),
        "benchmark": benchmark(args, device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
