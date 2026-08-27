"""Hydra compatibility wrapper around the released ABot-Recon runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from abot_recon.checkpoint import resolve_checkpoint
from abot_recon.config import InferenceConfig
from abot_recon.model import ReleasedABotReconModel


class ABotReconEval(nn.Module):
    """Expose ABot-Recon through the evaluation model interface."""

    supports_sparse_dense_output = True

    def __init__(
        self,
        ckpt: str = "checkpoints/abot_recon.safetensors",
        *,
        device: str = "cuda",
        height: int = 280,
        width: int = 504,
        fov_pad_rgb: Iterable[float] = (0.485, 0.456, 0.406),
        local_window_frames: int = 12,
        local_window_override: int | None = 12,
        infer_mode: str = "stream",
        attention_backend: str = "auto",
        amp_dtype: str = "bf16",
        max_frames: int = 22_000,
        num_frames_cap: int | None = 22_000,
    ) -> None:
        super().__init__()
        if str(infer_mode).lower() != "stream":
            raise ValueError("The released evaluator supports infer_mode='stream' only")
        if local_window_override not in (None, local_window_frames):
            raise ValueError("local_window_override must match local_window_frames")

        self.height = int(height)
        self.width = int(width)
        self.fov_pad_rgb = tuple(float(value) for value in fov_pad_rgb)
        self.local_window_override = int(local_window_frames)
        self.local_window_frames = int(local_window_frames)
        self.infer_mode = "stream"
        self.amp_dtype = str(amp_dtype).lower()
        self.num_frames_cap = int(num_frames_cap) if num_frames_cap is not None else None
        self.pretrained_model_name_or_path = str(resolve_checkpoint(ckpt))

        config = InferenceConfig(
            checkpoint=Path(ckpt),
            device=str(device),
            amp_dtype=self.amp_dtype,
            height=self.height,
            width=self.width,
            local_window_frames=self.local_window_frames,
            max_frames=int(max_frames),
            attention_backend=str(attention_backend),
            output_points=False,
            output_confidence=False,
            loop_closure=False,
        )
        self.runtime_model = ReleasedABotReconModel(config)
        self.attention_backend = self.runtime_model.attention_backend
        self.use_paged_kv = self.attention_backend == "paged"

    @property
    def model(self) -> nn.Module:
        """Checkpoint-exact network retained for existing evaluation adapters."""
        return self.runtime_model.network

    @torch.inference_mode()
    def inference_stream_iter(
        self,
        frames,
        *,
        num_frames=None,
        causal_global_attn=True,
        output_keys=None,
        dense_output_indices=None,
    ):
        self.runtime_model.reset()
        return self.model.inference_stream_iter(
            frames,
            num_frames=num_frames,
            causal_global_attn=causal_global_attn,
            output_keys=output_keys,
            dense_output_indices=dense_output_indices,
        )

    @torch.inference_mode()
    def inference_stream(self, images: torch.Tensor, causal_global_attn: bool = True):
        self.runtime_model.reset()
        return self.model.inference_stream(
            images,
            causal_global_attn=causal_global_attn,
        )

    def forward(self, images: torch.Tensor):
        if images.dim() == 4:
            images = images.unsqueeze(0)
        return self.inference_stream(images)
