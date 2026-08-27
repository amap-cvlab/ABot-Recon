from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    """Stable inference configuration matching the released checkpoint."""

    checkpoint: Path = Path("checkpoints/abot_recon.safetensors")
    device: str = "cuda"
    amp_dtype: str = "bf16"
    height: int = 280
    width: int = 504
    local_window_frames: int = 12
    max_frames: int = 22_000
    attention_backend: str = "auto"
    output_points: bool = False
    output_confidence: bool = False
    loop_closure: bool = True
    loop_salad_checkpoint: Path = Path("checkpoints/loop/dino_salad.ckpt")
    loop_dino_checkpoint: Path = Path("checkpoints/loop/dinov2_vitb14_pretrain.pth")
    loop_output_dir: Path = Path("outputs/loop")

    def __post_init__(self) -> None:
        if self.amp_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("amp_dtype must be one of: fp32, fp16, bf16")
        if self.attention_backend not in {"auto", "paged", "sdpa"}:
            raise ValueError("attention_backend must be one of: auto, paged, sdpa")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.local_window_frames != 12:
            raise ValueError("The released model requires a 12-frame local window")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")

    def override(self, **values: Any) -> "InferenceConfig":
        """Return a validated copy with API/CLI overrides applied."""
        return replace(self, **values)
