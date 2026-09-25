from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ReconstructionResult:
    camera_poses: torch.Tensor
    relative_poses: torch.Tensor
    camera_poses_noloop: torch.Tensor
    relative_poses_noloop: torch.Tensor
    camera_poses_loop: torch.Tensor | None = None
    relative_poses_loop: torch.Tensor | None = None
    local_points: torch.Tensor | None = None
    world_points: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    confidence_mask: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None
