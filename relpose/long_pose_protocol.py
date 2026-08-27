"""Shared safety checks for long-sequence pose-only evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def bind_dataset_context(model: Any, dataset_name: str, dataset_info: Any) -> None:
    """Expose dataset identity to adapters that have official per-dataset RGB transforms."""
    model.eval_dataset_name = str(dataset_name)
    model.eval_dataset_info = dataset_info


def validate_predicted_poses(poses: Any, expected_frames: int) -> torch.Tensor:
    tensor = torch.as_tensor(poses).detach().cpu()
    if tensor.ndim != 3 or tensor.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Expected poses [N,3|4,4], got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(expected_frames):
        raise ValueError(
            f"Pose/frame count mismatch: poses={tensor.shape[0]}, images={expected_frames}"
        )
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum().item())
        raise ValueError(f"Predicted poses contain {bad} non-finite values")
    rotations = tensor[:, :3, :3].double()
    determinants = torch.linalg.det(rotations)
    if torch.any(torch.abs(determinants - 1.0) > 5e-2):
        worst = float(torch.max(torch.abs(determinants - 1.0)).item())
        raise ValueError(f"Predicted rotation determinant is invalid; max |det-1|={worst:.6g}")
    return tensor.float()


def load_resumable_poses(
    sequence_dir: str | Path,
    expected_frames: int,
    *,
    enabled: bool,
) -> torch.Tensor | None:
    path = Path(sequence_dir) / "pred_poses.npy"
    if not enabled or not path.is_file():
        return None
    return validate_predicted_poses(np.load(path), expected_frames)


def trajectory_length(trajectory: Any) -> int:
    """Return length for either ``(poses, timestamps)`` or evo trajectories."""
    if isinstance(trajectory, (tuple, list)):
        if not trajectory:
            return 0
        return len(trajectory[0])
    if hasattr(trajectory, "positions_xyz"):
        return len(trajectory.positions_xyz)
    raise TypeError(f"Unsupported trajectory container: {type(trajectory)!r}")
