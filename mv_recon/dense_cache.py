"""Lossless sparse-frame dense prediction cache for offline metric sweeps."""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np


def save_metric_frame_cache(
    path: str,
    *,
    pred_world: np.ndarray,
    pred_c2w: np.ndarray,
    pred_mask: Optional[np.ndarray],
    observation_mask: Optional[np.ndarray],
    metric_indices: Optional[Sequence[int]],
    metric_frame_ids: Optional[Sequence[int]],
    sequence_name: str,
    pred_depth: Optional[np.ndarray] = None,
    pred_local_points: Optional[np.ndarray] = None,
) -> None:
    """Atomically save exact float32 XYZ grids and their camera metadata."""
    points = np.asarray(pred_world, dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected MxHxWx3 points, got {points.shape}")
    count = len(points)
    full_c2w = np.asarray(pred_c2w, dtype=np.float64)
    if metric_indices is None:
        if len(full_c2w) != count:
            raise ValueError(f"Pose count {len(full_c2w)} != point count {count}")
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.asarray(metric_indices, dtype=np.int64).reshape(-1)
        if len(indices) != count:
            raise ValueError(f"Index count {len(indices)} != point count {count}")
        if np.any(indices < 0) or np.any(indices >= len(full_c2w)):
            raise ValueError("metric_indices outside full pose sequence")
    frame_ids = (
        indices.copy()
        if metric_frame_ids is None
        else np.asarray(metric_frame_ids, dtype=np.int64).reshape(-1)
    )
    if len(frame_ids) != count:
        raise ValueError(f"Frame-ID count {len(frame_ids)} != point count {count}")

    def mask_or_ones(value):
        if value is None:
            return np.ones(points.shape[:-1], dtype=bool)
        mask = np.asarray(value, dtype=bool)
        if mask.shape != points.shape[:-1]:
            raise ValueError(f"Mask shape {mask.shape} != {points.shape[:-1]}")
        return mask

    arrays = {
        "pred_world": points,
        "pred_c2w": full_c2w[indices],
        "pred_mask": mask_or_ones(pred_mask),
        "observation_mask": mask_or_ones(observation_mask),
        "metric_indices": indices,
        "metric_frame_ids": frame_ids,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "sequence": sequence_name,
                    "xyz_dtype": "float32",
                    "pose_dtype": "float64",
                    "compression": "none",
                    "layout": "metric_frame,height,width,xyz",
                },
                sort_keys=True,
            )
        ),
    }
    if pred_depth is not None:
        depth = np.asarray(pred_depth, dtype=np.float32)
        if depth.shape != points.shape[:-1]:
            raise ValueError(f"Depth shape {depth.shape} != {points.shape[:-1]}")
        arrays["pred_depth"] = depth
    if pred_local_points is not None:
        local = np.asarray(pred_local_points, dtype=np.float32)
        if local.shape != points.shape:
            raise ValueError(f"Local point shape {local.shape} != {points.shape}")
        arrays["pred_local_points"] = local
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
