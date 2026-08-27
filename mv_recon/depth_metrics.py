"""Scale-aligned monocular depth metrics for sparse reconstruction frames."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


DEPTH_METRIC_KEYS = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "delta1",
    "delta2",
    "delta3",
)


def _select_poses(
    poses: np.ndarray,
    metric_indices: Optional[Sequence[int]],
    metric_count: int,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape == (metric_count, 4, 4):
        return poses
    if metric_indices is None:
        raise ValueError(
            f"Pose count {len(poses)} does not match metric count {metric_count}"
        )
    indices = np.asarray(metric_indices, dtype=np.int64).reshape(-1)
    if len(indices) != metric_count:
        raise ValueError(
            f"metric_indices count {len(indices)} != metric count {metric_count}"
        )
    if np.any(indices < 0) or np.any(indices >= len(poses)):
        raise ValueError("metric_indices are outside the full pose sequence")
    return poses[indices]


def camera_z_from_world(world_xyz: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Transform an ``M x H x W x 3`` world point map and return camera Z."""
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    if world_xyz.ndim != 4 or world_xyz.shape[-1] != 3:
        raise ValueError(f"Expected MxHxWx3 points, got {world_xyz.shape}")
    if w2c.shape != (len(world_xyz), 4, 4):
        raise ValueError(f"Expected {(len(world_xyz), 4, 4)} poses, got {w2c.shape}")
    flat = world_xyz.reshape(len(world_xyz), -1, 3)
    camera = np.einsum("bij,bkj->bki", w2c[:, :3, :3], flat)
    camera += w2c[:, None, :3, 3]
    return camera[..., 2].reshape(world_xyz.shape[:-1])


def evaluate_scale_aligned_depth(
    *,
    pred_world: np.ndarray,
    pred_c2w: np.ndarray,
    gt_world: np.ndarray,
    gt_w2c: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    observation_mask: Optional[np.ndarray] = None,
    metric_indices: Optional[Sequence[int]] = None,
) -> Mapping[str, float]:
    """Evaluate local depth after one robust median scale per sequence.

    Inference remains stride-1. ``metric_indices`` only selects the poses that
    correspond to sparse GT depth frames (for Oxford: 0, 10, 20, ...).
    """
    pred_world = np.asarray(pred_world)
    gt_world = np.asarray(gt_world)
    if pred_world.shape != gt_world.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred_world.shape} vs {gt_world.shape}")
    metric_count = len(pred_world)
    pred_c2w_metric = _select_poses(pred_c2w, metric_indices, metric_count)
    pred_w2c = np.linalg.inv(pred_c2w_metric)
    pred_depth = camera_z_from_world(pred_world, pred_w2c)
    gt_depth = camera_z_from_world(gt_world, np.asarray(gt_w2c, dtype=np.float64))

    valid = np.asarray(gt_mask, dtype=bool).copy()
    valid &= np.isfinite(gt_depth) & np.isfinite(pred_depth)
    valid &= (gt_depth > 1e-4) & (pred_depth > 1e-4)
    if pred_mask is not None:
        valid &= np.asarray(pred_mask, dtype=bool)
    if observation_mask is not None:
        valid &= np.asarray(observation_mask, dtype=bool)
    count = int(valid.sum())
    if count < 100:
        raise ValueError(f"Too few common valid depth pixels: {count}")

    gt = gt_depth[valid]
    pred = pred_depth[valid]
    scale = float(np.median(gt / pred))
    pred = pred * scale
    residual = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    metrics = {
        "depth_scale": scale,
        "num_depth_pixels": count,
        "abs_rel": float(np.mean(np.abs(residual) / gt)),
        "sq_rel": float(np.mean(residual * residual / gt)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25) * 100.0),
        "delta2": float(np.mean(ratio < 1.25**2) * 100.0),
        "delta3": float(np.mean(ratio < 1.25**3) * 100.0),
    }
    return metrics


def evaluate_scale_aligned_depth_maps(
    *,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    observation_mask: Optional[np.ndarray] = None,
    alignment_gt_depth_max: Optional[float] = None,
) -> Mapping[str, float]:
    """Evaluate depth using a near-range scale and the full evaluation mask."""
    pred_depth = np.asarray(pred_depth, dtype=np.float64)
    gt_depth = np.asarray(gt_depth, dtype=np.float64)
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"Prediction/GT depth mismatch: {pred_depth.shape} vs {gt_depth.shape}")
    valid = np.asarray(gt_mask, dtype=bool).copy()
    valid &= np.isfinite(gt_depth) & np.isfinite(pred_depth)
    valid &= (gt_depth > 1e-4) & (pred_depth > 1e-4)
    if pred_mask is not None:
        valid &= np.asarray(pred_mask, dtype=bool)
    if observation_mask is not None:
        valid &= np.asarray(observation_mask, dtype=bool)
    count = int(valid.sum())
    if count < 100:
        raise ValueError(f"Too few common valid depth pixels: {count}")
    scale_valid = valid.copy()
    if alignment_gt_depth_max is not None:
        scale_valid &= gt_depth <= float(alignment_gt_depth_max)
    scale_count = int(scale_valid.sum())
    if scale_count < 100:
        raise ValueError(f"Too few depth pixels for robust scale: {scale_count}")
    scale = float(np.median(gt_depth[scale_valid] / pred_depth[scale_valid]))
    gt = gt_depth[valid]
    pred = pred_depth[valid]
    pred = pred * scale
    residual = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "depth_scale": scale,
        "num_depth_pixels": count,
        "num_scale_pixels": scale_count,
        "scale_gt_depth_max": (
            np.nan if alignment_gt_depth_max is None else float(alignment_gt_depth_max)
        ),
        "abs_rel": float(np.mean(np.abs(residual) / gt)),
        "sq_rel": float(np.mean(residual * residual / gt)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25) * 100.0),
        "delta2": float(np.mean(ratio < 1.25**2) * 100.0),
        "delta3": float(np.mean(ratio < 1.25**3) * 100.0),
    }
