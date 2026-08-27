"""Reusable post-alignment point-cloud metric caches.

The cache is written after Sim3/SE3, voxelization, and ICP. It preserves the
exact clouds scored by the evaluator and their two nearest-neighbor distance
arrays, so distance-threshold metrics can be swept without model inference or
another KD-tree query.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from mv_recon.lingbot_protocol import f1_from_pr, threshold_suffix


def metrics_from_distances(
    pred_to_gt: np.ndarray,
    gt_to_pred: np.ndarray,
    thresholds: Sequence[float],
    primary_threshold: float,
) -> Dict[str, float]:
    """Recompute Acc/Comp/CD and P/R/F1 from cached NN distances."""
    pred_dist = np.asarray(pred_to_gt, dtype=np.float64).reshape(-1)
    gt_dist = np.asarray(gt_to_pred, dtype=np.float64).reshape(-1)
    values = sorted(set(float(value) for value in thresholds) | {float(primary_threshold)})
    if not len(pred_dist) or not len(gt_dist):
        raise ValueError("Point-cloud distance arrays must be non-empty")
    if any(value <= 0 for value in values):
        raise ValueError(f"Thresholds must be positive, got {values}")

    accuracy = float(np.mean(pred_dist))
    completeness = float(np.mean(gt_dist))
    result: Dict[str, float] = {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": (accuracy + completeness) / 2.0,
        "num_pred": int(len(pred_dist)),
        "num_gt": int(len(gt_dist)),
        "eval_threshold": float(primary_threshold),
    }
    for threshold in values:
        precision = float(np.mean(pred_dist < threshold) * 100.0)
        recall = float(np.mean(gt_dist < threshold) * 100.0)
        f1 = f1_from_pr([precision], [recall])[0]
        suffix = threshold_suffix(threshold)
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"f1_{suffix}"] = f1
    primary = threshold_suffix(primary_threshold)
    result["precision"] = result[f"precision_{primary}"]
    result["recall"] = result[f"recall_{primary}"]
    result["f1"] = result[f"f1_{primary}"]
    return result


def save_pc_metric_cache(
    path: str,
    *,
    metrics: Dict,
    sequence_name: str,
    dataset_name: str,
    model_name: str,
) -> None:
    """Atomically save scored clouds, NN distances, transforms, and protocol."""
    pred = np.asarray(metrics["pred_points"])
    gt = np.asarray(metrics["gt_points"])
    if pred.ndim != 2 or pred.shape[1] < 4:
        raise ValueError(f"Expected pred_points Nx4 [xyz, distance], got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] < 4:
        raise ValueError(f"Expected gt_points Nx4 [xyz, distance], got {gt.shape}")

    metadata = {
        "format_version": 1,
        "sequence": sequence_name,
        "dataset": dataset_name,
        "model": model_name,
        "stage": "post_alignment_post_voxel_post_icp",
        "xyz_dtype": "float32",
        "distance_dtype": "float32",
        "eval_threshold": float(metrics["eval_threshold"]),
        "eval_thresholds": [float(v) for v in metrics["eval_thresholds"]],
        "icp_threshold": float(metrics["icp_threshold"]),
        "voxel_size": float(metrics["voxel_size"]),
        "with_scale": bool(metrics["with_scale"]),
        "num_pred": int(metrics["num_pred"]),
        "num_gt": int(metrics["num_gt"]),
    }
    arrays = {
        "pred_xyz": np.asarray(pred[:, :3], dtype=np.float32),
        "gt_xyz": np.asarray(gt[:, :3], dtype=np.float32),
        "pred_to_gt_dist": np.asarray(pred[:, 3], dtype=np.float32),
        "gt_to_pred_dist": np.asarray(gt[:, 3], dtype=np.float32),
        "T_umeyama": np.asarray(metrics["T_umeyama"], dtype=np.float64),
        "T_icp": np.asarray(metrics["T_icp"], dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_pc_metric_cache(path: str) -> Dict:
    """Load one cache and validate its count metadata."""
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.array(data[name], copy=True) for name in data.files}
    result["metadata"] = json.loads(str(result.pop("metadata_json")))
    if len(result["pred_xyz"]) != len(result["pred_to_gt_dist"]):
        raise ValueError(f"Prediction cache count mismatch: {path}")
    if len(result["gt_xyz"]) != len(result["gt_to_pred_dist"]):
        raise ValueError(f"GT cache count mismatch: {path}")
    return result


def recompute_cached_metrics(
    path: str,
    thresholds: Sequence[float],
    primary_threshold: float,
) -> Dict[str, float]:
    cache = load_pc_metric_cache(path)
    return metrics_from_distances(
        cache["pred_to_gt_dist"],
        cache["gt_to_pred_dist"],
        thresholds,
        primary_threshold,
    )

