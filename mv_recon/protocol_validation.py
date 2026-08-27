"""Fail-fast checks for the locked point-cloud evaluation protocol."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np


_INDOOR_INPUT_HW = {
    "abot_recon": (280, 504),
    "lingbot_map": (434, 574),
    "horizonstream": (378, 518),
    "cut3r": (384, 512),
    "ttt3r": (384, 512),
    "longstream": (378, 518),
    "infinitevggt": (392, 518),
    "ovggt": (392, 518),
    "stream3r_window5": (392, 518),
}

_OXFORD_INPUT_HW = {
    **_INDOOR_INPUT_HW,
    "lingbot_map": (378, 518),
}


def _model_family(model_name: str) -> str:
    if model_name in {"horizonstream", "horizonstream_sim3", "horizonstream_se3"}:
        return "horizonstream"
    return model_name


def _expected_input_hw(model_name: str, dataset_name: str):
    model_name = _model_family(model_name)
    if dataset_name == "Oxford-Spires-S1-I10":
        return _OXFORD_INPUT_HW.get(model_name)
    if dataset_name in {
        "7scenes-hs-paper",
        "TUM-Dynamics-Full",
    }:
        return _INDOOR_INPUT_HW.get(model_name)
    return None


def _validate_rotations(pred_c2w: Optional[np.ndarray]) -> None:
    if pred_c2w is None:
        raise RuntimeError("Formal point-cloud protocol requires predicted camera poses")
    poses = np.asarray(pred_c2w)
    if poses.ndim != 3 or poses.shape[-2:] not in {(3, 4), (4, 4)}:
        raise RuntimeError(f"Invalid predicted pose shape: {poses.shape}")
    rotations = poses[:, :3, :3]
    if not np.isfinite(rotations).all():
        raise RuntimeError("Predicted rotations contain NaN/Inf")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=5e-2, rtol=0.0):
        raise RuntimeError(
            "Predicted rotations are not proper SO(3): "
            f"det range=[{determinants.min():.6f},{determinants.max():.6f}]"
        )


def validate_formal_pointcloud_protocol(
    *,
    model_name: str,
    dataset_name: str,
    runtime: Mapping[str, Any],
    pred_c2w: Optional[np.ndarray],
    with_scale: bool,
    metric_frame_ids: Optional[Sequence[int]],
    eval_threshold: Optional[float] = None,
    eval_thresholds: Optional[Sequence[float]] = None,
    voxel_size: Optional[float] = None,
    icp_threshold: Optional[float] = None,
    nearest_depth_to_gt: Optional[bool] = None,
    pointmap_resize_mode: Optional[str] = None,
    alignment_depth_max: Optional[float] = None,
    observation_mask: Optional[np.ndarray] = None,
) -> None:
    """Reject a formal run that diverges from the documented protocol."""
    model_family = _model_family(model_name)
    if not with_scale:
        raise RuntimeError("Formal cross-model point-cloud evaluation requires Sim3")

    expected_hw = _expected_input_hw(model_name, dataset_name)
    actual_hw = (int(runtime["input_h"]), int(runtime["input_w"]))
    if expected_hw is not None and actual_hw != expected_hw:
        raise RuntimeError(
            f"Formal input shape mismatch for {model_name}/{dataset_name}: "
            f"actual={actual_hw}, expected={expected_hw}"
        )

    compute_dtype = str(runtime["forward_compute_dtype"]).lower()
    if model_name in {"cut3r", "ttt3r"} and compute_dtype != "fp32":
        raise RuntimeError(
            f"{model_name} formal point-cloud forward must be fp32, got {compute_dtype}"
        )
    if model_family == "horizonstream" and compute_dtype not in {"fp16", "float16"}:
        raise RuntimeError(
            f"Horizon formal point-cloud forward must be fp16, got {compute_dtype}"
        )

    if model_name == "abot_recon":
        online_state = str(runtime.get("online_state", "")).lower()
        if "paged-kv-true" not in online_state:
            raise RuntimeError(
                "Formal ABot-Recon point-cloud forward requires paged KV; "
                f"runtime online_state={online_state!r}"
            )

    if nearest_depth_to_gt is not None and not bool(nearest_depth_to_gt):
        raise RuntimeError("Formal point-cloud protocol requires nearest depth resize")
    if pointmap_resize_mode is not None and str(pointmap_resize_mode).lower() != "nearest":
        raise RuntimeError(
            "Formal point-cloud protocol requires nearest XYZ resize, got "
            f"{pointmap_resize_mode!r}"
        )
    expected_alignment_max = None
    if model_family == "horizonstream":
        expected_alignment_max = (
            80.0 if dataset_name == "Oxford-Spires-S1-I10" else 40.0
        )
    if expected_alignment_max is None:
        if alignment_depth_max is not None:
            raise RuntimeError(
                f"Formal {model_name} alignment must not use a prediction-depth "
                f"cutoff, got {alignment_depth_max}"
            )
    elif alignment_depth_max is None or not np.isclose(
        float(alignment_depth_max), expected_alignment_max
    ):
        raise RuntimeError(
            f"Formal Horizon alignment depth cutoff mismatch for {dataset_name}: "
            f"actual={alignment_depth_max}, expected={expected_alignment_max}"
        )

    if model_name in {
        "abot_recon",
        "horizonstream",
        "horizonstream_sim3",
        "horizonstream_se3",
    } and observation_mask is None:
        raise RuntimeError(
            f"Formal {model_name} evaluation requires an observed-FOV mask"
        )

    expected_primary_threshold = (
        4.0 if dataset_name == "Oxford-Spires-S1-I10" else None
    )
    if expected_primary_threshold is not None and (
        eval_threshold is None
        or not np.isclose(float(eval_threshold), expected_primary_threshold)
    ):
        raise RuntimeError(
            "Formal Oxford primary F1 threshold must be 4.0 m, "
            f"got {eval_threshold}"
        )
    expected_thresholds = (
        [2.0, 4.0]
        if dataset_name == "Oxford-Spires-S1-I10"
        else [0.05, 0.25]
    )
    if eval_thresholds is not None:
        actual_thresholds = list(eval_thresholds)
        if len(actual_thresholds) != len(expected_thresholds) or not np.allclose(
            np.asarray(actual_thresholds, dtype=np.float64),
            np.asarray(expected_thresholds, dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Formal F1 thresholds mismatch for {dataset_name}: "
                f"actual={actual_thresholds}, expected={expected_thresholds}"
            )
    expected_voxel = 0.05 if dataset_name == "Oxford-Spires-S1-I10" else 4.0 / 512.0
    expected_icp = 0.5 if dataset_name == "Oxford-Spires-S1-I10" else 0.1
    if voxel_size is not None and not np.isclose(voxel_size, expected_voxel):
        raise RuntimeError(
            f"Formal voxel mismatch for {dataset_name}: "
            f"actual={voxel_size}, expected={expected_voxel}"
        )
    if icp_threshold is not None and not np.isclose(icp_threshold, expected_icp):
        raise RuntimeError(
            f"Formal ICP threshold mismatch for {dataset_name}: "
            f"actual={icp_threshold}, expected={expected_icp}"
        )

    if dataset_name == "Oxford-Spires-S1-I10":
        ids = np.asarray(metric_frame_ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            raise RuntimeError("Oxford formal run has no metric frame IDs")
        if ids[0] != 0 or np.any(np.diff(ids) != 10):
            raise RuntimeError(
                "Oxford metric frame IDs must be exactly 0,10,20,...; "
                f"got first={ids[:5].tolist()}, last={ids[-5:].tolist()}"
            )

    _validate_rotations(pred_c2w)
