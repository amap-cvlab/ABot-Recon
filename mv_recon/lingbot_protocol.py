"""Lingbot-map aligned point-cloud evaluation protocol for mv_recon.

Matches ``lingbot-map/benchmark``:
  - Umeyama on GT ∩ pred pixel correspondences
  - Optional voxel downsample (indoor: 4/512 m)
  - Open3D point-to-point ICP (max_iter=20)
  - Metrics: Accuracy / Completeness (mean NN), Chamfer, Precision / Recall / F1

Reference:
  - lingbot-map/benchmark/benchmark/evaluation/points.py
  - lingbot-map/benchmark/benchmark/geometry/registration.py
  - lingbot-map/benchmark/datasets/seven_scenes.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

# Indoor default used by LingBot 7-Scenes.
DEFAULT_VOXEL_SIZE = 4.0 / 512.0  # ≈ 0.0078125 m
DEFAULT_ICP_THRESHOLD = 0.1
DEFAULT_EVAL_THRESHOLD_INDOOR = 0.05
DEFAULT_EVAL_THRESHOLD_TUM = 0.25  # Horizon Table 4 reports TUM F1@0.25 m
DEFAULT_VOXEL_SIZE_OXFORD = 0.05
DEFAULT_ICP_THRESHOLD_OXFORD = 0.5
DEFAULT_EVAL_THRESHOLD_OXFORD = 4.0


# ---------------------------------------------------------------------------
# Dataset options
# ---------------------------------------------------------------------------

def get_dataset_eval_options(dataset_name: str) -> Dict[str, float]:
    """Return ICP / voxel / F1-threshold options for a mv_recon dataset key.

    Args:
        dataset_name: one of the formal 7Scenes, TUM, or Oxford dataset keys.

    Returns:
        Dict with keys ``icp_threshold``, ``voxel_size``, ``eval_threshold``.
        ``voxel_size <= 0`` disables voxel downsampling.
    """
    name = dataset_name.lower()
    if "tum" in name:
        return {
            "icp_threshold": DEFAULT_ICP_THRESHOLD,
            "voxel_size": DEFAULT_VOXEL_SIZE,
            "eval_threshold": DEFAULT_EVAL_THRESHOLD_TUM,
        }
    if "oxford" in name:
        # Outdoor TLS map. 5 cm matches Oxford/LingBot map preparation and
        # prevents tens of millions of near-duplicate TLS samples dominating
        # KD-tree metrics. The final point metrics keep all valid model depths.
        return {
            "icp_threshold": DEFAULT_ICP_THRESHOLD_OXFORD,
            "voxel_size": DEFAULT_VOXEL_SIZE_OXFORD,
            "eval_threshold": DEFAULT_EVAL_THRESHOLD_OXFORD,
        }
    # Horizon Table-4 style 7Scenes diagnostic: indoor voxel/ICP, but F1@0.25.
    if "hs-paper" in name or "horizon_seq01" in name or name.endswith("f1t025"):
        return {
            "icp_threshold": DEFAULT_ICP_THRESHOLD,
            "voxel_size": DEFAULT_VOXEL_SIZE,
            "eval_threshold": DEFAULT_EVAL_THRESHOLD_TUM,
        }
    # 7Scenes/default indoor.
    return {
        "icp_threshold": DEFAULT_ICP_THRESHOLD,
        "voxel_size": DEFAULT_VOXEL_SIZE,
        "eval_threshold": DEFAULT_EVAL_THRESHOLD_INDOOR,
    }


def resolve_dataset_eval_options(
    dataset_name: str,
    eval_threshold_override: Optional[float] = None,
) -> Dict[str, float]:
    """Return dataset defaults with an optional positive F1 threshold override.

    Kept as the public override-aware companion to
    :func:`get_dataset_eval_options`; several evaluation tests and launchers
    import this name directly.
    """
    options = dict(get_dataset_eval_options(dataset_name))
    if eval_threshold_override is None:
        return options
    threshold = float(eval_threshold_override)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"eval_threshold_override must be finite and > 0, got "
            f"{eval_threshold_override!r}"
        )
    options["eval_threshold"] = threshold
    return options


def resolve_pc_gt_load_img_size(
    dataset_name: str,
    nearest_depth_to_gt: bool,
    configured_load_img_size: Optional[int],
) -> Optional[int]:
    """GT resize width used when ``nearest_depth_to_gt`` is enabled.

    Matches official lingbot-map BSS prepare:
      - 7Scenes / TUM: native sensor resolution (``0`` = do not resize GT)
      - others: keep configured value

    When ``nearest_depth_to_gt`` is False, returns ``configured_load_img_size`` unchanged.
    """
    if not nearest_depth_to_gt:
        return configured_load_img_size

    name = dataset_name.lower()
    if ("7scenes" in name) or ("tum" in name):
        return 0
    return configured_load_img_size


def resolve_lingbot_prepare_width(dataset_name: str) -> int:
    """Width for official BSS *prepare* step before area_budget inference.

    Oxford's official loader prepares images at width 518 (height floor-to-14)
    before applying the area budget. Indoor datasets use native resolution.
    """
    name = dataset_name.lower()
    if "oxford" in name:
        return 518
    return 0


# ---------------------------------------------------------------------------
# Registration helpers (lingbot-compatible)
# ---------------------------------------------------------------------------

def apply_transform(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to xyz (extra channels preserved)."""
    original_shape = points.shape
    xyz = points[..., :3]
    xyz_flat = xyz.reshape(-1, 3)
    ones = np.ones((xyz_flat.shape[0], 1), dtype=xyz_flat.dtype)
    xyz_h = np.hstack([xyz_flat, ones])
    xyz_t = (xyz_h @ transformation.T)[:, :3]
    xyz_t = xyz_t.reshape(original_shape[:-1] + (3,))
    if original_shape[-1] > 3:
        return np.concatenate([xyz_t, points[..., 3:]], axis=-1)
    return xyz_t


def umeyama_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    with_scale: bool = True,
) -> np.ndarray:
    """Umeyama alignment: 4x4 matrix mapping source → target.

    ``with_scale=True`` → Sim(3); ``False`` → SE(3) (scale fixed to 1, keep
    model metric scale — used for HorizonStream).

    Requires one-to-one correspondences (same length / order).
    """
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Source and target must have same shape, got "
            f"{source_points.shape} and {target_points.shape}"
        )
    source_xyz = source_points[..., :3].reshape(-1, 3).astype(np.float64)
    target_xyz = target_points[..., :3].reshape(-1, 3).astype(np.float64)

    X = source_xyz.T  # (3, N)
    Y = target_xyz.T
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    X_c = X - mu_x
    Y_c = Y - mu_y
    var_x = np.square(X_c).sum() / X.shape[1]
    if var_x < 1e-18:
        raise ValueError("Umeyama: source variance is zero")
    cov_xy = (Y_c @ X_c.T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2, 2] = -1
    R = U @ S @ VH
    if with_scale:
        c = float(np.trace(np.diag(D) @ S) / var_x)
    else:
        c = 1.0
    t = mu_y - c * R @ mu_x

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = c * R
    T[:3, 3:4] = t
    return T


# Models that predict metric (absolute) scale: align with SE(3) only.
METRIC_SCALE_MODELS = frozenset({"horizonstream"})


def resolve_pc_align_with_scale(
    model_keyname: str,
    cfg_value: Optional[bool] = None,
) -> bool:
    """Whether Umeyama may estimate scale (Sim3) vs rigid-only SE3.

    Default: Sim3 for up-to-scale models (lingbot / HybridLong). Metric-scale
    models (HorizonStream) keep model scale → SE3. Explicit config overrides.
    """
    if cfg_value is not None:
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_none(cfg_value):
                cfg_value = None
        except Exception:
            pass
    if cfg_value is not None:
        return bool(cfg_value)
    return model_keyname not in METRIC_SCALE_MODELS


def icp_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    icp_threshold: float = DEFAULT_ICP_THRESHOLD,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Open3D point-to-point ICP; returns 4x4 transform (source → target)."""
    source_xyz = np.asarray(source_points[..., :3].reshape(-1, 3), dtype=np.float64)
    target_xyz = np.asarray(target_points[..., :3].reshape(-1, 3), dtype=np.float64)
    if source_xyz.shape[0] < 3 or target_xyz.shape[0] < 3:
        raise ValueError(
            f"ICP needs >=3 points, got source={source_xyz.shape[0]}, "
            f"target={target_xyz.shape[0]}"
        )
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_xyz)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_xyz)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=tolerance,
        relative_rmse=tolerance,
        max_iteration=max_iterations,
    )
    result = o3d.pipelines.registration.registration_icp(
        source=src,
        target=tgt,
        max_correspondence_distance=icp_threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria,
    )
    return np.asarray(result.transformation, dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsample; preserves rgb if present (N,6)."""
    if voxel_size <= 0:
        return np.asarray(points, dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6].astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel_size)
    xyz = np.asarray(pcd_down.points)
    if points.shape[1] >= 6:
        rgb = np.asarray(pcd_down.colors)
        return np.hstack([xyz, rgb]).astype(np.float32)
    return xyz.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics (lingbot-compatible: mean Acc/Comp + Chamfer + P/R/F1)
# ---------------------------------------------------------------------------

def nearest_neighbor_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Per-point NN distances from source → target (xyz only)."""
    source_xyz = np.asarray(source_points[..., :3].reshape(-1, 3), dtype=np.float64)
    target_xyz = np.asarray(target_points[..., :3].reshape(-1, 3), dtype=np.float64)
    tree = KDTree(target_xyz)
    dist, _ = tree.query(source_xyz, workers=-1)
    return dist


def accuracy(source_points: np.ndarray, target_points: np.ndarray) -> float:
    """Mean NN distance pred → GT (Accuracy)."""
    return float(np.mean(nearest_neighbor_distances(source_points, target_points)))


def completeness(source_points: np.ndarray, target_points: np.ndarray) -> float:
    """Mean NN distance GT → pred (Completeness)."""
    return float(np.mean(nearest_neighbor_distances(target_points, source_points)))


def restrict_masks_to_observed_fov(
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray],
    observation_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Restrict both metric directions to pixels actually seen by the model.

    The returned GT mask controls the GT cloud used by Completeness, while the
    returned prediction mask controls correspondences and Accuracy. Inputs are
    never mutated, which also keeps raw-valid diagnostics trustworthy.
    """
    eval_gt_mask = np.array(gt_mask, dtype=bool, copy=True)
    eval_pred_mask = (
        None if pred_mask is None else np.array(pred_mask, dtype=bool, copy=True)
    )
    if observation_mask is None:
        return eval_gt_mask, eval_pred_mask

    observed = np.asarray(observation_mask, dtype=bool)
    if observed.shape != eval_gt_mask.shape:
        raise ValueError(
            f"Observation mask shape {observed.shape} != "
            f"GT mask shape {eval_gt_mask.shape}"
        )
    eval_gt_mask &= observed
    eval_pred_mask = (
        observed.copy()
        if eval_pred_mask is None
        else eval_pred_mask & observed
    )
    return eval_gt_mask, eval_pred_mask


def chamfer_distance(acc: float, comp: float) -> float:
    return (acc + comp) / 2.0


def precision_at_thresholds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    thresholds: Sequence[float],
) -> List[float]:
    dist = nearest_neighbor_distances(source_points, target_points)
    return [float(np.mean(dist < t) * 100.0) for t in thresholds]


def recall_at_thresholds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    thresholds: Sequence[float],
) -> List[float]:
    dist = nearest_neighbor_distances(target_points, source_points)
    return [float(np.mean(dist < t) * 100.0) for t in thresholds]


def f1_from_pr(precision: Sequence[float], recall: Sequence[float]) -> List[float]:
    out = []
    for p, r in zip(precision, recall):
        if p + r > 0:
            out.append(float(2.0 * p * r / (p + r)))
        else:
            out.append(0.0)
    return out


def threshold_suffix(threshold: float) -> str:
    """Stable CSV-key suffix for a metric distance threshold."""
    return f"{float(threshold):.4f}".rstrip("0").rstrip(".")


def normalize_eval_thresholds(
    primary_threshold: float,
    thresholds: Optional[Sequence[float]] = None,
) -> List[float]:
    """Return unique positive thresholds and always include the primary one."""
    values = [float(primary_threshold)] if thresholds is None else [
        float(value) for value in thresholds
    ]
    values.append(float(primary_threshold))
    if any(value <= 0 for value in values):
        raise ValueError(f"Evaluation thresholds must be positive, got {values}")
    return sorted(set(values))


def threshold_metric_keys(thresholds: Sequence[float]) -> Tuple[str, ...]:
    """Formal threshold columns: F1 only; P/R remain internal intermediates."""
    keys = []
    for threshold in thresholds:
        suffix = threshold_suffix(threshold)
        keys.append(f"f1_{suffix}")
    return tuple(keys)


def evaluate_pointcloud(
    source_points: np.ndarray,
    target_points: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Union[float, List[float], np.ndarray]]:
    """Compute Acc / Comp / Chamfer / P / R / F1 (lingbot ``evaluate_pointcloud``)."""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10]
    thresholds = list(thresholds)

    source = np.asarray(source_points).reshape(-1, source_points.shape[-1])
    target = np.asarray(target_points).reshape(-1, target_points.shape[-1])

    dist_s2t = nearest_neighbor_distances(source, target)
    dist_t2s = nearest_neighbor_distances(target, source)

    acc = float(np.mean(dist_s2t))
    comp = float(np.mean(dist_t2s))
    chamfer = chamfer_distance(acc, comp)

    prec_list = [float(np.mean(dist_s2t < t) * 100.0) for t in thresholds]
    rec_list = [float(np.mean(dist_t2s < t) * 100.0) for t in thresholds]
    f1_list = f1_from_pr(prec_list, rec_list)

    results: Dict[str, Union[float, List[float], np.ndarray]] = {
        "chamfer": chamfer,
        "accuracy": acc,
        "completeness": comp,
        "pred_points": np.hstack([source[:, :3], dist_s2t.reshape(-1, 1)]),
        "gt_points": np.hstack([target[:, :3], dist_t2s.reshape(-1, 1)]),
        "thresholds": thresholds,
    }
    if len(thresholds) == 1:
        results["precision"] = prec_list[0]
        results["recall"] = rec_list[0]
        results["f1"] = f1_list[0]
    else:
        for i, t in enumerate(thresholds):
            t_str = threshold_suffix(t)
            results[f"precision_{t_str}"] = prec_list[i]
            results[f"recall_{t_str}"] = rec_list[i]
            results[f"f1_{t_str}"] = f1_list[i]
    return results


# ---------------------------------------------------------------------------
# Full reconstruction evaluation pipeline
# ---------------------------------------------------------------------------

def _default_pred_mask(pred_pts: np.ndarray) -> np.ndarray:
    """Finite xyz → valid pred mask (same spatial rank as pred_pts[..., 0]).

    Prefer passing an explicit depth>1e-4 mask (official BSSLoader) from the
    model interface when available.
    """
    return np.isfinite(pred_pts).all(axis=-1)


def subsample_indices(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Indices for uniform random subsample (``max_points<=0`` → all)."""
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=int(max_points), replace=False)


def subsample_points(
    points: np.ndarray,
    max_points: int,
    seed: int = 0,
    colors: Optional[np.ndarray] = None,
):
    """Uniform random subsample if cloud exceeds ``max_points`` (``<=0`` = keep all).

    If ``colors`` is provided, returns ``(points, colors)`` with the same indices.
    """
    pts = np.asarray(points)
    idx = subsample_indices(len(pts), max_points, seed=seed)
    if colors is None:
        return pts[idx]
    cols = np.asarray(colors)
    if len(cols) != len(pts):
        raise ValueError(f"colors length {len(cols)} != points length {len(pts)}")
    return pts[idx], cols[idx]


def images_to_rgb_hwc(images) -> np.ndarray:
    """Convert batch RGB to ``(S, H, W, 3)`` float in ``[0, 1]``.

    Accepts ``(S, 3, H, W)`` / ``(S, H, W, 3)`` torch/numpy, float or uint8.
    """
    if hasattr(images, "detach"):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"images must be 4D, got shape {arr.shape}")
    if arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[-1] != 3:
        raise ValueError(f"cannot find RGB channel dim in shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def colored_aligned_clouds_for_ply(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    gt_mask: np.ndarray,
    images,
    T_umeyama: np.ndarray,
    T_icp: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense Sim(3)+ICP aligned clouds with original RGB (for visualization PLY).

    Uses pixel-aligned GT-grid RGB — **not** the metric voxelized clouds (indoor
    voxel drops 1:1 correspondence). Pred points use ``gt ∩ pred``; GT uses
    ``gt_mask``. Colors come from the same GT-resolution ``images``.
    """
    pred_pts = np.asarray(pred_pts)
    gt_pts = np.asarray(gt_pts)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if pred_mask is None:
        pred_mask = _default_pred_mask(pred_pts)
    else:
        pred_mask = np.asarray(pred_mask, dtype=bool)
    common = gt_mask & pred_mask
    rgb = images_to_rgb_hwc(images)
    if rgb.shape[:3] != pred_pts.shape[:-1]:
        raise ValueError(
            f"RGB spatial {rgb.shape[:3]} != pred spatial {pred_pts.shape[:-1]}"
        )
    T = np.asarray(T_umeyama, dtype=np.float64)
    if T_icp is not None:
        T = np.asarray(T_icp, dtype=np.float64) @ T
    pred_xyz = apply_transform(pred_pts[common], T).astype(np.float32)
    pred_rgb = rgb[common].astype(np.float32)
    gt_xyz = gt_pts[gt_mask].astype(np.float32)
    gt_rgb = rgb[gt_mask].astype(np.float32)
    return pred_xyz, pred_rgb, gt_xyz, gt_rgb


def save_xyz_ply(path: str, points: np.ndarray) -> None:
    """Write an xyz-only binary PLY via Open3D."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"Failed to write PLY: {path}")


def save_xyzrgb_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write xyz+RGB binary PLY via Open3D. ``colors`` in ``[0,1]`` or ``[0,255]``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if len(pts) != len(cols):
        raise ValueError(f"points {len(pts)} vs colors {len(cols)}")
    if cols.max() > 1.5:
        cols = cols / 255.0
    cols = np.clip(cols, 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"Failed to write colored PLY: {path}")


def evaluate_reconstruction(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    alignment_gt_mask: Optional[np.ndarray] = None,
    alignment_pred_mask: Optional[np.ndarray] = None,
    icp_threshold: float = DEFAULT_ICP_THRESHOLD,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    eval_threshold: float = DEFAULT_EVAL_THRESHOLD_INDOOR,
    eval_thresholds: Optional[Sequence[float]] = None,
    min_correspondences: int = 6,
    return_clouds: bool = False,
    with_scale: bool = True,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Align pred to GT (Umeyama → optional voxel → ICP) and score like lingbot.

    Args:
        pred_pts: (..., 3) predicted world points (same spatial layout as gt).
        gt_pts: (..., 3) GT world points.
        gt_mask: bool mask of valid GT pixels / points (same leading dims).
        pred_mask: optional valid pred mask used for ICP and final metrics;
            defaults to finite xyz.
        alignment_gt_mask: optional GT-side mask used only to estimate the
            initial Umeyama transform. It never removes GT from final metrics.
        alignment_pred_mask: optional additional mask used only to estimate the
            initial Umeyama transform. Points excluded here remain in ICP and
            final Acc/Comp/CD/F1 through ``pred_mask``.
        icp_threshold: max correspondence distance for ICP (metres).
        voxel_size: voxel size in metres; ``<=0`` disables downsampling.
        eval_threshold: primary distance threshold for compatibility aliases.
        eval_thresholds: optional thresholds evaluated together from the same
            two nearest-neighbor distance arrays; no extra model inference.
        min_correspondences: minimum common-mask points for Umeyama.
        return_clouds: if True, also return ``pred_points`` / ``gt_points`` used
            for metric computation (aligned eval clouds).
        with_scale: if True, Umeyama Sim(3); if False, SE(3) only (keep pred
            scale — HorizonStream metric depth/pose). ICP is always rigid.

    Returns:
        Metric dict with accuracy, completeness, chamfer, precision, recall, f1,
        plus diagnostic counts. Raises ValueError if alignment is impossible.
    """
    pred_pts = np.asarray(pred_pts)
    gt_pts = np.asarray(gt_pts)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    if pred_mask is None:
        pred_mask = _default_pred_mask(pred_pts)
    else:
        pred_mask = np.asarray(pred_mask, dtype=bool)

    if pred_pts.shape != gt_pts.shape:
        raise ValueError(
            f"pred/gt shape mismatch: {pred_pts.shape} vs {gt_pts.shape}"
        )
    if gt_mask.shape != pred_pts.shape[:-1]:
        raise ValueError(
            f"gt_mask shape {gt_mask.shape} != pred spatial {pred_pts.shape[:-1]}"
        )
    if pred_mask.shape != pred_pts.shape[:-1]:
        raise ValueError(
            f"pred_mask shape {pred_mask.shape} != pred spatial {pred_pts.shape[:-1]}"
        )

    if alignment_gt_mask is None:
        alignment_gt_mask = gt_mask
    else:
        alignment_gt_mask = np.asarray(alignment_gt_mask, dtype=bool)
        if alignment_gt_mask.shape != pred_pts.shape[:-1]:
            raise ValueError(
                f"alignment_gt_mask shape {alignment_gt_mask.shape} != "
                f"pred spatial {pred_pts.shape[:-1]}"
            )

    if alignment_pred_mask is None:
        alignment_pred_mask = pred_mask
    else:
        alignment_pred_mask = np.asarray(alignment_pred_mask, dtype=bool)
        if alignment_pred_mask.shape != pred_pts.shape[:-1]:
            raise ValueError(
                f"alignment_pred_mask shape {alignment_pred_mask.shape} != "
                f"pred spatial {pred_pts.shape[:-1]}"
            )

    common_mask = gt_mask & pred_mask
    n_common = int(common_mask.sum())
    alignment_common_mask = (
        common_mask & alignment_gt_mask & alignment_pred_mask
    )
    n_alignment_common = int(alignment_common_mask.sum())
    if n_alignment_common < min_correspondences:
        raise ValueError(
            "Insufficient Umeyama correspondences after alignment filtering: "
            f"{n_alignment_common} < {min_correspondences}"
        )

    pred_align_corr = pred_pts[alignment_common_mask][:, :3]
    gt_align_corr = gt_pts[alignment_common_mask][:, :3]
    T_umeyama = umeyama_registration(
        pred_align_corr, gt_align_corr, with_scale=with_scale
    )

    # Pred for ICP/eval deliberately uses the full evaluation mask. Alignment
    # filtering must not silently remove difficult points from Acc/Comp/CD/F1.
    pred_corr = pred_pts[common_mask][:, :3]
    pred_after_umeyama = apply_transform(pred_corr, T_umeyama)
    # GT for ICP/eval: full GT-valid cloud (not restricted to common_mask).
    gt_full = gt_pts[gt_mask][:, :3]

    if voxel_size > 0:
        pred_ds = voxel_downsample(pred_after_umeyama, voxel_size)
        gt_ds = voxel_downsample(gt_full, voxel_size)
    else:
        pred_ds = pred_after_umeyama.astype(np.float32)
        gt_ds = gt_full.astype(np.float32)

    T_icp = icp_registration(
        source_points=pred_ds,
        target_points=gt_ds,
        icp_threshold=icp_threshold,
    )

    if voxel_size > 0:
        # Formal datasets use voxelized clouds when voxel_size is positive.
        pred_eval = apply_transform(pred_ds, T_icp)
        gt_eval = gt_ds
    else:
        # Generic non-voxel path.
        T_total = T_icp @ T_umeyama
        pred_eval = apply_transform(pred_corr, T_total)
        gt_eval = gt_full

    thresholds = normalize_eval_thresholds(eval_threshold, eval_thresholds)
    metrics = evaluate_pointcloud(
        source_points=pred_eval,
        target_points=gt_eval,
        thresholds=thresholds,
    )
    if len(thresholds) > 1:
        primary_suffix = threshold_suffix(eval_threshold)
        metrics["precision"] = metrics[f"precision_{primary_suffix}"]
        metrics["recall"] = metrics[f"recall_{primary_suffix}"]
        metrics["f1"] = metrics[f"f1_{primary_suffix}"]
    metrics["num_pred"] = int(len(pred_eval))
    metrics["num_gt"] = int(len(gt_eval))
    metrics["num_correspondences"] = n_common
    metrics["num_alignment_correspondences"] = n_alignment_common
    metrics["icp_threshold"] = float(icp_threshold)
    metrics["voxel_size"] = float(voxel_size)
    metrics["eval_threshold"] = float(eval_threshold)
    metrics["eval_thresholds"] = thresholds
    metrics["with_scale"] = bool(with_scale)
    metrics["T_umeyama"] = T_umeyama
    metrics["T_icp"] = T_icp
    if return_clouds:
        metrics["pred_points"] = np.asarray(pred_eval, dtype=np.float32)
        metrics["gt_points"] = np.asarray(gt_eval, dtype=np.float32)
    return metrics


def metrics_to_csv_row(seq_name: str, metrics: Dict) -> Dict[str, Union[str, float]]:
    """Flatten lingbot metrics into a CSV-friendly row."""
    row = {
        "seq": seq_name,
        "accuracy": float(metrics["accuracy"]),
        "completeness": float(metrics["completeness"]),
        "chamfer": float(metrics["chamfer"]),
        "f1": float(metrics["f1"]),
        "eval_threshold": float(metrics["eval_threshold"]),
        "num_pred": int(metrics["num_pred"]),
        "num_gt": int(metrics["num_gt"]),
    }
    thresholds = metrics.get("eval_thresholds", [metrics["eval_threshold"]])
    if len(thresholds) > 1:
        for key in threshold_metric_keys(thresholds):
            row[key] = float(metrics[key])
    return row


METRIC_KEYS = (
    "accuracy",
    "completeness",
    "chamfer",
    "f1",
)
