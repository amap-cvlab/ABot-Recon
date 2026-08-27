"""Trajectory dump + BEV visualization for mv_recon."""

from __future__ import annotations

import os
import os.path as osp
from typing import Optional, Tuple, Union

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert (N,3,4) or (N,4,4) w2c → (N,4,4) c2w."""
    w2c = np.asarray(w2c, dtype=np.float64)
    n = w2c.shape[0]
    if w2c.shape[-2:] == (3, 4):
        bottom = np.zeros((n, 1, 4), dtype=np.float64)
        bottom[:, 0, 3] = 1.0
        w2c_4 = np.concatenate([w2c, bottom], axis=1)
    elif w2c.shape[-2:] == (4, 4):
        w2c_4 = w2c
    else:
        raise ValueError(f"Unexpected w2c shape {w2c.shape}")
    return np.linalg.inv(w2c_4)


def c2w_3x4_to_4x4(c2w: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim == 2:
        c2w = c2w[None]
    n = c2w.shape[0]
    if c2w.shape[-2:] == (4, 4):
        return c2w
    if c2w.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected c2w shape {c2w.shape}")
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, :3, :] = c2w
    out[:, 3, 3] = 1.0
    return out


def gt_c2w_from_batch(data: dict) -> Optional[np.ndarray]:
    """Extract GT camera-to-world (N,4,4) from dataset batch (extrs are w2c)."""
    if "extrs" not in data or data["extrs"] is None:
        return None
    extrs = _to_numpy(data["extrs"])
    return w2c_to_c2w(extrs).astype(np.float64)


def apply_sim3_to_c2w(c2w: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 Sim3 ``T`` (maps pred-world → gt-world) to c2w poses.

    For T = [[sR, t], [0, 1]]:
      R' = sR/|s| wait — store sR in upper-left; scale = ||sR||_cols.
    We recover scale from the linear part and apply:
      R_new = R_lin / scale
      t_new = R_lin @ t_old + t
    equivalently compose: c2w_new = T @ c2w  when T encodes scaled rotation,
    but T @ c2w mixes scale into rotation columns. Prefer explicit form:
      c' = s R c + t,  R' = R_u @ R
    where T_umeyama = [[sR_u, t],[0,1]].
    """
    c2w = c2w_3x4_to_4x4(c2w)
    T = np.asarray(T, dtype=np.float64)
    R_scaled = T[:3, :3]
    t = T[:3, 3]
    # column norms of sR should be ~s
    scales = np.linalg.norm(R_scaled, axis=0)
    scale = float(np.mean(scales))
    if scale < 1e-12:
        raise ValueError("Degenerate Sim3 scale")
    R = R_scaled / scale

    out = np.zeros_like(c2w)
    for i in range(c2w.shape[0]):
        out[i, :3, :3] = R @ c2w[i, :3, :3]
        out[i, :3, 3] = scale * (R @ c2w[i, :3, 3]) + t
        out[i, 3, 3] = 1.0
    return out


def compose_pc_align_transform(
    T_umeyama: np.ndarray,
    T_icp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full pred→gt transform used by point-cloud metrics: T_icp @ T_umeyama."""
    T_u = np.asarray(T_umeyama, dtype=np.float64)
    if T_icp is None:
        return T_u
    return np.asarray(T_icp, dtype=np.float64) @ T_u


def trajectory_umeyama_transform(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    with_scale: bool,
) -> np.ndarray:
    """Estimate the trajectory-centre Umeyama transform pred-world → GT-world."""
    pred_xyz = c2w_3x4_to_4x4(pred_c2w)[:, :3, 3]
    gt_xyz = c2w_3x4_to_4x4(gt_c2w)[:, :3, 3]
    if len(pred_xyz) < 3:
        raise ValueError(f"Trajectory Umeyama needs at least 3 poses, got {len(pred_xyz)}")
    pred_centered = pred_xyz - pred_xyz.mean(axis=0)
    gt_centered = gt_xyz - gt_xyz.mean(axis=0)
    covariance = gt_centered.T @ pred_centered / len(pred_xyz)
    U, singular_values, Vt = np.linalg.svd(covariance)
    sign = np.eye(3, dtype=np.float64)
    sign[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ sign @ Vt
    if with_scale:
        variance = float(np.mean(np.sum(pred_centered ** 2, axis=1)))
        if variance < 1e-15:
            raise ValueError("Degenerate predicted trajectory for Sim3 alignment")
        scale = float(np.sum(singular_values * np.diag(sign)) / variance)
    else:
        scale = 1.0
    translation = gt_xyz.mean(axis=0) - scale * (rotation @ pred_xyz.mean(axis=0))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def save_sequence_traj_bev(
    save_dir: str,
    seq_name: str,
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    T_umeyama: np.ndarray,
    T_icp: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    alignment_label: str = "point-cloud alignment",
    trajectory_with_scale: bool = True,
    verbose: bool = False,
) -> Tuple[str, str]:
    """Save raw/aligned pred + GT TUM traj and a BEV (xy) GT+pred plot.

    Camera trajectories are aligned from their own camera centres. Pointmap
    alignment is intentionally not reused because point and pose heads need not
    share an exactly consistent world transform.
    Returns (aligned_tum_path, bev_png_path).
    """
    from relpose.evo_utils import get_tum_poses, plot_trajectory, save_tum_poses

    os.makedirs(save_dir, exist_ok=True)
    pred_c2w = c2w_3x4_to_4x4(_to_numpy(pred_c2w))
    gt_c2w = c2w_3x4_to_4x4(_to_numpy(gt_c2w))
    n = min(len(pred_c2w), len(gt_c2w))
    pred_c2w = pred_c2w[:n]
    gt_c2w = gt_c2w[:n]

    T = trajectory_umeyama_transform(
        pred_c2w, gt_c2w, with_scale=trajectory_with_scale
    )
    pred_aligned = apply_sim3_to_c2w(pred_c2w, T)

    pred_raw_traj = get_tum_poses(list(pred_c2w))
    pred_aln_traj = get_tum_poses(list(pred_aligned))
    gt_traj = get_tum_poses(list(gt_c2w))

    raw_path = osp.join(save_dir, "pred_traj_raw.txt")
    aln_path = osp.join(save_dir, "pred_traj_umeyama.txt")
    gt_path = osp.join(save_dir, "gt_traj.txt")
    save_tum_poses(pred_raw_traj, raw_path, verbose=verbose)
    save_tum_poses(pred_aln_traj, aln_path, verbose=verbose)
    save_tum_poses(gt_traj, gt_path, verbose=verbose)

    # ATE under one trajectory-centre Umeyama alignment (no second evo align).
    pred_xyz = pred_aligned[:, :3, 3]
    gt_xyz = gt_c2w[:, :3, 3]
    err = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    ate_rmse = float(np.sqrt(np.mean(err ** 2)))
    ate_mean = float(np.mean(err))
    ate_med = float(np.median(err))
    with open(osp.join(save_dir, "traj_ate_umeyama.txt"), "w") as f:
        f.write(f"ate_rmse={ate_rmse:.6f}\n")
        f.write(f"ate_mean={ate_mean:.6f}\n")
        f.write(f"ate_median={ate_med:.6f}\n")
        f.write(f"num_poses={n}\n")

    bev_base = osp.join(save_dir, "vis_traj_bev_umeyama.png")
    plot_title = title or seq_name
    plot_title = (
        f"{plot_title} | ATE_RMSE={ate_rmse:.3f}m "
        f"(trajectory {alignment_label})"
    )
    plot_trajectory(
        pred_aln_traj,
        gt_traj,
        title=plot_title,
        filename=bev_base,
        align=False,
        correct_scale=False,
        verbose=verbose,
        plot_mode_key="xy",
    )
    # evo exports as <filename>_traj_error.png when filename ends with .png quirks;
    # plot_collection.export uses the given filename stem. Prefer the known suffix.
    exported = bev_base.replace(".png", "") + "_traj_error.png"
    if osp.exists(exported):
        bev_path = exported
    elif osp.exists(bev_base):
        bev_path = bev_base
    else:
        # fall back: whatever png appeared
        pngs = [p for p in os.listdir(save_dir) if p.endswith(".png")]
        bev_path = osp.join(save_dir, pngs[0]) if pngs else exported

    return aln_path, bev_path


def unpack_infer_mv_result(
    result,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict],
]:
    """Unpack points with optional camera, masks, and dense auxiliary maps."""
    if isinstance(result, (tuple, list)):
        if len(result) == 6:
            pts, c2w, pred_mask, observation_mask, alignment_pred_mask, dense_aux = result
            c2w_out = None if c2w is None else c2w_3x4_to_4x4(_to_numpy(c2w))
            mask_out = None if pred_mask is None else np.asarray(pred_mask, dtype=bool)
            observed_out = None if observation_mask is None else np.asarray(observation_mask, dtype=bool)
            alignment_out = None if alignment_pred_mask is None else np.asarray(alignment_pred_mask, dtype=bool)
            aux_out = None if dense_aux is None else {key: _to_numpy(value) for key, value in dense_aux.items()}
            return pts, c2w_out, mask_out, observed_out, alignment_out, aux_out
        if len(result) == 5:
            pts, c2w, pred_mask, observation_mask, alignment_pred_mask = result
            c2w_out = None if c2w is None else c2w_3x4_to_4x4(_to_numpy(c2w))
            mask_out = None if pred_mask is None else np.asarray(pred_mask, dtype=bool)
            observed_out = (
                None
                if observation_mask is None
                else np.asarray(observation_mask, dtype=bool)
            )
            alignment_out = (
                None
                if alignment_pred_mask is None
                else np.asarray(alignment_pred_mask, dtype=bool)
            )
            return pts, c2w_out, mask_out, observed_out, alignment_out, None
        if len(result) == 4:
            pts, c2w, pred_mask, observation_mask = result
            c2w_out = None if c2w is None else c2w_3x4_to_4x4(_to_numpy(c2w))
            mask_out = None if pred_mask is None else np.asarray(pred_mask, dtype=bool)
            observed_out = (
                None
                if observation_mask is None
                else np.asarray(observation_mask, dtype=bool)
            )
            return pts, c2w_out, mask_out, observed_out, None, None
        if len(result) == 3:
            pts, c2w, pred_mask = result
            c2w_out = None if c2w is None else c2w_3x4_to_4x4(_to_numpy(c2w))
            mask_out = None if pred_mask is None else np.asarray(pred_mask, dtype=bool)
            return pts, c2w_out, mask_out, None, None, None
        if len(result) == 2:
            pts, c2w = result
            if c2w is None:
                return pts, None, None, None, None, None
            return pts, c2w_3x4_to_4x4(_to_numpy(c2w)), None, None, None, None
    return result, None, None, None, None, None
