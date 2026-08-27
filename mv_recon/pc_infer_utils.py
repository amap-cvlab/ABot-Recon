"""Shared helpers for mv_recon point-cloud inference → GT-grid alignment."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig


DEPTH_VALID_THR = 1e-4


def nearest_depth_to_gt_enabled(hydra_cfg: DictConfig) -> bool:
    """Fair-compare / lingbot-bench: GT at native/prepare res + NEAREST depth upsample."""
    return bool(getattr(hydra_cfg, "nearest_depth_to_gt", False))


def pointmap_resize_mode(hydra_cfg: DictConfig) -> str:
    """Interpolation used only for direct XYZ point-map predictions.

    Keep the legacy ``nearest_depth_to_gt`` fallback so old configs remain
    reproducible. New experiments should set ``pointmap_resize_mode``
    explicitly to avoid conflating XYZ interpolation with depth interpolation.
    """
    configured = getattr(hydra_cfg, "pointmap_resize_mode", None)
    if configured is None:
        return "nearest" if nearest_depth_to_gt_enabled(hydra_cfg) else "bilinear"
    mode = str(configured).lower()
    if mode not in {"nearest", "bilinear"}:
        raise ValueError(
            f"pointmap_resize_mode must be 'nearest' or 'bilinear', got {configured!r}"
        )
    return mode


def resize_map_to_hw(
    maps: torch.Tensor,
    target_hw: Tuple[int, int],
    *,
    nearest: bool,
) -> torch.Tensor:
    """Resize spatial maps to ``target_hw``.

    Accepts:
      - depth: (S, H, W) or (S, 1, H, W)
      - points/features: (S, C, H, W)

    When ``nearest=True`` (fair PC protocol), uses NEAREST — matches official
    lingbot-map saver upsampling depth to GT resolution.
    """
    th, tw = int(target_hw[0]), int(target_hw[1])
    x = maps.float()
    squeeze_back = False
    if x.dim() == 3:
        x = x.unsqueeze(1)
        squeeze_back = True
    if x.shape[-2] == th and x.shape[-1] == tw:
        return x.squeeze(1) if squeeze_back else x
    if nearest:
        out = F.interpolate(x, size=(th, tw), mode="nearest")
    else:
        out = F.interpolate(
            x, size=(th, tw), mode="bilinear", align_corners=False, antialias=True
        )
    return out.squeeze(1) if squeeze_back else out


def pred_mask_from_depth(depth: torch.Tensor | np.ndarray, thr: float = DEPTH_VALID_THR) -> np.ndarray:
    """Official-style pred validity: depth > thr (after resize to GT grid)."""
    if isinstance(depth, torch.Tensor):
        d = depth.detach().float().cpu().numpy()
    else:
        d = np.asarray(depth)
    return d > float(thr)
