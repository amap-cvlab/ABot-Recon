"""Training-aligned RGB preprocessing for ABot-Recon evaluation.

Matches ``BaseDataset._resize_width_crop_or_pad_mean``:
  1. Scale so width == ``target_w`` (default 504); height scales with aspect.
  2. If resized_h > target_h (280): center-crop height.
  3. If resized_h < target_h: vertical mean-pad with ImageNet mean RGB.

Pad rows must be stripped from predicted local/world points before GT metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

DEFAULT_TARGET_H = 280
DEFAULT_TARGET_W = 504
DEFAULT_PAD_RGB = (0.485, 0.456, 0.406)


@dataclass(frozen=True)
class FovPadInfo:
    """Geometry of width-lock resize followed by vertical crop or pad.

    ``crop_top``/``crop_bottom`` are measured in the width-resized image, before
    the model-size crop.  Keeping them is essential when mapping model pointmaps
    back to a GT image: a center crop observes only the corresponding GT ROI.
    """

    pad_top: int
    pad_bottom: int
    target_h: int
    target_w: int
    content_h: int  # rows of real content before pad (= target_h - pad_top - pad_bottom)
    src_h: int = 0
    src_w: int = 0
    resized_h: int = 0
    crop_top: int = 0
    crop_bottom: int = 0

    @property
    def has_pad(self) -> bool:
        return self.pad_top > 0 or self.pad_bottom > 0

    @property
    def has_crop(self) -> bool:
        return self.crop_top > 0 or self.crop_bottom > 0

    def valid_row_slice(self) -> slice:
        end = self.target_h - self.pad_bottom
        return slice(self.pad_top, end)

    def gt_content_row_slice(self, gt_h: int) -> slice:
        """Rows in a same-aspect GT image corresponding to model content."""
        gt_h = int(gt_h)
        if gt_h <= 0:
            raise ValueError(f"Invalid GT height: {gt_h}")
        if not self.has_crop:
            return slice(0, gt_h)
        if self.resized_h <= 0:
            raise ValueError("Crop metadata requires resized_h > 0")
        start = int(round(self.crop_top * gt_h / self.resized_h))
        end = int(round((self.resized_h - self.crop_bottom) * gt_h / self.resized_h))
        start = min(max(start, 0), gt_h - 1)
        end = min(max(end, start + 1), gt_h)
        return slice(start, end)


@dataclass(frozen=True)
class FovFrame:
    image: torch.Tensor  # (3, H, W) float in [0, 1]
    pad: FovPadInfo


def _as_chw01(rgb: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
    if isinstance(rgb, Image.Image):
        t = TF.to_tensor(rgb.convert("RGB"))
        return t
    if isinstance(rgb, np.ndarray):
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 ndarray, got shape {arr.shape}")
        t = torch.from_numpy(arr.copy())
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
            if float(t.max()) > 1.5:
                t = t / 255.0
        return t.permute(2, 0, 1).contiguous()
    if not torch.is_tensor(rgb):
        raise TypeError(f"Unsupported rgb type: {type(rgb)}")
    t = rgb.detach().float().cpu()
    if t.dim() == 3 and t.shape[0] == 3:
        pass
    elif t.dim() == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1).contiguous()
    else:
        raise ValueError(f"Expected CHW or HWC rgb tensor, got {tuple(t.shape)}")
    if float(t.max()) > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def resize_width_crop_or_pad_mean(
    rgb: Union[torch.Tensor, np.ndarray, Image.Image],
    *,
    target_h: int = DEFAULT_TARGET_H,
    target_w: int = DEFAULT_TARGET_W,
    pad_rgb: Sequence[float] = DEFAULT_PAD_RGB,
) -> FovFrame:
    """FOV-preserving resize used by HybridLong training (width-lock + pad/crop)."""
    target_h = int(target_h)
    target_w = int(target_w)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size {(target_h, target_w)}")

    t = _as_chw01(rgb)
    _, src_h, src_w = t.shape
    scale = float(target_w) / float(max(src_w, 1))
    resized_h = max(1, int(round(float(src_h) * scale)))
    t = TF.resize(
        t,
        [resized_h, target_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    pad_top = 0
    pad_bottom = 0
    crop_top = 0
    crop_bottom = 0
    content_h = target_h

    if resized_h > target_h:
        crop_top = int(round((resized_h - target_h) * 0.5))
        crop_bottom = int(resized_h - target_h - crop_top)
        t = TF.crop(t, crop_top, 0, target_h, target_w)
        content_h = target_h
    elif resized_h < target_h:
        pad_top = int((target_h - resized_h) // 2)
        pad_bottom = int(target_h - resized_h - pad_top)
        content_h = resized_h
        pad = torch.tensor(
            [float(c) for c in pad_rgb], dtype=t.dtype, device=t.device
        ).view(3, 1, 1)
        canvas = pad.expand(3, target_h, target_w).clone()
        canvas[:, pad_top : pad_top + resized_h, :] = t
        t = canvas
    else:
        content_h = target_h

    pad_info = FovPadInfo(
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        target_h=target_h,
        target_w=target_w,
        content_h=content_h,
        src_h=src_h,
        src_w=src_w,
        resized_h=resized_h,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
    )
    # Preserve the tiny bicubic boundary overshoots used by the published
    # evaluation pipeline; clamping here changes long-horizon pose composition.
    return FovFrame(image=t, pad=pad_info)


def load_filelist_fov(
    filelist: Iterable[str],
    *,
    target_h: int = DEFAULT_TARGET_H,
    target_w: int = DEFAULT_TARGET_W,
    pad_rgb: Sequence[float] = DEFAULT_PAD_RGB,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, FovPadInfo]:
    """Load RGB paths → ``(1, S, 3, H, W)`` with shared FOV pad (same aspect → same pad).

    Raises if frames disagree on pad geometry (should not happen for fixed sensor res).
    """
    frames: List[torch.Tensor] = []
    pads: List[FovPadInfo] = []
    for path in filelist:
        im = Image.open(path).convert("RGB")
        fr = resize_width_crop_or_pad_mean(
            im, target_h=target_h, target_w=target_w, pad_rgb=pad_rgb
        )
        frames.append(fr.image)
        pads.append(fr.pad)
    if not frames:
        raise FileNotFoundError("Empty filelist for FOV load")

    ref = pads[0]
    for i, p in enumerate(pads[1:], start=1):
        if p != ref:
            raise ValueError(
                f"Inconsistent FOV pad across frames: frame0={ref} vs frame{i}={p}"
            )

    batch = torch.stack(frames, dim=0).unsqueeze(0)  # 1,S,3,H,W
    if device is not None:
        batch = batch.to(device)
    return batch, ref


def strip_vertical_pad_map(
    maps: Union[torch.Tensor, np.ndarray],
    pad: FovPadInfo,
    *,
    spatial_dims: Tuple[int, int] = (-2, -1),
) -> Union[torch.Tensor, np.ndarray]:
    """Remove vertical mean-pad rows from a spatial map.

    ``spatial_dims`` defaults to last-two dims being (H, W). For ``(..., H, W, C)``
    pass ``spatial_dims=(-3, -2)``.
    """
    h_dim, w_dim = spatial_dims
    if torch.is_tensor(maps):
        h = int(maps.shape[h_dim])
        if h != pad.target_h:
            raise ValueError(
                f"Map H={h} does not match FOV target_h={pad.target_h}; "
                "strip pad only on model-resolution outputs"
            )
        if not pad.has_pad:
            return maps
        # Normalize negative dims
        ndim = maps.dim()
        hd = h_dim if h_dim >= 0 else ndim + h_dim
        sl = [slice(None)] * ndim
        sl[hd] = pad.valid_row_slice()
        return maps[tuple(sl)]

    arr = np.asarray(maps)
    h = int(arr.shape[h_dim])
    if h != pad.target_h:
        raise ValueError(
            f"Map H={h} does not match FOV target_h={pad.target_h}; "
            "strip pad only on model-resolution outputs"
        )
    if not pad.has_pad:
        return arr
    ndim = arr.ndim
    hd = h_dim if h_dim >= 0 else ndim + h_dim
    sl = [slice(None)] * ndim
    sl[hd] = pad.valid_row_slice()
    return arr[tuple(sl)]


def strip_vertical_pad_hwc(
    maps: Union[torch.Tensor, np.ndarray],
    pad: FovPadInfo,
) -> Union[torch.Tensor, np.ndarray]:
    """Strip pad from ``(..., H, W, C)`` tensors (local/world points)."""
    return strip_vertical_pad_map(maps, pad, spatial_dims=(-3, -2))
