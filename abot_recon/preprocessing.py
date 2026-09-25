from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from torchvision.transforms import InterpolationMode


DEFAULT_PAD_RGB = (0.485, 0.456, 0.406)


@dataclass(frozen=True)
class FovTransform:
    source_height: int
    source_width: int
    resized_height: int
    target_height: int
    target_width: int
    crop_top: int = 0
    crop_bottom: int = 0
    pad_top: int = 0
    pad_bottom: int = 0

    @property
    def valid_rows(self) -> slice:
        return slice(self.pad_top, self.target_height - self.pad_bottom)


def _to_chw01(image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return tvf.to_tensor(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.asarray(image).copy())
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB array, got {tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1)
    elif torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected three-dimensional image, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(f"Expected CHW or HWC RGB tensor, got {tuple(tensor.shape)}")
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")
    tensor = tensor.float()
    if tensor.numel() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp_(0.0, 1.0).contiguous()


def preprocess_image(
    image: Image.Image | np.ndarray | torch.Tensor,
    *,
    height: int = 280,
    width: int = 504,
    pad_rgb: Sequence[float] = DEFAULT_PAD_RGB,
) -> tuple[torch.Tensor, FovTransform]:
    """Width-lock resize followed by center crop or ImageNet-mean vertical pad."""
    tensor = _to_chw01(image)
    _, source_height, source_width = tensor.shape
    resized_height = max(1, round(source_height * width / max(source_width, 1)))
    tensor = tvf.resize(
        tensor,
        [resized_height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    crop_top = crop_bottom = pad_top = pad_bottom = 0
    if resized_height > height:
        crop_top = round((resized_height - height) * 0.5)
        crop_bottom = resized_height - height - crop_top
        tensor = tvf.crop(tensor, crop_top, 0, height, width)
    elif resized_height < height:
        pad_top = (height - resized_height) // 2
        pad_bottom = height - resized_height - pad_top
        color = torch.as_tensor(pad_rgb, dtype=tensor.dtype).view(3, 1, 1)
        canvas = color.expand(3, height, width).clone()
        canvas[:, pad_top : pad_top + resized_height] = tensor
        tensor = canvas
    transform = FovTransform(
        source_height=source_height,
        source_width=source_width,
        resized_height=resized_height,
        target_height=height,
        target_width=width,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
    # Preserve the tiny bicubic boundary overshoots used by the published
    # inference pipeline; clamping here changes long-horizon pose composition.
    return tensor, transform


def iter_preprocessed(
    paths: Iterable[str | Path], *, height: int = 280, width: int = 504
):
    reference: FovTransform | None = None
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            tensor, transform = preprocess_image(image, height=height, width=width)
        if reference is None:
            reference = transform
        elif transform != reference:
            raise ValueError(
                f"Inconsistent frame geometry at index {index}: {transform} != {reference}"
            )
        yield tensor, transform

