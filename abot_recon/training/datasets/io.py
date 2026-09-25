from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def image_open(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def imread_cv2(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise OSError(f"Could not read image: {path}")
    if flags == cv2.IMREAD_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def np_load(path: str | Path):
    return np.load(path, allow_pickle=False)
