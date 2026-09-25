import functools
import torchvision.transforms as tvf
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.v2.functional as TF
import PIL
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
    bilinear = PIL.Image.Resampling.BILINEAR
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC
    bilinear = PIL.Image.BILINEAR

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])            # to [-1, 1]
ImgToTensor = tvf.ToTensor()
# ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])

CustomNorm = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def inverse_ImgNorm(img_norm):
    mean = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1).to(img_norm.device)
    std = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1).to(img_norm.device)
    return img_norm * std + mean

def inverse_CustomNorm(custom_norm_img):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(custom_norm_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(custom_norm_img.device)
    return custom_norm_img * std + mean


class JpegLoss:
    def __init__(self, seed=2024, prob=0.5, quality_range=(20, 100)):
        self.prob = prob
        self.quality_range = quality_range
        self.rng = np.random.default_rng(seed)

    def sample_params(self, rng=None):
        rng = self.rng if rng is None else rng
        enabled = bool(rng.uniform() < self.prob)
        return {
            "enabled": enabled,
            "quality": int(rng.integers(*self.quality_range)) if enabled else None,
        }

    def apply_with_params(self, img, params):
        if params.get("enabled", False):
            img_cv = np.array(img)[:, :, ::-1]
            quality = int(params["quality"])
            _, encoded = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img_cv = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            img = Image.fromarray(img_cv[:, :, ::-1])  # BGR to RGB
        return img

    def __call__(self, img):
        return self.apply_with_params(img, self.sample_params())

class Blurring:
    def __init__(self, seed=2025, prob=0.5, resize_ratio_range=(0.25, 1), interpolation_methods=None):
        self.prob = prob
        self.resize_ratio_range = resize_ratio_range
        # self.interpolation_methods = interpolation_methods or [
        #     cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        # ]
        self.interpolation_methods = interpolation_methods or [
            lanczos, bicubic, bilinear
        ]
        self.rng = np.random.default_rng(seed)

    def sample_params(self, rng=None):
        rng = self.rng if rng is None else rng
        enabled = bool(rng.uniform() < self.prob)
        return {
            "enabled": enabled,
            "ratio": float(rng.uniform(*self.resize_ratio_range)) if enabled else None,
            "interpolation": rng.choice(self.interpolation_methods) if enabled else None,
        }

    def apply_with_params(self, img, params):
        if params.get("enabled", False):
            w, h = img.size
            ratio = float(params["ratio"])
            interpolation = params["interpolation"]
            resized_small = img.resize((int(w * ratio), int(h * ratio)), resample=lanczos)
            img = resized_small.resize((w, h), resample=interpolation)
        return img

    def __call__(self, img):
        return self.apply_with_params(img, self.sample_params())

class ColorJitter:
    def __init__(self, seed=2026, brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.1, 0.1), gamma=(0.7, 1.3)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.to_tensor = tvf.v2.Compose([tvf.v2.ToImage(), tvf.v2.ToDtype(torch.float32, scale=True)])

    def sample_params(self, rng=None):
        rng = self.rng if rng is None else rng
        return {
            "brightness": float(rng.uniform(*self.brightness)),
            "contrast": float(rng.uniform(*self.contrast)),
            "saturation": float(rng.uniform(*self.saturation)),
            "hue": float(rng.uniform(*self.hue)),
            "gamma": float(rng.uniform(*self.gamma)),
        }

    def apply_with_params(self, img, params):
        if not isinstance(img, torch.Tensor):
            # img = TF.to_tensor(img)
            img = self.to_tensor(img)

        img = TF.adjust_brightness(img, params["brightness"])
        img = TF.adjust_contrast(img, params["contrast"])
        img = TF.adjust_saturation(img, params["saturation"])
        img = TF.adjust_hue(img, params["hue"])
        img = TF.adjust_gamma(img, params["gamma"])

        img = TF.to_pil_image(img)
        return img

    def __call__(self, img):
        return self.apply_with_params(img, self.sample_params())


def _unwrap_partial_transform(transform):
    if isinstance(transform, functools.partial) and not transform.args and not transform.keywords:
        return transform.func
    return transform


def _sequence_transform_steps(transform):
    transform = _unwrap_partial_transform(transform)
    steps = getattr(transform, "transforms", None)
    if isinstance(steps, (list, tuple)):
        return steps
    return None


def sample_sequence_transform_params(transform, rng):
    """Sample one fixed parameter set for known photometric sequence augmentations."""
    steps = _sequence_transform_steps(transform)
    if steps is None:
        return None

    params = []
    has_sequence_step = False
    for step in steps:
        sample = getattr(step, "sample_params", None)
        apply = getattr(step, "apply_with_params", None)
        if callable(sample) and callable(apply):
            params.append(sample(rng))
            has_sequence_step = True
        else:
            params.append(None)
    return params if has_sequence_step else None


def apply_transform_with_sequence_params(transform, img, params):
    steps = _sequence_transform_steps(transform)
    if steps is None or params is None or len(params) != len(steps):
        return transform(img)

    out = img
    for step, step_params in zip(steps, params):
        apply = getattr(step, "apply_with_params", None)
        if step_params is not None and callable(apply):
            out = apply(out, step_params)
        else:
            out = step(out)
    return out

CustomNormJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), CustomNorm])

CustomNormJpegLoss = tvf.Compose([
    JpegLoss(prob=0.5, quality_range=(20, 100)),  
    CustomNorm 
])

CustomNormBlurring = tvf.Compose([
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])

CustomNormJpegLossBlurring = tvf.Compose([
    JpegLoss(prob=0.5, quality_range=(20, 100)), 
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])

CustomNormJitterJpegLossBlurring = tvf.Compose([
    ColorJitter(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3), 
        saturation=(0.7, 1.3), 
        hue=(-0.1, 0.1), 
        gamma=(0.7, 1.3) 
    ),
    JpegLoss(prob=0.5, quality_range=(20, 100)),
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])

JitterJpegLossBlurring = tvf.Compose([
    ColorJitter(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3), 
        saturation=(0.7, 1.3), 
        hue=(-0.1, 0.1), 
        gamma=(0.7, 1.3) 
    ),
    JpegLoss(prob=0.5, quality_range=(20, 100)),
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    tvf.ToTensor() 
])
