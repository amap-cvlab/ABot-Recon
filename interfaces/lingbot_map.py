from functools import lru_cache
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from omegaconf import DictConfig
from PIL import Image, ImageOps

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import LingBotMAP
from relpose.forward_timing import time_forward
from mv_recon.runtime_manifest import record_model_runtime


@lru_cache(maxsize=1)
def _lingbot_utils():
    """Import official LingBot utilities only when that adapter is executed."""
    try:
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        from lingbot_map.utils.pose_enc import (
            pose_encoding_to_extri_intri as pose_encoding_to_extri_intri_impl,
        )
        from lingbot_map.utils.geometry import (
            closed_form_inverse_se3 as closed_form_inverse_se3_impl,
            unproject_depth_map_to_point_map as unproject_depth_map_to_point_map_impl,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LingBot-Map is required only for LingBot evaluation. Install its "
            "official source under third_party/LingBot-Map or set "
            "model.lingbot_map.cfg.source_root to its checkout."
        ) from exc
    return (
        load_and_preprocess_images,
        pose_encoding_to_extri_intri_impl,
        closed_form_inverse_se3_impl,
        unproject_depth_map_to_point_map_impl,
    )


def lingbot_load_images(*args, **kwargs):
    return _lingbot_utils()[0](*args, **kwargs)


def pose_encoding_to_extri_intri(*args, **kwargs):
    return _lingbot_utils()[1](*args, **kwargs)


def closed_form_inverse_se3(*args, **kwargs):
    return _lingbot_utils()[2](*args, **kwargs)


def unproject_depth_map_to_point_map(*args, **kwargs):
    return _lingbot_utils()[3](*args, **kwargs)


_OFFICIAL_LONG_POSE_HW = {
    "kitti": (280, 504),
    "vbr": (280, 504),
    "oxford": (378, 518),
}


class LazyLingBotImages:
    """Tensor-like sequence that decodes only the slice requested by LingBot."""

    def __init__(
        self,
        filelist,
        model,
        device,
        prepare_width=0,
        prepare_interpolation="linear",
        loader_mode="area_budget",
    ):
        if not filelist:
            raise ValueError("LingBot requires at least one image")
        self.filelist = list(filelist)
        self.model = model
        self._device = torch.device(device)
        self.prepare_width = int(prepare_width or 0)
        self.prepare_interpolation = str(prepare_interpolation)
        self.loader_mode = str(loader_mode)
        first = self._load(0, 1)
        self._shape = (1, len(self.filelist), *first.shape[2:])
        self._first = first

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 5

    @property
    def device(self):
        return self._device

    def __len__(self):
        return self._shape[0]

    def dim(self):
        return 5

    def float(self):
        return self

    def to(self, device, non_blocking=False):
        if torch.device(device).type == "cpu":
            return self
        if torch.device(device) != self.device:
            raise ValueError(f"LazyLingBotImages is bound to {self.device}, got {device}")
        return self

    def _load(self, start, stop):
        if self.loader_mode == "long_pose_official":
            return _load_long_pose_official_tensor(
                self.filelist[start:stop], self.model, str(self._device)
            )
        if self.loader_mode != "area_budget":
            raise ValueError(f"Unknown LingBot lazy loader mode: {self.loader_mode!r}")
        return _load_area_budget(
            self.filelist[start:stop],
            area_budget=int(getattr(self.model, "area_budget", 255000)),
            align=int(getattr(self.model, "align", 14)),
            device=str(self._device),
            prepare_width=self.prepare_width,
            prepare_interpolation=self.prepare_interpolation,
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("expected images[batch, frames]")
        batch, frames = key
        if batch not in (slice(None, None, None), 0):
            raise IndexError("LingBot lazy input has batch size one")
        if isinstance(frames, int):
            start = frames if frames >= 0 else len(self.filelist) + frames
            stop = start + 1
        else:
            start, stop, step = frames.indices(len(self.filelist))
            if step != 1:
                raise IndexError("strided slices are unsupported")
        if start == 0 and self._first is not None:
            first, self._first = self._first, None
            if stop == 1:
                return first
            return torch.cat([first, self._load(1, stop)], dim=1)
        return self._load(start, stop)


def _area_budget_hw(
    orig_w: int,
    orig_h: int,
    area_budget: int = 255000,
    align: int = 14,
) -> Tuple[int, int]:
    """Official bench resize: downscale so W*H <= area_budget, dims multiple of align."""
    scale = min(np.sqrt(area_budget / float(orig_w * orig_h)), 1.0)
    new_w = max((int(orig_w * scale) // align) * align, align)
    new_h = max((int(orig_h * scale) // align) * align, align)
    return new_w, new_h


def _prepare_width_aligned_hw(orig_w: int, orig_h: int, prepare_width: int = 518) -> Tuple[int, int]:
    """Resize to a fixed width and floor height to the patch-size multiple."""
    target_width = int(prepare_width)
    aspect_ratio = float(orig_h) / float(orig_w)
    target_height = int(target_width * aspect_ratio)
    target_height = max((target_height // 14) * 14, 14)
    return target_width, target_height


def _load_area_budget(
    filelist: List[str],
    area_budget: int,
    align: int,
    device: str,
    prepare_width: int = 0,
    prepare_interpolation: str = "linear",
) -> torch.Tensor:
    """Load RGB like official prepare (+optional) then area_budget.

    If ``prepare_width > 0`` (Oxford), first resize raw to the requested width
    with height floor-to-14, then apply the area budget. Otherwise apply the
    area budget directly to native pixels.
    """
    to_tensor = TF.ToTensor()
    imgs = []
    target_wh: Optional[Tuple[int, int]] = None
    for path in filelist:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            rgb = np.asarray(pil)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]

        # Oxford prepare step before area_budget / model forward.
        if prepare_width is not None and int(prepare_width) > 0:
            pw, ph = _prepare_width_aligned_hw(w0, h0, int(prepare_width))
            if (w0, h0) != (pw, ph):
                if prepare_interpolation == "pil_lanczos":
                    rgb = np.asarray(
                        Image.fromarray(rgb).resize(
                            (pw, ph), Image.Resampling.LANCZOS
                        ),
                        dtype=np.uint8,
                    )
                elif prepare_interpolation == "linear":
                    rgb = cv2.resize(
                        rgb, (pw, ph), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    raise ValueError(
                        "prepare_interpolation must be linear or pil_lanczos, "
                        f"got {prepare_interpolation!r}"
                    )
                h0, w0 = ph, pw

        tw, th = _area_budget_hw(w0, h0, area_budget=area_budget, align=align)
        if target_wh is None:
            target_wh = (tw, th)
        elif target_wh != (tw, th):
            tw, th = target_wh
        if (w0, h0) != (tw, th):
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        imgs.append(to_tensor(rgb))
    images = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, S, 3, H, W)
    return images.to(device)


def _long_pose_profile(model: LingBotMAP) -> str:
    dataset_name = str(getattr(model, "eval_dataset_name", "")).lower()
    if dataset_name == "kitti-long":
        return "kitti"
    if dataset_name == "vbr-long":
        return "vbr"
    if dataset_name == "oxford_spires_processed-long":
        return "oxford"
    raise ValueError(
        "LingBot long_pose_official preprocessing requires eval_dataset_name "
        f"to be kitti-long, vbr-long, or oxford_spires_processed-long; got {dataset_name!r}"
    )


def _load_long_pose_official_tensor(
    filelist: List[str], model: LingBotMAP, device: str
) -> torch.Tensor:
    """Eager tensor implementation of LingBot's released long-eval transforms."""
    profile = _long_pose_profile(model)
    frames = []
    for path in filelist:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        width, height = image.size
        if profile in {"kitti", "vbr"}:
            target_w, target_h = 504, 280
            scale = max(target_w / float(width), target_h / float(height))
            resized_w = int(round(width * scale))
            resized_h = int(round(height * scale))
            rgb = cv2.resize(
                np.asarray(image),
                (resized_w, resized_h),
                interpolation=cv2.INTER_AREA,
            )
            top = (resized_h - target_h) // 2
            left = (resized_w - target_w) // 2
            rgb = rgb[top : top + target_h, left : left + target_w]
            if rgb.shape[:2] != (target_h, target_w):
                raise ValueError(f"Invalid LingBot {profile} crop for {path}: {rgb.shape}")
            frames.append(TF.ToTensor()(rgb))
        else:
            target_w = 518
            target_h = max(
                model.patch_size,
                int((target_w * height / float(width)) // model.patch_size)
                * model.patch_size,
            )
            resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            frames.append(TF.ToTensor()(resized))
    if not frames:
        raise FileNotFoundError("Empty filelist for LingBot long-pose preprocessing")
    return torch.stack(frames, dim=0).unsqueeze(0).to(device)


def _load_long_pose_official(
    filelist: List[str], model: LingBotMAP, device: str
) -> Union[torch.Tensor, LazyLingBotImages]:
    """Dataset-specific RGB transforms used by LingBot's released long eval.

    Long sequences stay lazy so only the scale-frame/current-frame slice is
    resident on the GPU. Short sequences retain the original eager behavior.
    """
    if len(filelist) > 64:
        return LazyLingBotImages(
            filelist,
            model,
            device,
            loader_mode="long_pose_official",
        )
    return _load_long_pose_official_tensor(filelist, model, device)


def _load_and_preprocess(
    filelist: List[str],
    model: LingBotMAP,
    device: str,
    prepare_width: int = 0,
    prepare_interpolation: str = "linear",
) -> torch.Tensor:
    """Load images using the selected official LingBot evaluation protocol."""
    preprocess = getattr(model, "preprocess_mode", "area_budget")
    if preprocess == "area_budget" and len(filelist) > 64:
        return LazyLingBotImages(
            filelist,
            model,
            device,
            prepare_width,
            prepare_interpolation,
        )
    if preprocess == "area_budget":
        return _load_area_budget(
            filelist,
            area_budget=int(getattr(model, "area_budget", 255000)),
            align=int(getattr(model, "align", 14)),
            device=device,
            prepare_width=int(prepare_width or 0),
            prepare_interpolation=prepare_interpolation,
        )
    if preprocess == "crop":
        images = lingbot_load_images(
            filelist, mode="crop", image_size=model.img_size, patch_size=model.patch_size
        )
        return images.to(device)
    if preprocess == "long_pose_official":
        return _load_long_pose_official(filelist, model, device)
    raise ValueError(
        f"Unknown lingbot preprocess_mode={preprocess!r} "
        "(use area_budget|crop|long_pose_official)"
    )


def _prepare_width_from_cfg(hydra_cfg: DictConfig) -> int:
    return int(getattr(hydra_cfg, "lingbot_prepare_width", 0) or 0)


def _prepare_interpolation_from_cfg(hydra_cfg: DictConfig) -> str:
    return str(getattr(hydra_cfg, "lingbot_prepare_interpolation", "linear"))


def _assert_expected_input_hw(images, hydra_cfg: DictConfig, model=None) -> None:
    if model is not None and getattr(model, "preprocess_mode", None) == "long_pose_official":
        profile = _long_pose_profile(model)
        expected = _OFFICIAL_LONG_POSE_HW[profile]
        actual = tuple(int(value) for value in images.shape[-2:])
        if actual != expected:
            raise RuntimeError(
                "LingBot official long-pose input shape mismatch: "
                f"profile={profile}, actual HxW={actual}, expected={expected}"
            )
    configured = getattr(hydra_cfg, "lingbot_expected_input_hw", None)
    if configured is None:
        return
    expected = tuple(int(value) for value in configured)
    actual = tuple(int(value) for value in images.shape[-2:])
    if actual != expected:
        raise RuntimeError(
            f"LingBot input shape mismatch: actual HxW={actual}, expected={expected}"
        )


def _record_runtime(model, images, filelist):
    amp = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    preprocess = str(getattr(model, "preprocess_mode", "area_budget"))
    keyframe_interval = getattr(model, "keyframe_interval", None)
    if keyframe_interval is None:
        keyframe_interval = (
            (len(filelist) + 319) // 320
            if getattr(model, "mode", "streaming") == "streaming" and len(filelist) > 320
            else 1
        )
    record_model_runtime(
        model,
        input_hw=images.shape[-2:],
        input_storage_dtype="float32",
        forward_compute_dtype=amp,
        preprocess=f"official_lingbot_{preprocess}",
        online_state=(
            "streaming_paged_kv"
            f"+scale_frames-{model.num_scale_frames}"
            f"+window-{model.window_size}"
        ),
        forward_frames=len(filelist),
        extra={
            "keyframe_interval": int(keyframe_interval),
            "area_budget": getattr(model, "area_budget", None),
            "patch_size": getattr(model, "patch_size", None),
        },
    )


def _squeeze_batch(tensor):
    """Squeeze leading batch dim if shape is (1, ...)."""
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _postprocess(predictions, images):
    """Post-process raw predictions: decode pose_enc to c2w extrinsics, move to CPU."""
    extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    predictions["extrinsic"] = extrinsic_c2w
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = _squeeze_batch(predictions[key].detach().cpu())

    return predictions


def infer_cameras_c2w(filelist: List[str], model: LingBotMAP, hydra_cfg: DictConfig):
    """Standard interface: returns (N, 3, 4) c2w poses and (N, 3, 3) intrinsics."""
    images = _load_and_preprocess(
        filelist,
        model,
        hydra_cfg.device,
        prepare_width=_prepare_width_from_cfg(hydra_cfg),
        prepare_interpolation=_prepare_interpolation_from_cfg(hydra_cfg),
    )
    _assert_expected_input_hw(images, hydra_cfg, model)
    _record_runtime(model, images, filelist)
    if hasattr(model, "reset_kv_cache_manager"):
        model.reset_kv_cache_manager()
    skip_dense_heads = bool(hydra_cfg.get("pose_only_skip_dense_heads", True))
    with time_forward(model, hydra_cfg, num_frames=len(filelist)):
        predictions = model(images, pose_only=skip_dense_heads)

    extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    extrinsic_c2w = _squeeze_batch(extrinsic_c2w)
    intrinsic = _squeeze_batch(intrinsic)

    return extrinsic_c2w.detach().cpu(), intrinsic.detach().cpu()


def infer_cameras_w2c(filelist: List[str], model: LingBotMAP, hydra_cfg: DictConfig):
    """Standard interface: returns (N, 4, 4) w2c poses by inverting c2w."""
    c2w_3x4, intrinsic = infer_cameras_c2w(filelist, model, hydra_cfg)
    w2c_4x4 = closed_form_inverse_se3(c2w_3x4.numpy())
    return torch.from_numpy(w2c_4x4), intrinsic


def _resize_depth_to_hw(
    depth_map: torch.Tensor,
    target_hw: Tuple[int, int],
    nearest: bool,
) -> torch.Tensor:
    from mv_recon.pc_infer_utils import resize_map_to_hw

    return resize_map_to_hw(depth_map, target_hw, nearest=nearest)


def infer_mv_pointclouds(
    filelist: List[str],
    model: LingBotMAP,
    hydra_cfg: DictConfig,
    data_size: Tuple[int, int],
):
    """Multi-view PC inference for lingbot-aligned metrics.

    Preprocess default = official ``area_budget`` (LINEAR resize).

    If ``hydra_cfg.nearest_depth_to_gt`` is True (lingbot configs enable it):
      decode pose at model res → scale K anisotropically → NEAREST-resize depth
      to ``data_size`` (GT native HxW when dataset ``load_img_size<=0``).
    Else:
      bilinear-resize depth to ``data_size`` (legacy evaluation path).

    Oxford uses ``hydra_cfg.lingbot_prepare_width=518`` before area_budget.

    Returns:
        (world_points, c2w_4x4, pred_mask) where pred_mask is depth > 1e-4
        (official BSSLoader convention).
    """
    from mv_recon.pc_infer_utils import (
        nearest_depth_to_gt_enabled,
        pred_mask_from_depth,
    )

    images = _load_and_preprocess(
        filelist,
        model,
        hydra_cfg.device,
        prepare_width=_prepare_width_from_cfg(hydra_cfg),
        prepare_interpolation=_prepare_interpolation_from_cfg(hydra_cfg),
    )
    _assert_expected_input_hw(images, hydra_cfg)
    _record_runtime(model, images, filelist)
    if hasattr(model, "reset_kv_cache_manager"):
        model.reset_kv_cache_manager()
    configured = getattr(hydra_cfg, "mv_recon_output_indices", None)
    dense_indices = None if configured is None else [int(i) for i in configured]
    predictions = model(images, dense_output_indices=dense_indices)

    model_hw = images.shape[-2:]  # (H, W)
    nearest_to_gt = nearest_depth_to_gt_enabled(hydra_cfg)
    target_hw = (int(data_size[0]), int(data_size[1]))

    # Decode at model resolution (official), then scale K if upsampling to GT.
    extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_size_hw=model_hw
    )
    extrinsic_c2w = _squeeze_batch(extrinsic_c2w).detach().cpu()
    intrinsic = _squeeze_batch(intrinsic).detach().cpu().clone()
    full_extrinsic_c2w = extrinsic_c2w

    if (model_hw[0], model_hw[1]) != target_hw:
        scale_h = target_hw[0] / float(model_hw[0])
        scale_w = target_hw[1] / float(model_hw[1])
        intrinsic[:, 0, :] *= scale_w
        intrinsic[:, 1, :] *= scale_h

    depth_map = predictions["depth"]
    if depth_map.dim() == 5:
        depth_map = depth_map.squeeze(0)
    if depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    depth_map = depth_map.detach().float().cpu()
    if configured is not None:
        indices = torch.as_tensor(list(configured), dtype=torch.long)
        intrinsic = intrinsic.index_select(0, indices)
        extrinsic_c2w = extrinsic_c2w.index_select(0, indices)
        if not predictions.get("dense_output_indices_applied", False):
            # Clean upstream LingBot returns dense maps for every frame. The
            # optional compatibility patch performs this selection in-model.
            depth_map = depth_map.index_select(0, indices)
    w2c_4x4 = closed_form_inverse_se3(extrinsic_c2w.numpy())
    extrinsic_w2c = torch.from_numpy(w2c_4x4[:, :3, :])
    depth_map = _resize_depth_to_hw(depth_map, target_hw, nearest=nearest_to_gt)

    pred_mask = pred_mask_from_depth(depth_map)

    world_points = unproject_depth_map_to_point_map(
        depth_map.unsqueeze(-1),
        extrinsic_w2c,
        intrinsic,
    )

    c2w_4x4 = np.zeros((full_extrinsic_c2w.shape[0], 4, 4), dtype=np.float64)
    c2w_4x4[:, :3, :] = full_extrinsic_c2w.numpy().astype(np.float64)
    c2w_4x4[:, 3, 3] = 1.0
    return (
        world_points,
        c2w_4x4,
        pred_mask,
        None,
        None,
        {"pred_depth": depth_map.numpy().astype(np.float32, copy=False)},
    )
