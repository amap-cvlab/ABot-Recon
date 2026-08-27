import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import List, Tuple

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import LongStream
from relpose.forward_timing import time_forward
from mv_recon.runtime_manifest import record_model_runtime


def _point_head_valid_mask(
    world_points: np.ndarray, local_points: np.ndarray
) -> np.ndarray:
    """Point-head validity from finite XYZ and positive camera-frame Z."""
    world_points = np.asarray(world_points)
    local_points = np.asarray(local_points)
    if world_points.shape != local_points.shape:
        raise ValueError(
            f"world/local point grids disagree: {world_points.shape} vs {local_points.shape}"
        )
    return (
        np.isfinite(world_points).all(axis=-1)
        & np.isfinite(local_points).all(axis=-1)
        & (local_points[..., 2] > 1e-4)
    )

def _load_eager(filelist: List[str], model: LongStream) -> torch.Tensor:
    views = model.image_loader(
        filelist,
        size=model.img_size,
        verbose=False,
        crop=False,
        patch_size=model.patch_size,
    )
    imgs = torch.cat([view["img"] for view in views], dim=0)
    return (imgs.unsqueeze(0) + 1.0) / 2.0


class LazyLongStreamImages:
    """Tensor-like CPU sequence loaded only when LongStream requests a chunk."""

    def __init__(self, filelist: List[str], model: LongStream):
        if not filelist:
            raise ValueError("LongStream requires at least one image")
        self.filelist = list(filelist)
        self.model = model
        first = _load_eager(self.filelist[:1], self.model)
        self._shape = (1, len(self.filelist), *first.shape[2:])
        self._first = first

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return torch.device("cpu")

    def dim(self):
        return 5

    def float(self):
        # Dust3R preprocessing already returns float32.
        return self

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("expected images[batch, frames]")
        batch, frames = key
        if batch not in (0, slice(None, None, None)):
            raise IndexError("LongStream lazy input has batch size one")
        if isinstance(frames, int):
            start = frames if frames >= 0 else len(self.filelist) + frames
            stop = start + 1
        elif isinstance(frames, slice):
            start, stop, step = frames.indices(len(self.filelist))
            if step != 1:
                raise IndexError("strided lazy image slices are unsupported")
        else:
            raise IndexError("frame index must be int or slice")
        if stop <= start:
            return torch.empty((1, 0, *self._shape[2:]), dtype=torch.float32)
        if start == 0 and self._first is not None:
            result = self._first
            self._first = None
            if stop > 1:
                result = torch.cat((result, _load_eager(self.filelist[1:stop], self.model)), dim=1)
            return result
        return _load_eager(self.filelist[start:stop], self.model)


def _load_and_preprocess(filelist: List[str], model: LongStream, *, eager: bool = False):
    """Load images using LongStream's dust3r-style loader.

    Returns (1, S, C, H, W) tensor in [0, 1] range.
    """
    if eager:
        return _load_eager(filelist, model)
    return LazyLongStreamImages(filelist, model)


def _record_runtime(model, images, filelist):
    inference_mode = getattr(model, "inference_mode", "streaming")
    window_size = getattr(model, "window_size", "official-default")
    keyframe_stride = getattr(model, "keyframe_stride", "official-default")
    refresh = getattr(model, "refresh", "official-default")
    record_model_runtime(
        model,
        input_hw=images.shape[-2:],
        input_storage_dtype="float32",
        forward_compute_dtype="fp32_outer+bf16_attention",
        preprocess="official_longstream_long_edge_518_patch14_crop_false",
        online_state=(
            f"{inference_mode}+window-{window_size}"
            f"+keyframe_stride-{keyframe_stride}+refresh-{refresh}"
        ),
        forward_frames=len(filelist),
    )


def _w2c_3x4_to_c2w_3x4(w2c_3x4: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 4) w2c to (N, 3, 4) c2w via matrix inversion."""
    num_frames = w2c_3x4.shape[0]
    # Build 4x4
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=w2c_3x4.dtype).expand(num_frames, 1, 4)
    w2c_4x4 = torch.cat([w2c_3x4, bottom], dim=1)  # (N, 4, 4)
    c2w_4x4 = torch.inverse(w2c_4x4)
    return c2w_4x4[:, :3, :]  # (N, 3, 4)


def _w2c_3x4_to_w2c_4x4(w2c_3x4: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 4) w2c to (N, 4, 4) w2c."""
    num_frames = w2c_3x4.shape[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=w2c_3x4.dtype).expand(num_frames, 1, 4)
    return torch.cat([w2c_3x4, bottom], dim=1)  # (N, 4, 4)


def infer_cameras_c2w(filelist: List[str], model: LongStream, hydra_cfg: DictConfig):
    """Standard interface: returns (N, 3, 4) c2w poses and (N, 3, 3) intrinsics.

    LongStream's pose_encoding_to_extri_intri outputs w2c, so we invert to get c2w.
    Returns float64 tensors for numerical precision in eval.
    Note: model forward must run in float32; the caller's float64 autocast is
    disabled here to avoid dtype mismatch inside LayerNorm / patch_embed.
    """
    images = _load_and_preprocess(
        filelist, model, eager=bool(hydra_cfg.get("measure_forward_fps", False))
    )
    _record_runtime(model, images, filelist)
    if bool(hydra_cfg.get("measure_forward_fps", False)):
        # Keep disk I/O, preprocessing, and host-to-device transfer outside the
        # forward-only timer, matching the other benchmark adapters.
        images = images.to(hydra_cfg.device, non_blocking=True)

    with torch.amp.autocast(device_type=hydra_cfg.device, enabled=False):
        with time_forward(model, hydra_cfg, num_frames=len(filelist)):
            predictions = model(
                images.float(),
                pose_only=bool(hydra_cfg.get("pose_only_skip_dense_heads", True)),
            )

    extrinsic_w2c = predictions["extrinsic_w2c"].detach().cpu()  # (S, 3, 4)
    intrinsic = predictions["intrinsic"].detach().cpu()           # (S, 3, 3)

    extrinsic_c2w = _w2c_3x4_to_c2w_3x4(extrinsic_w2c)   # (S, 3, 4)

    return extrinsic_c2w, intrinsic


def infer_cameras_w2c(filelist: List[str], model: LongStream, hydra_cfg: DictConfig):
    """Standard interface: returns (N, 4, 4) w2c poses and (N, 3, 3) intrinsics.

    Returns float64 tensors to match GT extrinsics dtype (required by bmm in eval).
    Note: model forward must run in float32; the caller's float64 autocast is
    disabled here to avoid dtype mismatch inside LayerNorm / patch_embed.
    """
    images = _load_and_preprocess(filelist, model)
    _record_runtime(model, images, filelist)

    # Disable any active autocast so the model runs in its native float32 dtype.
    with torch.amp.autocast(device_type=hydra_cfg.device, enabled=False):
        predictions = model(
            images.float(),
            pose_only=bool(hydra_cfg.get("pose_only_skip_dense_heads", True)),
        )

    extrinsic_w2c = predictions["extrinsic_w2c"]   # (S, 3, 4)
    intrinsic = predictions["intrinsic"]             # (S, 3, 3)

    w2c_4x4 = _w2c_3x4_to_w2c_4x4(extrinsic_w2c)  # (S, 4, 4)

    return w2c_4x4.double(), intrinsic.double()


def infer_mv_pointclouds(
    filelist: List[str], model: LongStream, hydra_cfg: DictConfig, data_size: Tuple[int, int]
):
    """Standard interface for multi-view reconstruction.

    Uses the official demo-default point-head branch. LongStream also exposes a
    depth-unprojection diagnostic, but the common protocol intentionally reports
    one unambiguous LongStream result.
    """
    images = _load_and_preprocess(filelist, model)
    _record_runtime(model, images, filelist)
    configured = getattr(hydra_cfg, "mv_recon_output_indices", None)
    dense_indices = (
        None if configured is None else [int(index) for index in configured]
    )
    predictions = model(images, dense_output_indices=dense_indices)

    extrinsic_w2c = predictions["extrinsic_w2c"]   # (S, 3, 4)
    from mv_recon.pc_infer_utils import (
        nearest_depth_to_gt_enabled,
        resize_map_to_hw,
    )

    data_h, data_w = int(data_size[0]), int(data_size[1])
    nearest = nearest_depth_to_gt_enabled(hydra_cfg)
    camera_maps = predictions.get("camera_points")
    if camera_maps is None:
        raise RuntimeError("LongStream point-head output is missing")
    full_c2w = _w2c_3x4_to_c2w_3x4(extrinsic_w2c).numpy()
    if configured is not None:
        indices = torch.as_tensor(list(configured), dtype=torch.long)
        if not predictions.get("dense_output_indices_applied", False):
            camera_maps = camera_maps.index_select(0, indices)
        extrinsic_w2c = extrinsic_w2c.index_select(0, indices)
    camera_maps = resize_map_to_hw(
        camera_maps.permute(0, 3, 1, 2), (data_h, data_w), nearest=nearest
    ).permute(0, 2, 3, 1)

    # Both official branches produce camera-space XYZ and use the decoded pose.
    num_frames = camera_maps.shape[0]
    all_world_points = []
    for i in range(num_frames):
        w2c_i = extrinsic_w2c[i]
        rotation = w2c_i[:3, :3]
        translation = w2c_i[:3, 3]
        camera_points = camera_maps[i]
        pts_flat = camera_points.reshape(-1, 3)
        world_pts = (rotation.t() @ (pts_flat.t() - translation[:, None])).t()
        all_world_points.append(world_pts.reshape(camera_points.shape).numpy())

    world_points = np.stack(all_world_points, axis=0)  # (S, H, W, 3)
    pred_mask = _point_head_valid_mask(world_points, camera_maps.numpy())
    return world_points, full_c2w, pred_mask
