"""Adapters for official streaming point-map models used by mv_recon.

The wrappers keep each model's native causal state policy while sharing only
the protocol boundary: native RGB preprocessing, world-point extraction,
camera conversion, and mapping predictions back to the GT image grid.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from relpose.forward_timing import materialize_for_forward_timing, time_forward
from mv_recon.runtime_manifest import record_model_runtime


@dataclass(frozen=True)
class CropGeometry:
    src_h: int
    src_w: int
    resized_h: int
    resized_w: int
    crop_top: int
    crop_bottom: int

    @property
    def has_crop(self) -> bool:
        return self.crop_top > 0 or self.crop_bottom > 0

    def native_rows(self) -> slice:
        if not self.has_crop:
            return slice(0, self.src_h)
        y0 = int(round(self.crop_top * self.src_h / self.resized_h))
        y1 = int(round((self.resized_h - self.crop_bottom) * self.src_h / self.resized_h))
        y0 = min(max(y0, 0), self.src_h - 1)
        y1 = min(max(y1, y0 + 1), self.src_h)
        return slice(y0, y1)

    def rows_on_grid(self, grid_h: int) -> slice:
        """Map the observed native-image row interval onto another pixel grid."""
        native = self.native_rows()
        y0 = int(round(native.start * int(grid_h) / float(self.src_h)))
        y1 = int(round(native.stop * int(grid_h) / float(self.src_h)))
        y0 = min(max(y0, 0), int(grid_h) - 1)
        y1 = min(max(y1, y0 + 1), int(grid_h))
        return slice(y0, y1)


def official_crop_geometry(path: str, target: int = 518, patch: int = 14) -> CropGeometry:
    """Geometry implemented by VGGT/STream3R's official crop loader."""
    with Image.open(path) as image:
        src_w, src_h = image.size
    resized_w = int(target)
    resized_h = int(round(src_h * resized_w / src_w / patch) * patch)
    crop_top = max(0, (resized_h - target) // 2)
    crop_bottom = max(0, resized_h - target - crop_top)
    return CropGeometry(src_h, src_w, resized_h, resized_w, crop_top, crop_bottom)


def _resize_chw(tensor: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        tensor,
        size=tuple(int(v) for v in out_hw),
        mode="nearest",
    )


def map_world_points_to_native(
    points: np.ndarray,
    geometry: CropGeometry,
    data_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map model-grid XYZ to native GT pixels and expose the observed FOV."""
    points_t = torch.as_tensor(points).permute(0, 3, 1, 2).float()
    data_h, data_w = (int(data_size[0]), int(data_size[1]))
    rows = geometry.rows_on_grid(data_h)
    roi_h = rows.stop - rows.start
    resized = _resize_chw(points_t, (roi_h, data_w)).permute(0, 2, 3, 1).numpy()

    output = np.full((len(points), data_h, data_w, 3), np.nan, dtype=np.float32)
    observed = np.zeros((len(points), data_h, data_w), dtype=bool)
    output[:, rows, :, :] = resized
    observed[:, rows, :] = True
    return output, observed


def map_depth_to_native(
    depth: np.ndarray,
    geometry: CropGeometry,
    data_size: Tuple[int, int],
) -> np.ndarray:
    """Map model-grid local depth to the same native-image ROI as XYZ."""
    depth_t = torch.as_tensor(depth).float()
    while depth_t.ndim > 3 and depth_t.shape[-1] == 1:
        depth_t = depth_t.squeeze(-1)
    if depth_t.ndim != 3:
        raise ValueError(f"depth must be SxHxW, got {tuple(depth_t.shape)}")
    data_h, data_w = (int(data_size[0]), int(data_size[1]))
    rows = geometry.rows_on_grid(data_h)
    roi_h = rows.stop - rows.start
    resized = _resize_chw(depth_t[:, None], (roi_h, data_w))[:, 0].numpy()
    output = np.full((len(depth_t), data_h, data_w), np.nan, dtype=np.float32)
    output[:, rows, :] = resized
    return output


def _metric_output_indices(hydra_cfg: DictConfig, num_frames: int) -> Optional[List[int]]:
    configured = getattr(hydra_cfg, "mv_recon_output_indices", None)
    if configured is None:
        return None
    indices = [int(value) for value in configured]
    if not indices or min(indices) < 0 or max(indices) >= int(num_frames):
        raise ValueError(f"Invalid metric output indices for {num_frames} frames: {indices}")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("Metric output indices must be strictly increasing")
    return indices


def w2c_3x4_to_c2w(extrinsic: torch.Tensor) -> np.ndarray:
    extrinsic = torch.as_tensor(extrinsic).detach().cpu().float()
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=extrinsic.dtype)
    w2c = torch.cat([extrinsic, bottom.view(1, 1, 4).expand(len(extrinsic), -1, -1)], dim=1)
    return torch.linalg.inv(w2c).numpy()


class LazyOfficialFrames(Sequence[dict]):
    """Load and transfer one official-preprocessed RGB frame at a time."""

    def __init__(self, paths: Sequence[str], loader: Callable, device: str, mode: str = "crop"):
        self.paths = list(paths)
        self.loader = loader
        self.device = device
        self.mode = mode
        self._cache_index: Optional[int] = None
        self._cache_value: Optional[dict] = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._cache_index != index:
            image = self.loader([self.paths[index]], mode=self.mode)[0]
            self._cache_value = {"img": image.unsqueeze(0).to(self.device, non_blocking=True)}
            self._cache_index = index
        return self._cache_value


class _DiscardedDenseHead(torch.nn.Module):
    """Cheap shape-compatible output for a dense head ignored by pose inference."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, _tokens, images, **_kwargs):
        batch, frames = images.shape[:2]
        prediction = torch.empty(
            (batch, frames, 1, 1, self.channels),
            device=images.device,
            dtype=images.dtype,
        )
        confidence = torch.empty(
            (batch, frames, 1, 1), device=images.device, dtype=images.dtype
        )
        return prediction, confidence


_MISSING_HEAD = object()


@contextmanager
def _infinitevggt_pose_only_heads(model):
    """Suppress state-independent dense heads and always restore the model."""
    point_head = getattr(model, "point_head", _MISSING_HEAD)
    depth_head = getattr(model, "depth_head", _MISSING_HEAD)
    if point_head is not _MISSING_HEAD:
        model.point_head = _DiscardedDenseHead(3)
    if depth_head is not _MISSING_HEAD:
        model.depth_head = _DiscardedDenseHead(1)
    try:
        yield
    finally:
        if point_head is not _MISSING_HEAD:
            model.point_head = point_head
        if depth_head is not _MISSING_HEAD:
            model.depth_head = depth_head


@contextmanager
def _stream3r_pose_only_heads(model):
    """Disable STream3R dense readouts; its recurrent caches use no dense values."""
    point_head = getattr(model, "point_head", _MISSING_HEAD)
    depth_head = getattr(model, "depth_head", _MISSING_HEAD)
    if point_head is not _MISSING_HEAD:
        model.point_head = None
    if depth_head is not _MISSING_HEAD:
        model.depth_head = None
    try:
        yield
    finally:
        if point_head is not _MISSING_HEAD:
            model.point_head = point_head
        if depth_head is not _MISSING_HEAD:
            model.depth_head = depth_head


def _collect_recurrent_outputs(
    model,
    frames: Sequence[dict],
    num_frames: int,
    output_indices: Optional[Sequence[int]] = None,
):
    """Use the official frame-writer path so GPU outputs never accumulate."""
    requested = list(range(num_frames)) if output_indices is None else list(output_indices)
    output_positions = {frame_index: out_index for out_index, frame_index in enumerate(requested)}
    points = None
    depths = None
    pose_enc = None

    def writer(index, _frame, result):
        nonlocal points, depths, pose_enc
        pose = result["camera_pose"].squeeze(0).float().numpy()
        if pose_enc is None:
            pose_enc = np.empty((num_frames, *pose.shape), dtype=np.float32)
        pose_enc[index] = pose
        if index not in output_positions:
            return
        point = result["pts3d_in_other_view"].squeeze(0).float().numpy()
        depth = result["depth"].squeeze().float().numpy()
        if depth.shape != point.shape[:2]:
            raise ValueError(f"local depth {depth.shape} != point grid {point.shape[:2]}")
        if points is None:
            points = np.empty((len(requested), *point.shape), dtype=np.float32)
            depths = np.empty((len(requested), *depth.shape), dtype=np.float32)
        output_index = output_positions[index]
        points[output_index] = point
        depths[output_index] = depth

    with torch.inference_mode():
        model.inference(frames, frame_writer=writer, cache_results=False)
    if points is None or pose_enc is None:
        raise RuntimeError("Official streaming inference produced no frames")
    return points, depths, torch.from_numpy(pose_enc)


def _collect_recurrent_pose_outputs(
    model, frames: Sequence[dict], num_frames: int, timing_model=None, hydra_cfg=None
):
    """Collect only pose encodings while preserving the official recurrent state.

    The writer advertises that only camera pose should cross the GPU-to-CPU
    boundary. OVGGT still computes the dense signals required by its coverage
    anchor policy, while InfiniteVGGT may suppress its state-independent heads.
    """
    pose_enc = None
    image_hw = None

    def writer(index, frame, result):
        nonlocal pose_enc, image_hw
        pose = result["camera_pose"].squeeze(0).float().numpy()
        if pose_enc is None:
            pose_enc = np.empty((num_frames, *pose.shape), dtype=np.float32)
        pose_enc[index] = pose
        if image_hw is None:
            image_hw = tuple(int(value) for value in frame["img"].shape[-2:])

    # Supported by the pinned InfiniteVGGT/OVGGT inference loops. Point-cloud
    # inference uses a different writer without this marker and remains dense.
    writer.required_result_keys = frozenset({"camera_pose"})

    with torch.inference_mode():
        with time_forward(
            timing_model if timing_model is not None else model,
            hydra_cfg if hydra_cfg is not None else {},
            num_frames=num_frames,
        ):
            model.inference(frames, frame_writer=writer, cache_results=False)
    if pose_enc is None or image_hw is None:
        raise RuntimeError("Official streaming pose-only inference produced no frames")
    return torch.from_numpy(pose_enc), image_hw


def _infer_infinite_or_ovggt(filelist: List[str], model, hydra_cfg: DictConfig):
    frames = LazyOfficialFrames(filelist, model.image_loader, hydra_cfg.device, model.preprocess_mode)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    first_hw = frames[0]["img"].shape[-2:]
    record_model_runtime(
        model,
        input_hw=first_hw,
        input_storage_dtype="float32",
        forward_compute_dtype=str(dtype).removeprefix("torch."),
        preprocess=f"official_{model.family}_{model.preprocess_mode}_width518_patch14",
        online_state=(
            f"{model.family}+total_budget-{getattr(model, 'total_budget', 'official')}"
        ),
        forward_frames=len(filelist),
    )
    with torch.amp.autocast("cuda", dtype=dtype):
        points, depth, pose_enc = _collect_recurrent_outputs(
            model.model,
            frames,
            len(filelist),
            output_indices=_metric_output_indices(hydra_cfg, len(filelist)),
        )
    extrinsic, _ = model.pose_decoder(pose_enc.unsqueeze(0), points.shape[1:3])
    return points, w2c_3x4_to_c2w(extrinsic.squeeze(0)), depth


def _infer_infinite_or_ovggt_poses(filelist: List[str], model, hydra_cfg: DictConfig):
    frames = LazyOfficialFrames(filelist, model.image_loader, hydra_cfg.device, model.preprocess_mode)
    frames = materialize_for_forward_timing(
        frames,
        hydra_cfg,
        num_frames=len(filelist),
        label=getattr(model, "family", "official_streaming"),
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    first_hw = frames[0]["img"].shape[-2:]
    record_model_runtime(
        model,
        input_hw=first_hw,
        input_storage_dtype="float32",
        forward_compute_dtype=str(dtype).removeprefix("torch."),
        preprocess=f"official_{model.family}_{model.preprocess_mode}_width518_patch14",
        online_state=f"{model.family}+official_budget",
        forward_frames=len(filelist),
    )
    dense_context = (
        _infinitevggt_pose_only_heads(model.model)
        if model.family == "infinitevggt"
        else nullcontext()
    )
    with dense_context:
        with torch.amp.autocast("cuda", dtype=dtype):
            pose_enc, image_hw = _collect_recurrent_pose_outputs(
                model.model,
                frames,
                len(filelist),
                timing_model=model,
                hydra_cfg=hydra_cfg,
            )
    extrinsic, _ = model.pose_decoder(pose_enc.unsqueeze(0), image_hw)
    return w2c_3x4_to_c2w(extrinsic.squeeze(0))


def _infer_stream3r(filelist: List[str], model, hydra_cfg: DictConfig):
    session = model.make_session()
    point_parts = []
    depth_parts = []
    pose_parts = []
    chunk_size = max(1, int(model.chunk_size))
    requested = _metric_output_indices(hydra_cfg, len(filelist))
    requested_set = None if requested is None else set(requested)

    with torch.inference_mode():
        for start in range(0, len(filelist), chunk_size):
            chunk_paths = filelist[start : start + chunk_size]
            chunk = model.image_loader(chunk_paths, mode=model.preprocess_mode)
            if start == 0:
                record_model_runtime(
                    model,
                    input_hw=chunk.shape[-2:],
                    input_storage_dtype="float32",
                    forward_compute_dtype="fp32_outer+bf16_attention",
                    preprocess="official_stream3r_crop_width518_patch14",
                    online_state=(
                        f"stream3r_{model.mode}+window-{getattr(model, 'window_size', 5)}"
                        f"+chunk-{chunk_size}+geometry-native-world-points"
                    ),
                    forward_frames=len(filelist),
                )
            chunk = chunk.to(hydra_cfg.device, non_blocking=True)
            outputs = session.model(
                images=chunk,
                mode=model.mode,
                aggregator_kv_cache_list=session.aggregator_kv_cache_list,
                camera_head_kv_cache_list=session.camera_head_kv_cache_list,
            )
            local_indices = [
                index - start
                for index in range(start, start + len(chunk_paths))
                if requested_set is None or index in requested_set
            ]
            if local_indices:
                select = torch.as_tensor(local_indices, device=outputs["depth"].device)
                point_parts.append(
                    outputs["world_points"].index_select(1, select).detach().cpu()
                )
                depth_parts.append(outputs["depth"].index_select(1, select).detach().cpu())
            pose_parts.append(outputs["pose_enc"].detach().cpu())
            session.predictions = {"depth": outputs["depth"]}
            session._update_cache(
                outputs["aggregator_kv_cache_list"],
                outputs["camera_head_kv_cache_list"],
            )
            del outputs, chunk

    if not point_parts or not depth_parts:
        raise RuntimeError("STream3R produced no requested dense outputs")
    points = torch.cat(point_parts, dim=1).squeeze(0).numpy()
    depth = torch.cat(depth_parts, dim=1).squeeze(0).squeeze(-1).numpy()
    pose_enc = torch.cat(pose_parts, dim=1)
    model_hw = tuple(int(value) for value in point_parts[0].shape[2:4])
    extrinsic, _ = model.pose_decoder(pose_enc, model_hw)
    return (
        np.asarray(points, dtype=np.float32),
        w2c_3x4_to_c2w(extrinsic.squeeze(0)),
        np.asarray(depth, dtype=np.float32),
    )


def _infer_stream3r_poses(filelist: List[str], model, hydra_cfg: DictConfig):
    """Run the official session while retaining only compact pose encodings."""
    session = model.make_session()
    pose_parts = []
    image_hw = None
    chunk_size = max(1, int(model.chunk_size))

    with torch.inference_mode(), _stream3r_pose_only_heads(session.model):
        for start in range(0, len(filelist), chunk_size):
            chunk_paths = filelist[start : start + chunk_size]
            chunk = model.image_loader(chunk_paths, mode=model.preprocess_mode)
            if start == 0:
                record_model_runtime(
                    model,
                    input_hw=chunk.shape[-2:],
                    input_storage_dtype="float32",
                    forward_compute_dtype="fp32_outer+bf16_attention",
                    preprocess="official_stream3r_crop_width518_patch14",
                    online_state=f"stream3r_{model.mode}+window-{getattr(model, 'window_size', 5)}+chunk-{chunk_size}",
                    forward_frames=len(filelist),
                )
            chunk = chunk.to(hydra_cfg.device, non_blocking=True)
            image_hw = tuple(int(value) for value in chunk.shape[-2:])
            with time_forward(
                model,
                hydra_cfg,
                num_frames=len(chunk_paths),
                label="stream3r_chunk",
            ):
                outputs = session.model(
                    images=chunk,
                    mode=model.mode,
                    aggregator_kv_cache_list=session.aggregator_kv_cache_list,
                    camera_head_kv_cache_list=session.camera_head_kv_cache_list,
                )
                # The official window updater only reads depth.shape[2:4]. A
                # zero-element sentinel preserves that contract without running
                # or allocating the dense depth prediction.
                session.predictions = {
                    "depth": torch.empty((0, 0, *image_hw), device="cpu")
                }
                session._update_cache(
                    outputs["aggregator_kv_cache_list"],
                    outputs["camera_head_kv_cache_list"],
                )
            pose_parts.append(outputs["pose_enc"].detach().cpu())
            del outputs, chunk

    if not pose_parts or image_hw is None:
        raise RuntimeError("STream3R pose-only inference produced no frames")
    pose_enc = torch.cat(pose_parts, dim=1)
    extrinsic, _ = model.pose_decoder(pose_enc, image_hw)
    return w2c_3x4_to_c2w(extrinsic.squeeze(0))


def _infer_native(filelist: List[str], model, hydra_cfg: DictConfig):
    if model.family in {"infinitevggt", "ovggt"}:
        return _infer_infinite_or_ovggt(filelist, model, hydra_cfg)
    if model.family == "stream3r":
        return _infer_stream3r(filelist, model, hydra_cfg)
    raise ValueError(f"Unsupported official streaming family: {model.family}")


def _infer_native_poses(filelist: List[str], model, hydra_cfg: DictConfig):
    if model.family in {"infinitevggt", "ovggt"}:
        return _infer_infinite_or_ovggt_poses(filelist, model, hydra_cfg)
    if model.family == "stream3r":
        return _infer_stream3r_poses(filelist, model, hydra_cfg)
    raise ValueError(f"Unsupported official streaming family: {model.family}")


def infer_mv_pointclouds(
    filelist: List[str], model, hydra_cfg: DictConfig, data_size: Tuple[int, int]
):
    points, c2w, local_depth = _infer_native(filelist, model, hydra_cfg)
    geometry = official_crop_geometry(filelist[0], model.img_size, model.patch_size)
    points, observed = map_world_points_to_native(points, geometry, data_size)
    local_depth = map_depth_to_native(local_depth, geometry, data_size)
    pred_mask = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(local_depth)
        & (local_depth > 1e-4)
    )
    return points, c2w, pred_mask, observed


def infer_cameras_c2w(filelist: List[str], model, hydra_cfg: DictConfig):
    c2w = _infer_native_poses(filelist, model, hydra_cfg)
    return torch.from_numpy(c2w[:, :3]), None


def infer_cameras_w2c(filelist: List[str], model, hydra_cfg: DictConfig):
    c2w = _infer_native_poses(filelist, model, hydra_cfg)
    return torch.from_numpy(np.linalg.inv(c2w)), None
