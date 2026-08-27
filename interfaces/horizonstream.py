"""HorizonStream adapters for the evaluation suite (pose + multi-view point clouds)."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import HorizonStreamEval
from mv_recon.runtime_manifest import record_model_runtime
from relpose.forward_timing import time_forward

_HS_PKG = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "..",
        "HorizonStream",
    )
)
if _HS_PKG not in sys.path:
    sys.path.insert(0, _HS_PKG)

from horizonstream.utils.depth import unproject_depth_to_points
from horizonstream.utils.vendor.dust3r.utils.image import load_images_for_eval


class LazyHorizonImages:
    """Tensor-like RGB sequence decoded and transferred per official chunk."""

    def __init__(self, filelist, img_size, patch_size, crop, device):
        if not filelist:
            raise ValueError("HorizonStream requires at least one image")
        self.filelist = list(filelist)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.crop = bool(crop)
        self._device = torch.device(device)
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

    def dim(self):
        return 5

    def float(self):
        return self

    def _load(self, start, stop):
        views = load_images_for_eval(
            self.filelist[start:stop],
            size=self.img_size,
            verbose=False,
            crop=self.crop,
            patch_size=self.patch_size,
        )
        imgs = torch.cat([view["img"] for view in views], dim=0)
        return ((imgs.unsqueeze(0) + 1.0) / 2.0).to(
            self._device, non_blocking=True
        )

    def _load_indices(self, indices):
        views = load_images_for_eval(
            [self.filelist[index] for index in indices],
            size=self.img_size,
            verbose=False,
            crop=self.crop,
            patch_size=self.patch_size,
        )
        imgs = torch.cat([view["img"] for view in views], dim=0)
        return ((imgs.unsqueeze(0) + 1.0) / 2.0).to(
            self._device, non_blocking=True
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("expected images[batch, frames]")
        batch, frames = key
        if batch not in (slice(None, None, None), 0):
            raise IndexError("Horizon lazy input has batch size one")
        if isinstance(frames, (list, tuple, np.ndarray)):
            indices = [int(index) for index in frames]
            if not indices:
                raise IndexError("empty Horizon frame index list")
            if min(indices) < 0 or max(indices) >= len(self.filelist):
                raise IndexError("Horizon frame index is out of range")
            return self._load_indices(indices)
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


def horizon_model_roi_on_gt(
    model_hw: Tuple[int, int],
    data_hw: Tuple[int, int],
    *,
    img_size: int,
    crop: bool,
) -> Tuple[slice, slice]:
    """Map Horizon's center-cropped model canvas to the matching GT ROI.

    The official loader first preserves aspect ratio while resizing the long
    edge to ``img_size``, then center-crops to patch multiples.  A cropped
    prediction therefore observes only a center ROI of the uncropped GT image.
    """
    data_h, data_w = (int(data_hw[0]), int(data_hw[1]))
    model_h, model_w = (int(model_hw[0]), int(model_hw[1]))
    if not crop:
        return slice(0, data_h), slice(0, data_w)

    long_edge = max(data_h, data_w)
    resized_h = int(round(data_h * float(img_size) / float(long_edge)))
    resized_w = int(round(data_w * float(img_size) / float(long_edge)))
    if model_h > resized_h or model_w > resized_w:
        raise ValueError(
            f"model canvas {model_h}x{model_w} exceeds aspect-preserving "
            f"canvas {resized_h}x{resized_w}"
        )

    top = (resized_h - model_h) / 2.0
    left = (resized_w - model_w) / 2.0
    y0 = int(round(top * data_h / float(resized_h)))
    y1 = int(round((top + model_h) * data_h / float(resized_h)))
    x0 = int(round(left * data_w / float(resized_w)))
    x1 = int(round((left + model_w) * data_w / float(resized_w)))
    y0, y1 = max(0, y0), min(data_h, y1)
    x0, x1 = max(0, x0), min(data_w, x1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"empty Horizon GT ROI: y={y0}:{y1}, x={x0}:{x1}")
    return slice(y0, y1), slice(x0, x1)


def postprocess_horizon_points_to_gt(
    world_model: torch.Tensor,
    depth_model: torch.Tensor,
    data_size: Tuple[int, int],
    *,
    img_size: int,
    crop: bool,
    nearest: bool,
    return_observation_mask: bool = False,
):
    """Place model-resolution world points into their observed GT-image ROI."""
    from mv_recon.pc_infer_utils import pred_mask_from_depth, resize_map_to_hw

    data_h, data_w = int(data_size[0]), int(data_size[1])
    model_h, model_w = int(depth_model.shape[-2]), int(depth_model.shape[-1])
    rows, cols = horizon_model_roi_on_gt(
        (model_h, model_w),
        (data_h, data_w),
        img_size=img_size,
        crop=crop,
    )
    roi_h, roi_w = rows.stop - rows.start, cols.stop - cols.start

    world_nchw = world_model.permute(0, 3, 1, 2).contiguous()
    world_roi = resize_map_to_hw(world_nchw, (roi_h, roi_w), nearest=nearest)
    world_roi = world_roi.permute(0, 2, 3, 1).contiguous().numpy()
    depth_roi = resize_map_to_hw(depth_model, (roi_h, roi_w), nearest=nearest)
    mask_roi = pred_mask_from_depth(depth_roi)

    num_frames = int(world_model.shape[0])
    world_gt = np.zeros((num_frames, data_h, data_w, 3), dtype=world_roi.dtype)
    pred_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    observation_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    world_gt[:, rows, cols, :] = world_roi
    pred_mask[:, rows, cols] = mask_roi
    observation_mask[:, rows, cols] = True
    if return_observation_mask:
        return world_gt, pred_mask, observation_mask
    return world_gt, pred_mask


def scale_intrinsics_to_hw(
    intrinsic: torch.Tensor,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Scale pixel-space K from one image canvas to another."""
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError(f"invalid intrinsic resize: {source_hw} -> {target_hw}")
    scaled = intrinsic.detach().float().cpu().clone()
    scaled[..., 0, :] *= target_w / float(source_w)
    scaled[..., 1, :] *= target_h / float(source_h)
    return scaled


def postprocess_horizon_depth_to_gt(
    depth_model: torch.Tensor,
    intrinsic_model: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    data_size: Tuple[int, int],
    *,
    img_size: int,
    crop: bool,
    nearest: bool,
    return_observation_mask: bool = False,
    alignment_depth_max: float | None = None,
    return_alignment_mask: bool = False,
):
    """Resize depth into the observed ROI, scale predicted K, then unproject.

    This preserves pinhole geometry at the evaluation grid. Rows/columns removed
    by Horizon's input center crop remain unobserved and never enter metrics.
    """
    from mv_recon.pc_infer_utils import pred_mask_from_depth, resize_map_to_hw

    depth_model = depth_model.detach().float().cpu()
    intrinsic_model = intrinsic_model.detach().float().cpu()
    extrinsic_w2c = extrinsic_w2c.detach().float().cpu()
    data_h, data_w = int(data_size[0]), int(data_size[1])
    model_h, model_w = int(depth_model.shape[-2]), int(depth_model.shape[-1])
    rows, cols = horizon_model_roi_on_gt(
        (model_h, model_w), (data_h, data_w), img_size=img_size, crop=crop
    )
    roi_hw = (rows.stop - rows.start, cols.stop - cols.start)
    depth_roi = resize_map_to_hw(depth_model, roi_hw, nearest=nearest)
    intrinsic_roi = scale_intrinsics_to_hw(
        intrinsic_model, (model_h, model_w), roi_hw
    )

    world_frames = []
    for depth_i, intrinsic_i, w2c_i in zip(
        depth_roi, intrinsic_roi, extrinsic_w2c
    ):
        camera_points = unproject_depth_to_points(
            depth_i[None], intrinsic_i[None]
        )[0]
        rotation = w2c_i[:3, :3]
        translation = w2c_i[:3, 3]
        flat = camera_points.reshape(-1, 3)
        world = (rotation.t() @ (flat.t() - translation[:, None])).t()
        world_frames.append(world.reshape(camera_points.shape))
    world_roi = torch.stack(world_frames, dim=0).numpy()
    mask_roi = pred_mask_from_depth(depth_roi)
    alignment_mask_roi = mask_roi.copy()
    if alignment_depth_max is not None:
        alignment_mask_roi &= np.asarray(depth_roi) <= float(alignment_depth_max)

    num_frames = int(depth_model.shape[0])
    world_gt = np.zeros((num_frames, data_h, data_w, 3), dtype=world_roi.dtype)
    pred_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    observation_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    alignment_pred_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    world_gt[:, rows, cols, :] = world_roi
    pred_mask[:, rows, cols] = mask_roi
    observation_mask[:, rows, cols] = True
    alignment_pred_mask[:, rows, cols] = alignment_mask_roi
    if return_observation_mask and return_alignment_mask:
        return world_gt, pred_mask, observation_mask, alignment_pred_mask
    if return_observation_mask:
        return world_gt, pred_mask, observation_mask
    if return_alignment_mask:
        return world_gt, pred_mask, alignment_pred_mask
    return world_gt, pred_mask


def postprocess_horizon_depth_map_to_gt(
    depth_model: torch.Tensor,
    data_size: Tuple[int, int],
    *,
    img_size: int,
    crop: bool,
    nearest: bool,
) -> np.ndarray:
    """Place Horizon camera-Z predictions on the exact observed GT grid."""
    from mv_recon.pc_infer_utils import resize_map_to_hw

    depth_model = depth_model.detach().float().cpu()
    data_h, data_w = int(data_size[0]), int(data_size[1])
    model_h, model_w = int(depth_model.shape[-2]), int(depth_model.shape[-1])
    rows, cols = horizon_model_roi_on_gt(
        (model_h, model_w), (data_h, data_w), img_size=img_size, crop=crop
    )
    depth_roi = resize_map_to_hw(
        depth_model,
        (rows.stop - rows.start, cols.stop - cols.start),
        nearest=nearest,
    )
    output = np.zeros((len(depth_model), data_h, data_w), dtype=np.float32)
    output[:, rows, cols] = np.asarray(depth_roi, dtype=np.float32)
    return output


def _load_and_preprocess(
    filelist: List[str],
    img_size: int,
    patch_size: int,
    crop: bool,
    device: str,
):
    """Official HorizonStream loader: long-edge resize + optional center crop.

    Returns (1, S, C, H, W) in [0, 1].
    """
    if len(filelist) > 64:
        return LazyHorizonImages(filelist, img_size, patch_size, crop, device)
    views = load_images_for_eval(
        filelist,
        size=img_size,
        verbose=False,
        crop=crop,
        patch_size=patch_size,
    )
    imgs = torch.cat([view["img"] for view in views], dim=0)  # (S,C,H,W) in [-1,1]
    images = (imgs.unsqueeze(0) + 1.0) / 2.0
    return images.to(device)


def _record_runtime(model, images, filelist):
    use_amp, dtype = model._autocast_settings(torch.device(images.device))
    compute_dtype = str(dtype).removeprefix("torch.") if use_amp else "fp32"
    record_model_runtime(
        model,
        input_hw=images.shape[-2:],
        input_storage_dtype="float32",
        forward_compute_dtype=compute_dtype,
        preprocess="official_horizon_long_edge_518_center_crop_patch14",
        online_state=(
            f"streaming_window-{model.window_size}"
            f"+sliding-{model.sliding_size}"
            f"+abs_pose-{model.abs_pose_source}"
        ),
        forward_frames=len(filelist),
    )


def _w2c_3x4_to_c2w_3x4(w2c_3x4: torch.Tensor) -> torch.Tensor:
    # BF16/FP16 are valid forward dtypes, but CPU linalg.inv requires a
    # sufficiently precise floating-point dtype.
    w2c_3x4 = torch.as_tensor(w2c_3x4).float()
    num_frames = w2c_3x4.shape[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=w2c_3x4.dtype).expand(
        num_frames, 1, 4
    )
    w2c_4x4 = torch.cat([w2c_3x4, bottom], dim=1)
    c2w_4x4 = torch.inverse(w2c_4x4)
    return c2w_4x4[:, :3, :]


def _w2c_3x4_to_w2c_4x4(w2c_3x4: torch.Tensor) -> torch.Tensor:
    num_frames = w2c_3x4.shape[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=w2c_3x4.dtype).expand(
        num_frames, 1, 4
    )
    return torch.cat([w2c_3x4, bottom], dim=1)


def infer_cameras_c2w(filelist: List[str], model: HorizonStreamEval, hydra_cfg: DictConfig):
    images = _load_and_preprocess(
        filelist, model.img_size, model.patch_size, model.crop, hydra_cfg.device
    )
    _record_runtime(model, images, filelist)
    with torch.amp.autocast(device_type=str(hydra_cfg.device).split(":")[0], enabled=False):
        with time_forward(model, hydra_cfg, num_frames=len(filelist)):
            predictions = model(
                images.float(),
                pose_only=bool(hydra_cfg.get("pose_only_skip_dense_heads", True)),
            )
    extrinsic_w2c = predictions["extrinsic_w2c"].detach().cpu()
    intrinsic = predictions["intrinsic"].detach().cpu()
    return _w2c_3x4_to_c2w_3x4(extrinsic_w2c), intrinsic


def _retrieve_horizon_salad_candidates_lazy(images, cfg, device):
    """Official SALAD retrieval without retaining the full RGB sequence."""
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError("Horizon loop retrieval requires faiss") from exc

    from horizonstream.core.infer import _to_uint8_rgb
    from horizonstream.loop.runtime import (
        LoopCandidate,
        SaladVPRModel,
        _batch_to_imagenet_tensor,
        _load_salad_state_dict,
        _normalize_rows,
        _resize_rgb_batch,
        non_max_suppress_loop_candidates,
    )

    if not os.path.isfile(cfg.salad_ckpt_path):
        raise FileNotFoundError(f"SALAD checkpoint not found: {cfg.salad_ckpt_path}")
    if not os.path.isfile(cfg.salad_dino_weights_path):
        raise FileNotFoundError(
            f"SALAD DINO weights not found: {cfg.salad_dino_weights_path}"
        )

    torch_device = torch.device(device)
    hub_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if not os.path.isdir(hub_repo):
        raise FileNotFoundError(
            "Offline Horizon loop requires the cached DINOv2 torch-hub source: "
            f"{hub_repo}"
        )
    original_hub_load = torch.hub.load

    def local_dinov2_load(repo_or_dir, model_name, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov2":
            return original_hub_load(repo_or_dir, model_name, *args, **kwargs)
        kwargs.pop("source", None)
        return original_hub_load(
            hub_repo, model_name, *args, source="local", **kwargs
        )

    torch.hub.load = local_dinov2_load
    try:
        retrieval_model = SaladVPRModel(
            backbone_name=str(cfg.salad_backbone),
            dino_weights_path=str(cfg.salad_dino_weights_path),
        )
    finally:
        torch.hub.load = original_hub_load
    missing = _load_salad_state_dict(retrieval_model, str(cfg.salad_ckpt_path))
    if any(key.startswith("backbone.") for key in missing) and not os.path.isfile(
        cfg.salad_dino_weights_path
    ):
        raise RuntimeError("SALAD checkpoint is missing DINO backbone weights")
    retrieval_model = retrieval_model.to(torch_device).eval()

    descriptors = []
    frames = int(images.shape[1])
    batch_size = max(1, int(cfg.salad_batch_size))
    image_size = tuple(int(value) for value in cfg.salad_image_size)
    with torch.no_grad():
        for start in range(0, frames, batch_size):
            stop = min(start + batch_size, frames)
            rgb = _to_uint8_rgb(images[:, start:stop][0].permute(0, 2, 3, 1))
            resized = _resize_rgb_batch(rgb, image_size)
            batch = _batch_to_imagenet_tensor(resized, torch_device)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                descriptor = retrieval_model(batch)
            descriptors.append(descriptor.float().cpu().numpy().astype(np.float32))
            del batch, descriptor

    descriptors = _normalize_rows(np.concatenate(descriptors, axis=0))
    index = faiss.IndexFlatIP(int(descriptors.shape[1]))
    index.add(descriptors)
    top_k = max(1, int(cfg.retrieval_top_k))
    similarities, indices = index.search(descriptors, top_k + 1)

    candidates_by_key = {}
    min_gap = max(1, int(cfg.min_frame_separation))
    for src_pos in range(frames):
        for rank in range(1, top_k + 1):
            dst_pos = int(indices[src_pos, rank])
            if dst_pos < 0 or dst_pos == src_pos:
                continue
            score = float(similarities[src_pos, rank])
            if score <= float(cfg.salad_score_thresh):
                continue
            if abs(src_pos - dst_pos) <= min_gap:
                continue
            high, low = max(src_pos, dst_pos), min(src_pos, dst_pos)
            previous = candidates_by_key.get((high, low))
            if previous is None or score > previous.score:
                candidates_by_key[(high, low)] = LoopCandidate(
                    src_pos=high,
                    dst_pos=low,
                    src_frame=high,
                    dst_frame=low,
                    score=score,
                    method="salad_online",
                )

    candidates = non_max_suppress_loop_candidates(
        list(candidates_by_key.values()),
        nms_threshold=max(0, int(cfg.nms_radius)),
        limit=max(1, int(cfg.max_candidates)),
    )
    del retrieval_model, descriptors, index
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates


def run_horizon_loop_from_c2w(
    filelist: List[str],
    noloop_c2w,
    model: HorizonStreamEval,
    hydra_cfg: DictConfig,
    loop_cfg_dict,
    artifact_dir: str,
):
    """Run official SALAD + window re-inference + SE(3) PGO on cached poses."""
    from dataclasses import asdict

    from horizonstream.loop.online_loop_reinfer import (
        LoopReinferRefiner,
        OnlineLoopReinferConfig,
        _build_direct_loop_edges,
        _build_loop_cfg,
        _filter_loop_edges_by_score,
    )
    from horizonstream.loop.runtime import (
        build_keyframe_odometry_edges,
        optimize_keyframe_pose_graph,
        save_loop_edges_json,
    )

    os.makedirs(artifact_dir, exist_ok=True)
    cfg = OnlineLoopReinferConfig.from_dict(dict(loop_cfg_dict))
    if set(method.lower() for method in cfg.enabled_methods) != {"salad"}:
        raise ValueError("The released HorizonStream loop protocol supports SALAD only")

    images = _load_and_preprocess(
        filelist, model.img_size, model.patch_size, model.crop, hydra_cfg.device
    )
    candidates = _retrieve_horizon_salad_candidates_lazy(
        images, cfg, hydra_cfg.device
    )
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    candidates = candidates[: int(cfg.max_candidates)]

    if int(cfg.max_reinfer_candidates) <= 0 and bool(cfg.dbow_direct_loop_edges):
        loop_edges = _build_direct_loop_edges(candidates, cfg)
    else:
        refiner = LoopReinferRefiner(
            model=model.model,
            images=images,
            device=str(hydra_cfg.device),
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
            cfg=cfg,
        )
        loop_edges = []
        for candidate in candidates[: int(cfg.max_reinfer_candidates)]:
            edge = refiner.refine(candidate)
            if edge is not None:
                loop_edges.append(edge)
    loop_edges = _filter_loop_edges_by_score(
        loop_edges, cfg.loop_edge_score_threshold
    )

    base = torch.as_tensor(noloop_c2w).double().cpu().numpy()
    if base.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(
            np.asarray([0.0, 0.0, 0.0, 1.0]), (len(base), 1, 4)
        )
        base = np.concatenate([base, bottom], axis=1)
    odom_edges = build_keyframe_odometry_edges(base)
    loop_c2w = optimize_keyframe_pose_graph(
        base, odom_edges, loop_edges, _build_loop_cfg(cfg)
    )

    with open(os.path.join(artifact_dir, "candidates.json"), "w") as handle:
        json.dump([asdict(candidate) for candidate in candidates], handle, indent=2)
    save_loop_edges_json(
        os.path.join(artifact_dir, "loop_edges.json"), loop_edges
    )
    metadata = {
        "num_candidates": len(candidates),
        "num_refined_candidates": min(
            len(candidates), int(cfg.max_reinfer_candidates)
        ),
        "num_loop_edges": len(loop_edges),
        "loop_config": asdict(cfg),
    }
    with open(os.path.join(artifact_dir, "loop_summary.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    return torch.from_numpy(np.asarray(loop_c2w)).float()[:, :3], metadata


def infer_cameras_w2c(filelist: List[str], model: HorizonStreamEval, hydra_cfg: DictConfig):
    images = _load_and_preprocess(
        filelist, model.img_size, model.patch_size, model.crop, hydra_cfg.device
    )
    _record_runtime(model, images, filelist)
    with torch.amp.autocast(device_type=str(hydra_cfg.device).split(":")[0], enabled=False):
        predictions = model(
            images.float(),
            pose_only=bool(hydra_cfg.get("pose_only_skip_dense_heads", True)),
        )
    return (
        _w2c_3x4_to_w2c_4x4(predictions["extrinsic_w2c"]).double(),
        predictions["intrinsic"].double(),
    )


def infer_mv_pointclouds(
    filelist: List[str],
    model: HorizonStreamEval,
    hydra_cfg: DictConfig,
    data_size: Tuple[int, int],
):
    """Depth-backproject world points at ``data_size`` for lingbot-aligned PC metrics.

    No sky / conf filtering — same dense grid as lingbot_map.
    When ``nearest_depth_to_gt``: NEAREST upsample depth + pred_mask = depth > 1e-4.
    """
    from mv_recon.pc_infer_utils import nearest_depth_to_gt_enabled

    images = _load_and_preprocess(
        filelist, model.img_size, model.patch_size, model.crop, hydra_cfg.device
    )
    _record_runtime(model, images, filelist)
    configured = getattr(hydra_cfg, "mv_recon_output_indices", None)
    dense_indices = None if configured is None else [int(i) for i in configured]
    predictions = model(images, dense_output_indices=dense_indices)

    # Model inference runs under bf16 autocast, but geometry must use one
    # stable dtype. Keep postprocessing off the GPU and perform all
    # unprojection / frame transforms in float32.
    extrinsic_w2c = predictions["extrinsic_w2c"].detach().float().cpu()  # (S, 3, 4)
    intrinsic = predictions["intrinsic"].detach().float().cpu()  # (S, 3, 3)
    depth_map = predictions["depth"].detach().float().cpu()  # (S, H, W)
    full_extrinsic_w2c = extrinsic_w2c

    if configured is not None:
        indices = torch.as_tensor(list(configured), dtype=torch.long)
        intrinsic = intrinsic.index_select(0, indices)
        extrinsic_w2c = extrinsic_w2c.index_select(0, indices)

    nearest = nearest_depth_to_gt_enabled(hydra_cfg)
    alignment_depth_max = getattr(hydra_cfg, "pc_alignment_depth_max", None)
    world_points, pred_mask, observation_mask, alignment_pred_mask = postprocess_horizon_depth_to_gt(
        depth_map,
        intrinsic,
        extrinsic_w2c,
        data_size,
        img_size=model.img_size,
        crop=model.crop,
        nearest=nearest,
        return_observation_mask=True,
        alignment_depth_max=alignment_depth_max,
        return_alignment_mask=True,
    )
    depth_gt = postprocess_horizon_depth_map_to_gt(
        depth_map,
        data_size,
        img_size=model.img_size,
        crop=model.crop,
        nearest=nearest,
    )

    num_frames = full_extrinsic_w2c.shape[0]
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=extrinsic_w2c.dtype).expand(
        num_frames, 1, 4
    )
    w2c_4x4 = torch.cat([full_extrinsic_w2c, bottom], dim=1)
    c2w_4x4 = torch.inverse(w2c_4x4).detach().cpu().numpy().astype(np.float64)
    return (
        world_points,
        c2w_4x4,
        pred_mask,
        observation_mask,
        alignment_pred_mask if alignment_depth_max is not None else None,
        {"pred_depth": depth_gt},
    )
