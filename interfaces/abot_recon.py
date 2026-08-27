"""ABot-Recon adapters for evaluation (pose + multi-view point clouds).

Preprocess matches training FOV (width-lock 504 + mean-pad / center-crop to 280).
For mv_recon: strip pad rows from local/world points, then resize to GT grid
with the explicitly configured point-map interpolation mode.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.abot_recon import ABotReconEval
from relpose.forward_timing import materialize_for_forward_timing, time_forward
from models.vggt.utils.geometry import closed_form_inverse_se3
from mv_recon.fov_preprocess import (
    DEFAULT_PAD_RGB,
    FovPadInfo,
    load_filelist_fov,
    resize_width_crop_or_pad_mean,
    strip_vertical_pad_hwc,
)
from mv_recon.runtime_manifest import record_model_runtime


def _runtime_backend_metadata(model: ABotReconEval) -> dict:
    return {
        "rope2d_backend": os.environ.get("ABOT_RECON_ROPE2D_BACKEND", "auto").lower(),
        "attention_backend": (
            "paged-flashinfer" if bool(getattr(model, "use_paged_kv", False)) else "sdpa"
        ),
    }


def _load_filelist_fov_tensor(
    filelist: List[str],
    model: ABotReconEval,
    device: str,
) -> Tuple[torch.Tensor, FovPadInfo]:
    """``(1, N, 3, H, W)`` in [0, 1] via training FOV preprocess."""
    # ``filelist`` is already aligned with dataset GT arrays / seq-id-map.
    # Re-sorting only RGB here can silently break image-to-GT correspondence.
    paths = list(filelist)
    h, w = int(model.height), int(model.width)
    pad_rgb = getattr(model, "fov_pad_rgb", DEFAULT_PAD_RGB)
    return load_filelist_fov(
        paths,
        target_h=h,
        target_w=w,
        pad_rgb=pad_rgb,
        device=device,
    )


def _run_stream_inference(
    filelist: List[str],
    model: ABotReconEval,
    hydra_cfg: DictConfig,
    *,
    need_points: bool = False,
    dense_output_keys: Optional[Tuple[str, ...]] = None,
) -> Tuple[dict, FovPadInfo]:
    """Run streaming inference; returns (output_dict, fov_pad)."""
    if dense_output_keys is None:
        # Backward compatibility for mv_recon and unreleased diagnostics.
        dense_output_keys = ("points", "local_points") if need_points else ()
    dense_output_keys = tuple(dense_output_keys)
    supported_dense = {"points", "local_points", "conf"}
    unknown = set(dense_output_keys) - supported_dense
    if unknown:
        raise ValueError(f"Unsupported ABot-Recon dense outputs: {sorted(unknown)}")
    need_dense = bool(dense_output_keys)
    device = str(hydra_cfg.device)
    model.eval()

    if model.local_window_override is not None and hasattr(model.model, "set_local_window_frames"):
        model.model.set_local_window_frames(model.local_window_override)

    paths = list(filelist)
    if model.num_frames_cap is not None:
        paths = paths[: max(int(model.num_frames_cap), 1)]
    if not paths:
        raise FileNotFoundError("Empty filelist for ABot-Recon inference")

    # Streaming evaluation decodes one RGB frame at a time. Point-cloud runs
    # retain dense maps only at requested metric frames while preserving every
    # pose and every recurrent state update.
    if str(model.infer_mode).lower() == "stream" and hasattr(model, "inference_stream_iter"):
        pad_rgb = getattr(model, "fov_pad_rgb", DEFAULT_PAD_RGB)
        with Image.open(paths[0]) as image:
            first = resize_width_crop_or_pad_mean(
                image.convert("RGB"),
                target_h=int(model.height),
                target_w=int(model.width),
                pad_rgb=pad_rgb,
            )
        amp = model.amp_dtype
        amp_torch = (
            torch.bfloat16 if amp == "bf16" else torch.float16 if amp == "fp16" else torch.float32
        )
        record_model_runtime(
            model,
            input_hw=first.image.shape[-2:],
            input_storage_dtype=(
                amp if device.startswith("cuda") and amp in ("bf16", "fp16") else "float32"
            ),
            forward_compute_dtype=(amp if amp in ("bf16", "fp16") else "fp32"),
            preprocess="training_fov_width504_crop_or_meanpad_height280",
            online_state=(
                f"causal_stream+local_window-{model.local_window_override}"
                f"+memory-{getattr(model.model, 'memory_mode', 'checkpoint')}"
                f"+paged-kv-{bool(getattr(model, 'use_paged_kv', False))}"
            ),
            forward_frames=len(paths),
            extra=_runtime_backend_metadata(model),
        )

        descriptor_worker = None
        if bool(hydra_cfg.get("abot_recon_loop_enabled", False)):
            from abot_recon.loop_closure import start_descriptor_worker

            descriptor_worker = start_descriptor_worker(
                salad_checkpoint=hydra_cfg.abot_recon_loop_salad_ckpt,
                dino_checkpoint=hydra_cfg.abot_recon_loop_dino_weights,
                device=device,
            )

        def frame_iterator():
            for index, path in enumerate(paths):
                if index == 0:
                    frame = first
                else:
                    with Image.open(path) as image:
                        frame = resize_width_crop_or_pad_mean(
                            image.convert("RGB"),
                            target_h=int(model.height),
                            target_w=int(model.width),
                            pad_rgb=pad_rgb,
                        )
                if frame.pad != first.pad:
                    raise ValueError(
                        f"Inconsistent FOV geometry at frame {index}: {frame.pad} != {first.pad}"
                    )
                if descriptor_worker is not None:
                    descriptor_worker.submit(frame.image.unsqueeze(0).unsqueeze(0))
                tensor = frame.image.unsqueeze(0).unsqueeze(0)
                if device.startswith("cuda") and amp in ("bf16", "fp16"):
                    tensor = tensor.to(device=device, dtype=amp_torch, non_blocking=True)
                else:
                    tensor = tensor.to(device=device, non_blocking=True)
                yield tensor

        causal = bool(getattr(model.model, "causal_global_attn", True))
        dev_type = "cuda" if device.startswith("cuda") else "cpu"
        frames = materialize_for_forward_timing(
            frame_iterator(),
            hydra_cfg,
            num_frames=len(paths),
            label="ABot-Recon",
        )
        with torch.amp.autocast(
            device_type=dev_type,
            enabled=(device.startswith("cuda") and amp in ("bf16", "fp16")),
            dtype=amp_torch,
        ):
            with time_forward(model, hydra_cfg, num_frames=len(paths)):
                metric_indices = _metric_output_indices(hydra_cfg, len(paths))
                supports_sparse = bool(getattr(model, "supports_sparse_dense_output", False))
                dense_indices = (
                    None
                    if not need_dense or metric_indices is None or not supports_sparse
                    else metric_indices.tolist()
                )
                output_keys = ("camera_poses", *dense_output_keys)
                out = model.inference_stream_iter(
                    frames,
                    num_frames=len(paths),
                    causal_global_attn=causal,
                    output_keys=output_keys,
                    **({"dense_output_indices": dense_indices} if supports_sparse else {}),
                )
        if descriptor_worker is not None:
            descriptors, descriptor_stats = descriptor_worker.finish()
            out["_loop_descriptors"] = descriptors
            out["_loop_descriptor_stats"] = descriptor_stats
        return out, first.pad

    imgs, pad = _load_filelist_fov_tensor(paths, model, device)

    amp = model.amp_dtype
    amp_torch = (
        torch.bfloat16 if amp == "bf16" else torch.float16 if amp == "fp16" else torch.float32
    )
    record_model_runtime(
        model,
        input_hw=imgs.shape[-2:],
        input_storage_dtype=(
            amp if device.startswith("cuda") and amp in ("bf16", "fp16") else "float32"
        ),
        forward_compute_dtype=(amp if amp in ("bf16", "fp16") else "fp32"),
        preprocess="training_fov_width504_crop_or_meanpad_height280",
        online_state=(
            f"{model.infer_mode}+local_window-{model.local_window_override}"
            f"+paged-kv-{bool(getattr(model, 'use_paged_kv', False))}"
        ),
        forward_frames=len(paths),
        extra=_runtime_backend_metadata(model),
    )
    imgs_in = imgs
    if device.startswith("cuda") and amp in ("bf16", "fp16"):
        imgs_in = imgs.to(dtype=amp_torch)

    causal = bool(getattr(model.model, "causal_global_attn", True))
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        with torch.amp.autocast(
            device_type=dev_type,
            enabled=(device.startswith("cuda") and amp in ("bf16", "fp16")),
            dtype=amp_torch,
        ):
            out = model.inference_stream(imgs_in, causal_global_attn=causal)

    if need_dense and ({"points", "local_points"} & set(dense_output_keys)):
        if out.get("points") is None and out.get("local_points") is None:
            raise RuntimeError(
                "ABot-Recon inference returned neither points nor local_points "
                "(is this a camera-only forward?)"
            )
    return out, pad


def _poses_c2w_from_out(out: dict) -> torch.Tensor:
    poses = out.get("camera_poses")
    if poses is None:
        raise RuntimeError("ABot-Recon inference did not return camera_poses")
    # (1, N, 4, 4) or (N, 4, 4)
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    return poses.detach().float().cpu()


def extract_abot_recon_loop_descriptors(
    filelist: List[str],
    model: ABotReconEval,
    hydra_cfg: DictConfig,
):
    """Extract SALAD descriptors without rerunning the streaming pose model."""
    from abot_recon.loop_closure import start_descriptor_worker

    paths = list(filelist)
    if not paths:
        raise FileNotFoundError("Empty filelist for ABot-Recon loop descriptors")
    worker = start_descriptor_worker(
        salad_checkpoint=hydra_cfg.abot_recon_loop_salad_ckpt,
        dino_checkpoint=hydra_cfg.abot_recon_loop_dino_weights,
        device=str(hydra_cfg.device),
    )
    expected_pad = None
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                frame = resize_width_crop_or_pad_mean(
                    image.convert("RGB"),
                    target_h=int(model.height),
                    target_w=int(model.width),
                    pad_rgb=getattr(model, "fov_pad_rgb", DEFAULT_PAD_RGB),
                )
            if expected_pad is None:
                expected_pad = frame.pad
            elif frame.pad != expected_pad:
                raise ValueError(
                    f"Inconsistent FOV geometry at frame {index}: "
                    f"{frame.pad} != {expected_pad}"
                )
            worker.submit(frame.image.unsqueeze(0).unsqueeze(0))
        return worker.finish()
    except Exception:
        worker.cancel()
        raise


def run_abot_recon_loop_from_c2w(
    filelist: List[str],
    base_poses: torch.Tensor,
    model: ABotReconEval,
    hydra_cfg: DictConfig,
    *,
    descriptors=None,
    descriptor_stats=None,
    return_metadata: bool = False,
):
    """Apply the optional SALAD+PGO backend to a camera-only trajectory."""
    from abot_recon.loop_closure import LoopClosureConfig, apply_loop_closure

    salad_ckpt = Path(str(hydra_cfg.abot_recon_loop_salad_ckpt)).expanduser()
    dino_weights = Path(str(hydra_cfg.abot_recon_loop_dino_weights)).expanduser()
    for path in (salad_ckpt, dino_weights):
        if not path.exists():
            raise FileNotFoundError(f"ABot-Recon loop dependency not found: {path}")
    loop_cfg = LoopClosureConfig(
        enabled=True,
        salad_checkpoint=str(salad_ckpt),
        dino_checkpoint=str(dino_weights),
    )
    digest = hashlib.sha1("\n".join(filelist).encode()).hexdigest()[:12]
    save_root = (
        Path(str(hydra_cfg.get("output_dir", "outputs"))) / "abot_recon_loop_artifacts" / digest
    )
    poses, metadata = apply_loop_closure(
        model.runtime_model,
        list(filelist),
        base_poses,
        device=torch.device(str(hydra_cfg.device)),
        loop_cfg=loop_cfg,
        save_dir=save_root,
        descriptors=descriptors,
        descriptor_stats=descriptor_stats,
    )
    poses = poses.detach().float().cpu()
    return (poses, metadata) if return_metadata else poses


def infer_cameras_c2w(
    filelist: List[str],
    model: ABotReconEval,
    hydra_cfg: DictConfig,
) -> Tuple[torch.Tensor, None]:
    """``(N, 4, 4)`` c2w poses; intrinsics unused (``None``)."""
    out, _pad = _run_stream_inference(filelist, model, hydra_cfg, need_points=False)
    poses = _poses_c2w_from_out(out)
    if bool(hydra_cfg.get("abot_recon_loop_enabled", False)):
        poses = run_abot_recon_loop_from_c2w(
            filelist,
            poses,
            model,
            hydra_cfg,
            descriptors=out.get("_loop_descriptors"),
            descriptor_stats=out.get("_loop_descriptor_stats"),
        )
    return poses, None


def infer_custom_reconstruction(
    filelist: List[str],
    model: ABotReconEval,
    hydra_cfg: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, FovPadInfo]:
    """Return c2w poses, local point maps, confidence probabilities, and FOV pad."""
    out, pad = _run_stream_inference(
        filelist,
        model,
        hydra_cfg,
        dense_output_keys=("local_points", "conf"),
    )
    poses = _poses_c2w_from_out(out)
    local = _local_points_model_res(out)
    if local is None:
        raise RuntimeError("ABot-Recon reconstruction requires local_points")
    conf = out.get("conf")
    if conf is None:
        raise RuntimeError(
            "Checkpoint/model did not return confidence. Enable the confidence "
            "decoder and use a checkpoint containing conf_decoder/conf_head."
        )
    if conf.dim() == 5 and conf.shape[0] == 1:
        conf = conf[0]
    if conf.dim() == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    conf = torch.sigmoid(conf.detach().float().cpu())
    return poses, local, conf, pad


def infer_cameras_w2c(
    filelist: List[str],
    model: ABotReconEval,
    hydra_cfg: DictConfig,
) -> Tuple[torch.Tensor, None]:
    """``(N, 4, 4)`` w2c poses for relpose-angular."""
    c2w = _poses_c2w_from_out(
        _run_stream_inference(filelist, model, hydra_cfg, need_points=False)[0]
    )
    w2c = closed_form_inverse_se3(c2w.numpy())
    return torch.from_numpy(w2c).double(), None


def _metric_output_indices(hydra_cfg: DictConfig, num_frames: int) -> Optional[torch.Tensor]:
    configured = getattr(hydra_cfg, "mv_recon_output_indices", None)
    if configured is None:
        return None
    values = torch.as_tensor(list(configured), dtype=torch.long)
    if values.numel() == 0:
        raise ValueError("mv_recon_output_indices must not be empty")
    if int(values.min()) < 0 or int(values.max()) >= int(num_frames):
        raise ValueError(f"mv_recon_output_indices outside [0,{num_frames}): {values.tolist()}")
    if values.numel() > 1 and not bool(torch.all(values[1:] > values[:-1])):
        raise ValueError("mv_recon_output_indices must be strictly increasing")
    return values


def _select_frames(value: torch.Tensor, indices: Optional[torch.Tensor]) -> torch.Tensor:
    if indices is None:
        return value
    return value.index_select(0, indices.to(device=value.device))


def _local_points_model_res(
    out: dict, frame_indices: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Return ``(S, H, W, 3)`` camera-frame points at model resolution, or None."""
    local = out.get("local_points")
    if local is None:
        return None
    if local.dim() == 5 and local.shape[0] == 1:
        local = local[0]
    local = _select_frames(local, frame_indices)
    return local.detach().float().cpu()


def _world_points_model_res(
    out: dict,
    source: str = "points",
    frame_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``(S, H, W, 3)`` world points at model resolution.

    ``source='points'`` uses the model's direct world-point head (legacy
    evaluation). ``source='local_pose'`` reconstructs world points from local
    camera-frame points and the rel-pose head, which is directly comparable to
    depth+pose reconstruction used by HorizonStream and LingBot-MAP.
    """
    source = str(source)
    if source not in {"points", "local_pose"}:
        raise ValueError(f"Unknown pi3_pc_world_source={source!r}; use points|local_pose")
    pts = out.get("points")
    if source == "points" and pts is not None:
        if pts.dim() == 5 and pts.shape[0] == 1:
            pts = pts[0]
        pts = _select_frames(pts, frame_indices)
        return pts.detach().float().cpu()

    local = out.get("local_points")
    if local is None:
        raise RuntimeError(
            f"pi3_pc_world_source={source!r} requires local_points, but model returned none"
        )
    poses = out["camera_poses"]
    if local.dim() == 5 and local.shape[0] == 1:
        local = local[0]
    if poses.dim() == 4 and poses.shape[0] == 1:
        poses = poses[0]
    local = _select_frames(local, frame_indices)
    poses = _select_frames(poses, frame_indices)
    local = local.detach().float().cpu()
    poses = poses.detach().float().cpu()
    # world = R @ local + t
    s, h, w, _ = local.shape
    flat = local.reshape(s, -1, 3)  # S, HW, 3
    ones = torch.ones(s, flat.shape[1], 1, dtype=flat.dtype)
    homo = torch.cat([flat, ones], dim=-1)  # S, HW, 4
    world = torch.einsum("sij,snj->sni", poses, homo)[..., :3]
    return world.reshape(s, h, w, 3)


def postprocess_fov_points_to_gt(
    world: torch.Tensor,
    pad: FovPadInfo,
    data_size: Tuple[int, int],
    *,
    local: Optional[torch.Tensor] = None,
    nearest: bool = True,
    return_observation_mask: bool = False,
    alignment_depth_max: float | None = None,
    return_alignment_mask: bool = False,
):
    """Map model FOV back to its corresponding ROI on the full GT grid.

    ``world`` / ``local`` are model-res ``(S, H, W, 3)``. Mask uses camera-frame
    ``local[..., 2] > 1e-4`` when ``local`` is given.
    """
    from mv_recon.pc_infer_utils import pred_mask_from_depth, resize_map_to_hw

    world = strip_vertical_pad_hwc(world, pad)
    if local is not None:
        local = strip_vertical_pad_hwc(local, pad)
        depth = local[..., 2].contiguous()
    else:
        depth = world[..., 2].contiguous()

    data_h, data_w = int(data_size[0]), int(data_size[1])
    gt_rows = pad.gt_content_row_slice(data_h)
    roi_h = int(gt_rows.stop - gt_rows.start)
    target_hw = (roi_h, data_w)
    world_nchw = world.permute(0, 3, 1, 2).contiguous()
    world_nchw = resize_map_to_hw(world_nchw, target_hw, nearest=nearest)
    roi_pts = world_nchw.permute(0, 2, 3, 1).contiguous().numpy()
    depth_roi = resize_map_to_hw(depth, target_hw, nearest=nearest)
    roi_mask = pred_mask_from_depth(depth_roi)
    alignment_roi_mask = roi_mask.copy()
    if alignment_depth_max is not None:
        alignment_roi_mask &= depth_roi.detach().cpu().numpy() <= float(alignment_depth_max)

    num_frames = int(world.shape[0])
    world_pts = np.zeros((num_frames, data_h, data_w, 3), dtype=roi_pts.dtype)
    pred_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    observation_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    alignment_pred_mask = np.zeros((num_frames, data_h, data_w), dtype=bool)
    world_pts[:, gt_rows, :, :] = roi_pts
    pred_mask[:, gt_rows, :] = roi_mask
    observation_mask[:, gt_rows, :] = True
    alignment_pred_mask[:, gt_rows, :] = alignment_roi_mask
    if return_observation_mask and return_alignment_mask:
        return world_pts, pred_mask, observation_mask, alignment_pred_mask
    if return_observation_mask:
        return world_pts, pred_mask, observation_mask
    if return_alignment_mask:
        return world_pts, pred_mask, alignment_pred_mask
    return world_pts, pred_mask


def postprocess_fov_local_points_to_gt(
    local: torch.Tensor,
    pad: FovPadInfo,
    data_size: Tuple[int, int],
    *,
    nearest: bool = True,
) -> np.ndarray:
    """Map camera-frame XYZ to the observed GT ROI without extra mask copies."""
    from mv_recon.pc_infer_utils import resize_map_to_hw

    local = strip_vertical_pad_hwc(local, pad)
    data_h, data_w = int(data_size[0]), int(data_size[1])
    gt_rows = pad.gt_content_row_slice(data_h)
    target_hw = (int(gt_rows.stop - gt_rows.start), data_w)
    local_nchw = local.permute(0, 3, 1, 2).contiguous()
    local_nchw = resize_map_to_hw(local_nchw, target_hw, nearest=nearest)
    roi = local_nchw.permute(0, 2, 3, 1).contiguous().numpy()
    output = np.zeros((len(local), data_h, data_w, 3), dtype=np.float32)
    output[:, gt_rows, :, :] = roi
    return output


def infer_mv_pointclouds(
    filelist: List[str],
    model: ABotReconEval,
    hydra_cfg: DictConfig,
    data_size: Tuple[int, int],
):
    """Depth/pointmap at GT ``data_size`` for lingbot-aligned PC metrics.

    Pipeline:
      FOV preprocess (W-lock 504 + mean-pad / center-crop to H=280)
      → stream infer
      → strip vertical mean-pad from world / local points
      → NEAREST resize into the corresponding GT ROI
      → keep unobserved rows outside a center crop invalid
      → ``pred_mask = local_z > 1e-4`` (camera depth, not world-Z).
    """
    from mv_recon.pc_infer_utils import pointmap_resize_mode

    out, pad = _run_stream_inference(filelist, model, hydra_cfg, need_points=True)
    c2w = _poses_c2w_from_out(out)
    frame_indices = (
        None
        if getattr(model, "supports_sparse_dense_output", False)
        else _metric_output_indices(hydra_cfg, len(c2w))
    )
    point_source = str(getattr(hydra_cfg, "pi3_pc_world_source", "points"))
    world = _world_points_model_res(
        out, source=point_source, frame_indices=frame_indices
    )  # S,H,W,3 at 280×504 (possibly padded)
    local = _local_points_model_res(out, frame_indices=frame_indices)
    resize_mode = pointmap_resize_mode(hydra_cfg)
    alignment_depth_max = getattr(hydra_cfg, "pc_alignment_depth_max", None)
    world_pts, pred_mask, observation_mask, alignment_pred_mask = postprocess_fov_points_to_gt(
        world,
        pad,
        data_size,
        local=local,
        nearest=(resize_mode == "nearest"),
        return_observation_mask=True,
        alignment_depth_max=alignment_depth_max,
        return_alignment_mask=True,
    )
    local_pts = postprocess_fov_local_points_to_gt(
        local,
        pad,
        data_size,
        nearest=(resize_mode == "nearest"),
    )
    c2w_np = c2w.numpy().astype(np.float64)
    return (
        world_pts,
        c2w_np,
        pred_mask,
        observation_mask,
        alignment_pred_mask if alignment_depth_max is not None else None,
        {
            "pred_depth": local_pts[..., 2].astype(np.float32, copy=False),
            "pred_local_points": local_pts.astype(np.float32, copy=False),
        },
    )
