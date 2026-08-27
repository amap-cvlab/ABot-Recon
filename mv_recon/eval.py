import os
import torch
import numpy as np
import os.path as osp
import hydra
import logging
import time
from typing import Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from mv_recon.lingbot_protocol import (
    METRIC_KEYS,
    colored_aligned_clouds_for_ply,
    evaluate_reconstruction,
    get_dataset_eval_options,
    metrics_to_csv_row,
    normalize_eval_thresholds,
    resolve_dataset_eval_options,
    resolve_lingbot_prepare_width,
    resolve_pc_align_with_scale,
    resolve_pc_gt_load_img_size,
    restrict_masks_to_observed_fov,
    save_xyzrgb_ply,
    subsample_points,
    threshold_metric_keys,
)
from mv_recon.seq_sampling import ensure_seq_id_map, select_seq_id_map
from mv_recon.depth_metrics import camera_z_from_world, evaluate_scale_aligned_depth_maps
from mv_recon.dense_cache import save_metric_frame_cache
from mv_recon.pc_metric_cache import save_pc_metric_cache
from mv_recon.runtime_manifest import (
    clear_model_runtime,
    require_model_runtime,
    write_runtime_manifest,
)
from mv_recon.protocol_validation import validate_formal_pointcloud_protocol
from relpose.output_names import resolve_model_output_slug
from mv_recon.traj_vis import (
    gt_c2w_from_batch,
    save_sequence_traj_bev,
    unpack_infer_mv_result,
)
from utils.messages import set_default_arg, write_csv


def claim_model_output_dir(output_dir: str, model_output_slug: str) -> str:
    """Atomically claim a per-model directory, isolating concurrent launches."""
    os.makedirs(output_dir, exist_ok=True)
    candidate = osp.join(output_dir, model_output_slug)
    try:
        os.mkdir(candidate)
        return candidate
    except FileExistsError:
        candidate = osp.join(output_dir, f"{model_output_slug}-pid{os.getpid()}")
        os.mkdir(candidate)
        return candidate


def resolve_model_alignment_depth_max(
    hydra_cfg: DictConfig,
    model_keyname: str,
    default_value=None,
):
    """Resolve an optional prediction-depth cutoff for alignment only."""
    default = default_value
    per_model = getattr(hydra_cfg, "pc_alignment_depth_max_by_model", None)
    if per_model is None:
        return default
    value = per_model.get(model_keyname, default)
    try:
        if OmegaConf.is_none(value):
            return None
    except Exception:
        pass
    return None if value is None else float(value)


def select_metric_dense_outputs(
    pred_pts: np.ndarray,
    pred_mask: Optional[np.ndarray],
    observation_mask: Optional[np.ndarray],
    alignment_pred_mask: Optional[np.ndarray],
    metric_indices: Optional[Sequence[int]],
    num_inference_frames: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Select dense outputs for sparse metric frames without touching poses.

    Adapters may already honor ``hydra_cfg.mv_recon_output_indices``. This
    helper accepts both that efficient form (M outputs) and the legacy form
    (N outputs, selected here), while rejecting ambiguous lengths.
    """
    if metric_indices is None:
        return pred_pts, pred_mask, observation_mask, alignment_pred_mask
    indices = np.asarray(metric_indices, dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        raise ValueError("metric_indices must not be empty")
    if np.any(indices < 0) or np.any(indices >= int(num_inference_frames)):
        raise ValueError(
            f"metric_indices outside [0,{num_inference_frames}): {indices.tolist()}"
        )
    output_count = int(np.asarray(pred_pts).shape[0])
    if output_count == len(indices):
        return pred_pts, pred_mask, observation_mask, alignment_pred_mask
    if output_count != int(num_inference_frames):
        raise ValueError(
            f"Dense output count {output_count} is neither inference count "
            f"{num_inference_frames} nor metric count {len(indices)}"
        )

    def choose(value):
        if value is None:
            return None
        array = np.asarray(value)
        if array.shape[0] != output_count:
            raise ValueError(
                f"Dense mask count {array.shape[0]} != point count {output_count}"
            )
        return array[indices]

    return np.asarray(pred_pts)[indices], choose(pred_mask), choose(observation_mask), choose(alignment_pred_mask)


def load_metric_rgb(paths: Sequence[str], target_hw: Tuple[int, int]) -> np.ndarray:
    """Load only metric-frame RGB as uint8 HWC for optional colored PLYs."""
    height, width = (int(target_hw[0]), int(target_hw[1]))
    frames = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
    if not frames:
        raise ValueError("No metric RGB paths for PLY visualization")
    return np.stack(frames, axis=0)


def _checkpoint_label(model_info: DictConfig) -> str:
    cfg = model_info.cfg
    for key in ("ckpt", "checkpoint", "pretrained_model_name_or_path"):
        value = cfg.get(key, None)
        if value is not None:
            return str(value)
    return "unreported"


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_models: DictConfig   = hydra_cfg.eval_models    # see configs/evaluation/mv_recon.yaml
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    logger = logging.getLogger("mv_recon-eval")
    alignment_depth_max_default = getattr(
        hydra_cfg, "pc_alignment_depth_max", None
    )

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]

        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        model_output_slug = resolve_model_output_slug(model_keyname, model_info, model)
        model_output_dir = claim_model_output_dir(
            str(hydra_cfg.output_dir), model_output_slug
        )
        logger.info(
            f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from "
            f"{model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}"
        )
        logger.info(f"Isolated model output directory: {model_output_dir}")

        infer_func_cfg = model_info.get(
            "infer_mv_pointclouds",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_mv_pointclouds',
                '_partial_': True,
            })
        )
        infer_mv_pointclouds = hydra.utils.instantiate(infer_func_cfg)

        model_logger = logging.getLogger(f"mv_recon-eval-{model_keyname}")
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            nearest_depth_to_gt = bool(getattr(hydra_cfg, "nearest_depth_to_gt", False))
            # GT size for NEAREST prediction upsampling.
            ds_cfg = OmegaConf.create(
                OmegaConf.to_container(dataset_info.cfg, resolve=True)
            )
            if "load_img_size" in ds_cfg:
                gt_load = resolve_pc_gt_load_img_size(
                    dataset_name,
                    nearest_depth_to_gt,
                    ds_cfg.load_img_size,
                )
                ds_cfg.load_img_size = gt_load
            else:
                gt_load = None
            # Oxford prepares RGB at width 518 before applying area_budget.
            prepare_width = resolve_lingbot_prepare_width(dataset_name)
            alignment_depth_max = resolve_model_alignment_depth_max(
                hydra_cfg,
                model_keyname,
                default_value=alignment_depth_max_default,
            )
            with open_dict(hydra_cfg):
                hydra_cfg.lingbot_prepare_width = prepare_width
                hydra_cfg.pc_alignment_depth_max = alignment_depth_max
            dataset = hydra.utils.instantiate(ds_cfg)
            eval_opts = resolve_dataset_eval_options(
                dataset_name,
                getattr(hydra_cfg, "pc_eval_threshold", None),
            )
            eval_thresholds = normalize_eval_thresholds(
                eval_opts["eval_threshold"],
                getattr(hydra_cfg, "pc_eval_thresholds", None),
            )
            aggregate_metric_keys = METRIC_KEYS
            if len(eval_thresholds) > 1:
                aggregate_metric_keys += threshold_metric_keys(eval_thresholds)
            cfg_align = model_info.get(
                "pc_align_with_scale",
                getattr(hydra_cfg, "pc_align_with_scale", None),
            )
            with_scale = resolve_pc_align_with_scale(model_keyname, cfg_align)
            align_mode = "Sim3" if with_scale else "SE3(keep-model-scale)"

            output_root = osp.join(model_output_dir, dataset_name)
            os.makedirs(output_root, exist_ok=True)
            if hasattr(dataset, "write_depth_association_audit"):
                audit_path = osp.join(output_root, "_depth_association_audit.json")
                dataset.write_depth_association_audit(audit_path)
                model_logger.info(f"Saved depth association audit: {audit_path}")
            all_data_dict = {key: 0.0 for key in aggregate_metric_keys}
            all_data_dict["model"] = model_keyname

            model_logger.info(
                f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View "
                f"Pointcloud Reconstruction on dataset {dataset_name} "
                f"(lingbot protocol: icp={eval_opts['icp_threshold']}, "
                f"voxel={eval_opts['voxel_size']}, "
                f"F1@{eval_thresholds}, primary={eval_opts['eval_threshold']}, "
                f"align={align_mode}, "
                f"nearest_depth_to_gt={nearest_depth_to_gt}, "
                f"gt_load_img_size={gt_load}, "
                f"lingbot_prepare_width={prepare_width})..."
            )
            model_logger.info(
                "Prediction-depth alignment cutoff: "
                f"{alignment_depth_max if alignment_depth_max is not None else 'disabled'}"
            )
            sample_config: DictConfig = dataset_info.sampling
            model_logger.info(
                f"Sampling strategy: {sample_config.strategy}, "
                f"kf_every={getattr(sample_config, 'kf_every', 'n/a')}, "
                f"seq_id_map={dataset_info.seq_id_map}"
            )
            seq_id_map: dict = ensure_seq_id_map(
                map_path=dataset_info.seq_id_map,
                dataset=dataset,
                sample_config=sample_config,
                force=bool(getattr(hydra_cfg, "rebuild_seq_id_map", False)),
                logger=model_logger,
            )
            requested_seq_names = getattr(hydra_cfg, "eval_seq_names", None)
            seq_id_map = select_seq_id_map(seq_id_map, requested_seq_names)
            if requested_seq_names is not None:
                model_logger.info(
                    f"Selected {len(seq_id_map)} eval sequences "
                    f"({sum(len(ids) for ids in seq_id_map.values())} frames): "
                    f"{list(seq_id_map)}"
                )

            model_logger.info(f"Evaluating {dataset_name} with {model_keyname}...")
            samples_csv = osp.join(output_root, "_all_samples.csv")
            if osp.exists(samples_csv):
                os.remove(samples_csv)
            runtime_csv = osp.join(output_root, "runtime_manifest.csv")
            if osp.exists(runtime_csv):
                os.remove(runtime_csv)
            depth_samples_csv = osp.join(output_root, "_depth_samples.csv")
            depth_metrics_enabled = bool(
                getattr(hydra_cfg, "depth_metrics_enabled", False)
            )
            depth_metrics_only = bool(
                getattr(hydra_cfg, "depth_metrics_only", False)
            )
            if depth_metrics_only and not depth_metrics_enabled:
                raise ValueError("depth_metrics_only requires depth_metrics_enabled=true")
            if depth_metrics_enabled and osp.exists(depth_samples_csv):
                os.remove(depth_samples_csv)

            save_ply = bool(getattr(hydra_cfg, "save_ply", False))
            configured_ply_names = getattr(hydra_cfg, "save_ply_seq_names", None)
            save_ply_seq_names = (
                None
                if configured_ply_names is None
                else {str(name) for name in configured_ply_names}
            )
            saved_ply_seq_names = set()
            save_ply_max_seqs = int(getattr(hydra_cfg, "save_ply_max_seqs", 2))
            save_ply_max_points = int(getattr(hydra_cfg, "save_ply_max_points", 300000))
            save_traj_bev = bool(getattr(hydra_cfg, "save_traj_bev", True))
            ply_dir = osp.join(output_root, "ply")
            traj_root = osp.join(output_root, "traj")
            if save_ply:
                os.makedirs(ply_dir, exist_ok=True)
                selection = (
                    f"explicit sequences={sorted(save_ply_seq_names)}"
                    if save_ply_seq_names is not None
                    else f"first {save_ply_max_seqs} sequences"
                )
                model_logger.info(
                    f"Saving PLYs for {selection} under {ply_dir} "
                    f"(max_points={save_ply_max_points})"
                )
            if save_traj_bev:
                os.makedirs(traj_root, exist_ok=True)
                model_logger.info(
                    f"Saving per-seq traj (PC-Sim3 aligned BEV) under {traj_root}"
                )
            save_metric_frame_cache_enabled = bool(
                getattr(hydra_cfg, "save_metric_frame_cache", False)
            )
            frame_cache_dir = osp.join(output_root, "frame_point_cache")
            save_pc_metric_cache_enabled = bool(
                getattr(hydra_cfg, "save_pc_metric_cache", False)
            )
            pc_metric_cache_dir = osp.join(output_root, "pc_metric_cache")
            if save_pc_metric_cache_enabled:
                os.makedirs(pc_metric_cache_dir, exist_ok=True)
                model_logger.info(
                    "Saving post-alignment point clouds and NN-distance caches "
                    f"under {pc_metric_cache_dir}"
                )

            num_ok = 0
            num_ply_saved = 0
            for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
                max_eval_frames = int(getattr(hydra_cfg, "max_eval_frames", 0))
                if max_eval_frames > 0:
                    ids = list(ids)[:max_eval_frames]
                    model_logger.info(
                        f"Smoke frame cap for {seq_name}: {len(ids)} frames"
                    )
                since = time.time()
                data = dataset.get_data(sequence_name=seq_name, ids=ids)
                filelist: list         = data['image_paths']
                images                 = data.get('images')
                gt_pts: np.ndarray     = data['pointclouds']
                valid_mask: np.ndarray = data['valid_mask']
                alignment_gt_mask = data.get("alignment_gt_mask")
                model_logger.info(
                    f"Data loading time for sequence {seq_name}: {time.time() - since:.2f} seconds."
                )

                since = time.time()
                if data.get("image_hw") is not None:
                    data_h, data_w = (int(v) for v in data["image_hw"])
                elif images is not None:
                    data_h, data_w = images.shape[-2:]
                else:
                    raise ValueError(
                        f"{dataset_name}/{seq_name}: dataset returned neither images nor image_hw"
                    )
                metric_indices = data.get("metric_indices")
                with open_dict(hydra_cfg):
                    hydra_cfg.mv_recon_output_indices = (
                        None
                        if metric_indices is None
                        else [int(value) for value in np.asarray(metric_indices).reshape(-1)]
                    )
                clear_model_runtime(model)
                infer_out = infer_mv_pointclouds(
                    filelist, model, hydra_cfg, (data_h, data_w)
                )
                pred_pts, pred_c2w, pred_mask, observation_mask, alignment_pred_mask, dense_aux = (
                    unpack_infer_mv_result(infer_out)
                )
                pred_pts, pred_mask, observation_mask, alignment_pred_mask = (
                    select_metric_dense_outputs(
                        pred_pts,
                        pred_mask,
                        observation_mask,
                        alignment_pred_mask,
                        metric_indices,
                        len(filelist),
                    )
                )
                runtime_required = bool(
                    getattr(hydra_cfg, "require_runtime_manifest", False)
                )
                runtime = None
                try:
                    runtime = require_model_runtime(model)
                except RuntimeError:
                    if runtime_required:
                        raise
                if bool(getattr(hydra_cfg, "strict_pointcloud_protocol", False)):
                    validate_formal_pointcloud_protocol(
                        model_name=model_keyname,
                        dataset_name=dataset_name,
                        runtime=runtime,
                        pred_c2w=pred_c2w,
                        with_scale=with_scale,
                        metric_frame_ids=data.get("metric_frame_ids"),
                        eval_threshold=float(eval_opts["eval_threshold"]),
                        eval_thresholds=eval_thresholds,
                        voxel_size=float(eval_opts["voxel_size"]),
                        icp_threshold=float(eval_opts["icp_threshold"]),
                        nearest_depth_to_gt=nearest_depth_to_gt,
                        pointmap_resize_mode=str(
                            getattr(hydra_cfg, "pointmap_resize_mode", "")
                        ),
                        alignment_depth_max=alignment_depth_max,
                        observation_mask=observation_mask,
                    )
                assert pred_pts.shape == gt_pts.shape, (
                    f"Predicted points shape {pred_pts.shape} does not match "
                    f"ground truth shape {gt_pts.shape}."
                )
                if runtime is not None:
                    source_ids = data.get("source_frame_ids", ids)
                    metric_ids = data.get("metric_frame_ids")
                    if metric_ids is None and metric_indices is not None:
                        source_array = np.asarray(list(source_ids)).reshape(-1)
                        metric_ids = source_array[np.asarray(metric_indices, dtype=np.int64)]
                    manifest_csv, manifest_json = write_runtime_manifest(
                        output_root=output_root,
                        model_name=model_keyname,
                        dataset_name=dataset_name,
                        sequence_name=seq_name,
                        task="pointcloud",
                        filelist=filelist,
                        runtime=runtime,
                        metric_frame_ids=metric_ids,
                        metric_frame_count=len(gt_pts),
                        checkpoint=_checkpoint_label(model_info),
                        device=str(hydra_cfg.device),
                        protocol={
                            "source_frame_ids": source_ids,
                            "alignment": align_mode,
                            "alignment_pred_depth_max": alignment_depth_max,
                            "nearest_depth_to_gt": nearest_depth_to_gt,
                            "pointmap_resize_mode": str(getattr(hydra_cfg, "pointmap_resize_mode", "n/a")),
                            "voxel_size": float(eval_opts["voxel_size"]),
                            "icp_threshold": float(eval_opts["icp_threshold"]),
                            "f1_thresholds": eval_thresholds,
                            "formal_pose_summary": False,
                        },
                    )
                    model_logger.info(
                        f"Saved runtime manifest: {manifest_json} (+ {manifest_csv})"
                    )
                model_logger.info(
                    f"Inference time for sequence {seq_name}: {time.time() - since:.2f} seconds."
                )

                raw_valid_count = int(np.count_nonzero(valid_mask))
                eval_valid_mask, pred_mask = restrict_masks_to_observed_fov(
                    valid_mask, pred_mask, observation_mask
                )
                if alignment_gt_mask is not None:
                    alignment_gt_mask = np.asarray(alignment_gt_mask, dtype=bool)
                    if observation_mask is not None:
                        alignment_gt_mask &= np.asarray(observation_mask, dtype=bool)
                if alignment_pred_mask is not None and observation_mask is not None:
                    alignment_pred_mask = (
                        np.asarray(alignment_pred_mask, dtype=bool)
                        & np.asarray(observation_mask, dtype=bool)
                    )
                if observation_mask is not None:
                    model_logger.info(
                        f"Restricting point-cloud metrics to observed FOV: "
                        f"observed={int(observation_mask.sum())}/"
                        f"{observation_mask.size}, GT-valid="
                        f"{int(eval_valid_mask.sum())}/{raw_valid_count}"
                    )

                if depth_metrics_enabled:
                    metric_w2c = data.get("metric_extrs")
                    if metric_w2c is None:
                        raise ValueError(
                            f"{dataset_name}/{seq_name}: depth metrics require metric_extrs"
                        )
                    if dense_aux is None or "pred_depth" not in dense_aux:
                        raise ValueError(
                            f"{model_keyname}: depth metrics require explicit pred_depth"
                        )
                    pred_depth = np.asarray(dense_aux["pred_depth"], dtype=np.float32)
                    if len(pred_depth) == len(filelist) and metric_indices is not None:
                        pred_depth = pred_depth[np.asarray(metric_indices, dtype=np.int64)]
                    gt_depth = camera_z_from_world(gt_pts, np.asarray(metric_w2c))
                    scale_gt_max_cfg = getattr(
                        hydra_cfg, "depth_scale_gt_max", None
                    )
                    scale_gt_max = (
                        None
                        if scale_gt_max_cfg is None
                        else float(scale_gt_max_cfg)
                    )
                    depth_metrics = evaluate_scale_aligned_depth_maps(
                        pred_depth=pred_depth,
                        gt_depth=gt_depth,
                        gt_mask=eval_valid_mask,
                        pred_mask=pred_mask,
                        observation_mask=observation_mask,
                        alignment_gt_depth_max=scale_gt_max,
                    )
                    write_csv(
                        depth_samples_csv,
                        {"seq": seq_name, **depth_metrics},
                    )
                    model_logger.info(
                        f"Depth {seq_name}: scale={depth_metrics['depth_scale']:.6f}, "
                        f"AbsRel={depth_metrics['abs_rel']:.6f}, "
                        f"RMSE={depth_metrics['rmse']:.6f}, "
                        f"delta1={depth_metrics['delta1']:.2f} "
                        f"(N={depth_metrics['num_depth_pixels']})"
                    )
                    if save_metric_frame_cache_enabled:
                        safe_name = seq_name.replace("/", "_").replace(" ", "_")
                        cache_path = osp.join(frame_cache_dir, f"{safe_name}.npz")
                        save_metric_frame_cache(
                            cache_path,
                            pred_world=pred_pts,
                            pred_c2w=pred_c2w,
                            pred_mask=pred_mask,
                            observation_mask=observation_mask,
                            metric_indices=metric_indices,
                            metric_frame_ids=data.get("metric_frame_ids"),
                            sequence_name=seq_name,
                            pred_depth=pred_depth,
                            pred_local_points=(
                                None if dense_aux is None else dense_aux.get("pred_local_points")
                            ),
                        )
                        model_logger.info(
                            f"Saved lossless metric-frame XYZ cache: {cache_path} "
                            f"({osp.getsize(cache_path) / (1024**3):.2f} GiB)"
                        )
                    if depth_metrics_only:
                        num_ok += 1
                        torch.cuda.empty_cache()
                        continue
                pc_eval_roi = getattr(hydra_cfg, "pc_eval_roi", None)
                if pc_eval_roi is not None:
                    if len(pc_eval_roi) != 4:
                        raise ValueError(
                            f"pc_eval_roi must be [y0,y1,x0,x1], got {pc_eval_roi}"
                        )
                    y0, y1, x0, x1 = (int(value) for value in pc_eval_roi)
                    if not (0 <= y0 < y1 <= data_h and 0 <= x0 < x1 <= data_w):
                        raise ValueError(
                            f"pc_eval_roi={pc_eval_roi} outside data {data_h}x{data_w}"
                        )
                    roi_mask = np.zeros_like(eval_valid_mask, dtype=bool)
                    roi_mask[:, y0:y1, x0:x1] = True
                    eval_valid_mask = eval_valid_mask & roi_mask
                    if alignment_gt_mask is not None:
                        alignment_gt_mask &= roi_mask
                    if pred_mask is not None:
                        pred_mask = np.asarray(pred_mask, dtype=bool) & roi_mask
                    if alignment_pred_mask is not None:
                        alignment_pred_mask = (
                            np.asarray(alignment_pred_mask, dtype=bool) & roi_mask
                        )
                    model_logger.info(
                        f"Applying common point-cloud evaluation ROI: "
                        f"y={y0}:{y1}, x={x0}:{x1}, "
                        f"GT-valid={int(eval_valid_mask.sum())}/{raw_valid_count}"
                    )

                want_ply = save_ply and (
                    seq_name in save_ply_seq_names
                    if save_ply_seq_names is not None
                    else num_ply_saved < save_ply_max_seqs
                )
                eval_gt_mask_snapshot = np.asarray(eval_valid_mask, dtype=bool).copy()
                eval_pred_mask_snapshot = np.asarray(pred_mask, dtype=bool).copy()
                since = time.time()
                try:
                    metrics = evaluate_reconstruction(
                        pred_pts=pred_pts,
                        gt_pts=gt_pts,
                        gt_mask=eval_valid_mask,
                        pred_mask=pred_mask,
                        alignment_gt_mask=alignment_gt_mask,
                        alignment_pred_mask=alignment_pred_mask,
                        icp_threshold=eval_opts["icp_threshold"],
                        voxel_size=eval_opts["voxel_size"],
                        eval_threshold=eval_opts["eval_threshold"],
                        eval_thresholds=eval_thresholds,
                        with_scale=with_scale,
                    )
                except ValueError as exc:
                    model_logger.warning(
                        f"[{dataset_name} {seq_idx}/{len(seq_id_map)}] "
                        f"Seq {seq_name}: skip ({exc})"
                    )
                    continue
                if not np.array_equal(eval_valid_mask, eval_gt_mask_snapshot):
                    raise RuntimeError(
                        "Point-cloud alignment mutated the final GT evaluation mask"
                    )
                if not np.array_equal(pred_mask, eval_pred_mask_snapshot):
                    raise RuntimeError(
                        "Point-cloud alignment mutated the final prediction evaluation mask"
                    )
                model_logger.info(
                    f"Align+metrics time for sequence {seq_name}: "
                    f"{time.time() - since:.2f} seconds."
                )
                if save_pc_metric_cache_enabled:
                    safe_name = seq_name.replace("/", "_").replace(" ", "_")
                    cache_path = osp.join(pc_metric_cache_dir, f"{safe_name}.npz")
                    save_pc_metric_cache(
                        cache_path,
                        metrics=metrics,
                        sequence_name=seq_name,
                        dataset_name=dataset_name,
                        model_name=model_keyname,
                    )
                    model_logger.info(
                        f"Saved reusable point-cloud metric cache: {cache_path} "
                        f"({osp.getsize(cache_path) / (1024**2):.1f} MiB)"
                    )
                threshold_report = ""
                if len(eval_thresholds) > 1:
                    threshold_report = "".join(
                        f"F1@{threshold:.2f}: "
                        f"{metrics[f'f1_{threshold:g}']:.2f} "
                        for threshold in eval_thresholds
                    )
                model_logger.info(
                    f"[{dataset_name} {seq_idx}/{len(seq_id_map)}] Seq: {seq_name}, "
                    f"Acc: {metrics['accuracy']:.6f}, Comp: {metrics['completeness']:.6f}, "
                    f"Chamfer: {metrics['chamfer']:.6f}, "
                    f"F1: {metrics['f1']:.2f} "
                    f"{threshold_report}"
                    f"(Npred={metrics['num_pred']}, Ngt={metrics['num_gt']})"
                )
                if alignment_pred_mask is not None:
                    model_logger.info(
                        "Alignment-only filtering: "
                        f"Nalign={metrics['num_alignment_correspondences']}, "
                        f"NevalCorr={metrics['num_correspondences']}"
                    )

                if runtime is not None:
                    source_ids = data.get("source_frame_ids", ids)
                    metric_ids = data.get("metric_frame_ids")
                    if metric_ids is None and metric_indices is not None:
                        source_array = np.asarray(list(source_ids)).reshape(-1)
                        metric_ids = source_array[
                            np.asarray(metric_indices, dtype=np.int64)
                        ]
                    write_runtime_manifest(
                        output_root=output_root,
                        model_name=model_keyname,
                        dataset_name=dataset_name,
                        sequence_name=seq_name,
                        task="pointcloud",
                        filelist=filelist,
                        runtime=runtime,
                        metric_frame_ids=metric_ids,
                        metric_frame_count=len(gt_pts),
                        checkpoint=_checkpoint_label(model_info),
                        device=str(hydra_cfg.device),
                        protocol={
                            "source_frame_ids": source_ids,
                            "alignment": align_mode,
                            "alignment_pred_depth_max": alignment_depth_max,
                            "nearest_depth_to_gt": nearest_depth_to_gt,
                            "pointmap_resize_mode": str(
                                getattr(hydra_cfg, "pointmap_resize_mode", "n/a")
                            ),
                            "gt_grid_h": int(data_h),
                            "gt_grid_w": int(data_w),
                            "raw_gt_valid_count": raw_valid_count,
                            "eval_gt_valid_count": int(eval_valid_mask.sum()),
                            "eval_pred_valid_count": int(pred_mask.sum()),
                            "alignment_gt_valid_count": (
                                int(alignment_gt_mask.sum())
                                if alignment_gt_mask is not None
                                else int(eval_valid_mask.sum())
                            ),
                            "alignment_pred_valid_count": (
                                int(alignment_pred_mask.sum())
                                if alignment_pred_mask is not None
                                else int(pred_mask.sum())
                            ),
                            "alignment_correspondence_count": int(
                                metrics["num_alignment_correspondences"]
                            ),
                            "evaluation_correspondence_count": int(
                                metrics["num_correspondences"]
                            ),
                            "voxel_size": float(eval_opts["voxel_size"]),
                            "icp_threshold": float(eval_opts["icp_threshold"]),
                            "f1_thresholds": eval_thresholds,
                            "confidence_filter": False,
                            "sky_filter": False,
                            "outlier_filter": False,
                            "alignment_only_mask_isolated": True,
                            "formal_pose_summary": False,
                        },
                    )

                if want_ply:
                    safe_name = seq_name.replace("/", "_").replace(" ", "_")
                    ply_images = images
                    if ply_images is None:
                        ply_images = load_metric_rgb(
                            data.get("metric_image_paths", []), (data_h, data_w)
                        )
                    # Dense Sim3+ICP clouds + GT-grid RGB (not metric voxel cloud).
                    pred_xyz, pred_rgb, gt_xyz, gt_rgb = colored_aligned_clouds_for_ply(
                        pred_pts=pred_pts,
                        gt_pts=gt_pts,
                        gt_mask=eval_valid_mask,
                        images=ply_images,
                        T_umeyama=metrics["T_umeyama"],
                        T_icp=metrics.get("T_icp"),
                        pred_mask=pred_mask,
                    )
                    pred_save, pred_cols = subsample_points(
                        pred_xyz, save_ply_max_points, seed=seq_idx, colors=pred_rgb
                    )
                    gt_save, gt_cols = subsample_points(
                        gt_xyz, save_ply_max_points, seed=seq_idx, colors=gt_rgb
                    )
                    pred_path = osp.join(ply_dir, f"{safe_name}_pred_aligned.ply")
                    gt_path = osp.join(ply_dir, f"{safe_name}_gt.ply")
                    save_xyzrgb_ply(pred_path, pred_save, pred_cols)
                    save_xyzrgb_ply(gt_path, gt_save, gt_cols)
                    num_ply_saved += 1
                    saved_ply_seq_names.add(seq_name)
                    model_logger.info(
                        f"Saved RGB PLY sample [{num_ply_saved}]: "
                        f"{pred_path} ({len(pred_save)} pts), "
                        f"{gt_path} ({len(gt_save)} pts)"
                    )

                if save_traj_bev and pred_c2w is not None:
                    gt_c2w = gt_c2w_from_batch(data)
                    if gt_c2w is None:
                        model_logger.warning(
                            f"Seq {seq_name}: no GT extrs in batch, skip traj BEV"
                        )
                    else:
                        safe_name = seq_name.replace("/", "_").replace(" ", "_")
                        seq_traj_dir = osp.join(traj_root, safe_name)
                        try:
                            aln_path, bev_path = save_sequence_traj_bev(
                                save_dir=seq_traj_dir,
                                seq_name=seq_name,
                                pred_c2w=pred_c2w,
                                gt_c2w=gt_c2w,
                                T_umeyama=metrics["T_umeyama"],
                                T_icp=metrics.get("T_icp"),
                                title=f"{model_keyname} | {dataset_name} | {seq_name}",
                                alignment_label=align_mode,
                                trajectory_with_scale=with_scale,
                                verbose=bool(getattr(hydra_cfg, "verbose", False)),
                            )
                            model_logger.info(
                                f"Saved traj BEV: {bev_path} (+ {aln_path})"
                            )
                        except Exception as exc:
                            model_logger.warning(
                                f"Seq {seq_name}: traj BEV failed ({exc})"
                            )

                # Drop bulky transform mats from CSV row path
                metrics.pop("T_umeyama", None)
                metrics.pop("T_icp", None)

                write_csv(samples_csv, metrics_to_csv_row(seq_name, metrics))
                for key in aggregate_metric_keys:
                    all_data_dict[key] += float(metrics[key])
                num_ok += 1

                torch.cuda.empty_cache()

            if save_ply and save_ply_seq_names is not None:
                missing_ply_names = save_ply_seq_names - saved_ply_seq_names
                if missing_ply_names:
                    raise RuntimeError(
                        "Requested qualitative PLY sequences were not saved: "
                        f"{sorted(missing_ply_names)}"
                    )

            if num_ok == 0:
                model_logger.warning(
                    f"No successful sequences for {dataset_name}; skip aggregate CSV."
                )
                continue

            if depth_metrics_only:
                model_logger.info(
                    f"Completed depth-only evaluation on {num_ok} sequences for "
                    f"{dataset_name}; metrics: {depth_samples_csv}"
                )
                continue

            metric_dict = {
                metric: all_data_dict[metric] / num_ok
                for metric in aggregate_metric_keys
            }

            statistics_file = osp.join(model_output_dir, f"{dataset_name}-metric")
            if getattr(hydra_cfg, "save_suffix", None) is not None:
                statistics_file += f"-{hydra_cfg.save_suffix}"
            statistics_file += ".csv"
            write_csv(statistics_file, {"model": model_keyname, **metric_dict})

        del model
        torch.cuda.empty_cache()
        model_logger.info(f"Finished evaluating {model_keyname} on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
