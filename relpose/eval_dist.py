import os
import os.path as osp
import logging
import random
import traceback
import numpy as np
import torch
import hydra

from tqdm import tqdm
from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from utils.debug import setup_debug
from utils.files import list_imgs_a_sequence, get_all_sequences, _load_json_summary
from utils.messages import set_default_arg, write_csv, save_list_of_matrices, model_checkpoint_path
from relpose.evo_utils import calculate_averages, load_traj, eval_metrics, plot_trajectory, get_tum_poses, save_tum_poses
from relpose.output_names import resolve_model_output_slug
from relpose.long_pose_protocol import (
    bind_dataset_context,
    load_resumable_poses,
    trajectory_length,
    validate_predicted_poses,
)
from relpose.forward_timing import (
    forward_timing_enabled,
    reset_forward_timing,
    save_forward_timing,
    summarize_forward_timing,
)
from mv_recon.runtime_manifest import (
    clear_model_runtime,
    require_model_runtime,
    write_runtime_manifest,
)


def _checkpoint_label(model_info: DictConfig) -> str:
    cfg = model_info.cfg
    for key in ("ckpt", "checkpoint", "pretrained_model_name_or_path"):
        value = cfg.get(key, None)
        if value is not None:
            return str(value)
    return "unreported"


def _assert_formal_pose_dataset(dataset_name: str, hydra_cfg: DictConfig) -> None:
    if not bool(hydra_cfg.get("formal_pose_protocol", False)):
        return
    normalized = str(dataset_name).lower()
    if normalized.startswith("7scenes") or normalized.startswith("tum-dynamics"):
        raise ValueError(
            f"{dataset_name} is point-cloud-only in the formal protocol; "
            "pose evaluation is diagnostic only"
        )


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    logger = logging.getLogger("relpose-dist")
    seed = int(hydra_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_eval_models: DictConfig   = hydra_cfg.eval_models    # see configs/evaluation/relpose_stride1.yaml
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose_stride1.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model, decide the model name (to save)
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]
        model_pretrained_path = model_checkpoint_path(model_info.cfg)

        # 0.3 route the correct infer function for the model (shared across datasets)
        infer_func_cfg = model_info.get(
            "infer_cameras_c2w",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_cameras_c2w',
                '_partial_': True,
            })
        )
        infer_cameras_c2w = hydra.utils.instantiate(infer_func_cfg)

        model_logger = logging.getLogger(f"relpose-dist-{model_keyname}")
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            _assert_formal_pose_dataset(dataset_name, hydra_cfg)
            # 1. look up dataset config from configs/data, decide the dataset name
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset_info = all_data_info[dataset_name]

            # 2. get the sequence list
            seq_list = get_all_sequences(dataset_info)

            # 3. (Re-)build the model for this dataset to make sure stateful
            # caches (e.g. lingbot-map's KV cache that locks `patches_per_frame`
            # on the first frame it sees) are freshly initialized for the new
            # image resolution / token count.
            model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
            model = model.eval()
            bind_dataset_context(model, dataset_name, dataset_info)
            output_slug = resolve_model_output_slug(model_keyname, model_info, model)
            output_root = osp.join(hydra_cfg.output_dir, output_slug, dataset_name)
            os.makedirs(output_root, exist_ok=True)
            logger.info(
                f"[{idx_model}/{len(all_eval_models)}][{idx_dataset}/{len(all_eval_datasets)}] "
                f"Re-loaded Model {model_keyname} from {model_pretrained_path} for dataset {dataset_name} "
                f"(output_slug={output_slug})"
            )
            model_logger.info(
                f"[{idx_dataset}/{len(all_eval_datasets)}] Infering relpose(c2w) on {dataset_name} "
                f"dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}"
            )

            # Per-dataset stride takes precedence over the global one.
            global_stride = hydra_cfg.get("pose_eval_stride", 1)
            pose_eval_stride = int(dataset_info.get("pose_eval_stride", global_stride))
            if pose_eval_stride != global_stride:
                model_logger.info(
                    f"[{dataset_name}] per-dataset pose_eval_stride={pose_eval_stride} "
                    f"(overrides global={global_stride})"
                )

            # For JSON-driven datasets (e.g. oxford_spires-long), the GT row
            # for the i-th image is given by JSON's ``gt_row_indices`` rather
            # than ``np.arange``. We therefore slice the full TUM trajectory
            # by these explicit indices, then re-apply ``pose_eval_stride``.
            json_summary = None
            if dataset_info.img.get("source", None) == "json":
                json_summary = _load_json_summary(dataset_info.json_file)

            results = []
            timing_results = []
            runtime_csv = osp.join(output_root, "runtime_manifest.csv")
            if osp.isfile(runtime_csv):
                os.remove(runtime_csv)
            tbar = tqdm(seq_list, desc=f"[{dataset_name} eval]")
            for seq in tbar:
                try:
                    # 4.1 list all images of this sequence
                    filelist = list_imgs_a_sequence(dataset_info, seq)
                    filelist = filelist[:: pose_eval_stride]
                    max_eval_frames = int(hydra_cfg.get("max_eval_frames", 0) or 0)
                    if max_eval_frames > 0:
                        filelist = filelist[:max_eval_frames]

                    seq_save_dir = osp.join(output_root, seq)
                    os.makedirs(seq_save_dir, exist_ok=True)

                    # 4.2 real inference
                    # pr_poses: c2w poses, (N, 3, 4), in torch
                    # pr_intrs: focals + pps, (N, 3, 3), in numpy
                    pr_poses = load_resumable_poses(
                        seq_save_dir,
                        len(filelist),
                        enabled=bool(hydra_cfg.get("resume_existing", False)),
                    )
                    if pr_poses is None:
                        if forward_timing_enabled(hydra_cfg):
                            warmup_runs = int(hydra_cfg.get("forward_timing_warmup_runs", 1))
                            warmup_frames = int(
                                hydra_cfg.get("forward_timing_warmup_frames", len(filelist))
                                or len(filelist)
                            )
                            warmup_filelist = filelist[: min(warmup_frames, len(filelist))]
                            for warmup_index in range(warmup_runs):
                                reset_forward_timing(model)
                                infer_cameras_c2w(warmup_filelist, model, hydra_cfg)
                                model_logger.info(
                                    f"[{dataset_name}/{seq}] completed forward warmup "
                                    f"{warmup_index + 1}/{warmup_runs} "
                                    f"({len(warmup_filelist)} frames)"
                                )
                        reset_forward_timing(model)
                        clear_model_runtime(model)
                        pr_poses, pr_intrs = infer_cameras_c2w(filelist, model, hydra_cfg)
                        pr_poses = validate_predicted_poses(pr_poses, len(filelist))
                        runtime = require_model_runtime(model)
                        manifest_csv, manifest_json = write_runtime_manifest(
                            output_root=output_root,
                            model_name=model_keyname,
                            dataset_name=dataset_name,
                            sequence_name=seq,
                            task="pose",
                            filelist=filelist,
                            runtime=runtime,
                            metric_frame_ids=range(len(filelist)),
                            metric_frame_count=len(filelist),
                            checkpoint=_checkpoint_label(model_info),
                            device=str(hydra_cfg.device),
                            protocol={
                                "source_frame_ids": range(len(filelist)),
                                "alignment": "Sim3",
                                "pose_metrics": ["ATE", "RPE-t", "RPE-r"],
                                "formal_pose_summary": True,
                            },
                        )
                        model_logger.info(
                            f"Saved runtime manifest: {manifest_json} (+ {manifest_csv})"
                        )
                        if forward_timing_enabled(hydra_cfg):
                            timing = summarize_forward_timing(model, len(filelist))
                            if timing is None:
                                raise RuntimeError(
                                    f"{model_keyname} did not record a forward timing sample"
                                )
                            timing.update({
                                "model": output_slug,
                                "dataset": dataset_name,
                                "sequence": seq,
                            })
                            save_forward_timing(
                                osp.join(seq_save_dir, "forward_timing.json"), timing
                            )
                            timing_results.append(timing)
                            model_logger.info(
                                f"[{dataset_name}/{seq}] forward-only: "
                                f"{timing['avg_forward_ms_per_frame']:.3f} ms/frame, "
                                f"{timing['forward_fps']:.3f} FPS "
                                f"({timing['num_frames']} frames, "
                                f"{timing['forward_calls']} model calls)"
                            )
                    else:
                        if forward_timing_enabled(hydra_cfg):
                            raise RuntimeError(
                                "Forward FPS measurement cannot reuse pred_poses.npy; "
                                "set resume_existing=false"
                            )
                        pr_intrs = None
                        if bool(hydra_cfg.get("formal_pose_protocol", False)):
                            safe_seq = seq.replace("/", "_").replace(" ", "_")
                            manifest_path = osp.join(
                                output_root, "runtime_manifests", f"{safe_seq}.json"
                            )
                            if not osp.isfile(manifest_path):
                                raise RuntimeError(
                                    "Formal pose resume found pred_poses.npy without a "
                                    f"runtime manifest: {manifest_path}. Rerun with "
                                    "resume_existing=false."
                                )
                        model_logger.info(
                            f"[{dataset_name}/{seq}] Reusing validated pred_poses.npy "
                            f"({len(filelist)} poses); skipping model forward"
                        )
                    pred_traj = get_tum_poses(pr_poses)

                    # 4.3 save predicted poses & intrinsics
                    # save predicted poses
                    save_tum_poses(pred_traj, osp.join(output_root, seq, "pred_traj.txt"), verbose=hydra_cfg.verbose)
                    save_tum_poses(pred_traj, osp.join(output_root, seq, "pred_traj_raw.txt"), verbose=False)
                    np.save(osp.join(seq_save_dir, "pred_poses.npy"), pr_poses)
                    save_list_of_matrices(pr_poses.numpy().tolist(), osp.join(seq_save_dir, "pred_poses.json"))
                    # save predicted intrinsics (if available)
                    if pr_intrs is not None:
                        np.save(osp.join(seq_save_dir, "pred_intrinsics.npy"), pr_intrs)
                        save_list_of_matrices(pr_intrs.tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))

                    # 4.4 read ground truth trajectory
                    if json_summary is not None:
                        # JSON-driven: select GT rows for the selected image
                        # window (e.g. 3840 frames), then apply stride.
                        gt_row_indices = json_summary["scenes"][seq]["gt_row_indices"]
                        gt_row_indices = gt_row_indices[:: pose_eval_stride]
                        if max_eval_frames > 0:
                            gt_row_indices = gt_row_indices[:max_eval_frames]
                        gt_traj = load_traj(
                            gt_traj_file  = dataset_info.anno.path.format(seq=seq),
                            traj_format   = dataset_info.anno.format,
                            frame_indices = gt_row_indices,
                        )
                    else:
                        gt_traj = load_traj(
                            gt_traj_file = dataset_info.anno.path.format(seq=seq),
                            traj_format  = dataset_info.anno.format,
                            stride       = pose_eval_stride,
                            num_frames   = max_eval_frames or None,
                        )
                    # Note: pose_eval_stride is resolved above from dataset_info or hydra_cfg

                    if gt_traj is not None and trajectory_length(gt_traj) != trajectory_length(pred_traj):
                        raise ValueError(
                            f"GT/pred pose count mismatch for {dataset_name}/{seq}: "
                            f"gt={trajectory_length(gt_traj)}, pred={trajectory_length(pred_traj)}"
                        )

                    # 4.5 evaluate predicted trajectory with ground truth trajectory, plot the trajectory
                    if gt_traj is not None:
                        save_tum_poses(gt_traj, osp.join(output_root, seq, "gt_traj.txt"), verbose=False)
                        ate, rpe_trans, rpe_rot = eval_metrics(
                            pred_traj, gt_traj,
                            seq      = seq,
                            filename = osp.join(output_root, seq, "eval_metric.txt"),
                            verbose  = hydra_cfg.verbose,
                        )
                        plot_trajectory(pred_traj, gt_traj, title=seq, filename=osp.join(output_root, seq, "vis.png"), verbose=hydra_cfg.verbose)
                    else:
                        raise ValueError(f"Ground truth trajectory not found for sequence {seq} in dataset {dataset_name}.")

                    # 4.6 save sequence metrics to csv
                    seq_metrics = {
                        "model": output_slug,
                        "dataset": dataset_name,
                        "seq": seq,
                        "ATE": ate,
                        "RPE trans": rpe_trans,
                        "RPE rot": rpe_rot,
                    }
                    write_csv(
                        osp.join(output_root, "seq_metrics.csv"),
                        seq_metrics,
                        key_fields=("model", "dataset", "seq"),
                    )
                    results.append((seq, ate, rpe_trans, rpe_rot))

                    # 4.7. update metric for a sequence to tqdm bar
                    tbar.set_postfix_str(f"Seq {seq} ATE: {ate:5.2f} | RPE-trans: {rpe_trans:5.2f} | RPE-rot: {rpe_rot:5.2f}")

                except Exception as e:
                    if "out of memory" in str(e):
                        # Handle OOM
                        oom_detail = traceback.format_exc()
                        if torch.cuda.is_available():
                            oom_detail += (
                                "\nCUDA memory: "
                                f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
                                f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB, "
                                f"peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                                f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB\n"
                            )
                        torch.cuda.empty_cache()  # Clear the CUDA memory
                        with open(osp.join(output_root, "error_log.txt"), "a") as f:
                            f.write(f"OOM error in sequence {seq}, skipping this sequence.\n")
                            f.write(oom_detail)
                            f.write("\n")
                        logger.error("OOM in sequence %s\n%s", seq, oom_detail)
                        print(f"OOM error in sequence {seq}, skipping...")
                    elif "Degenerate covariance rank" in str(
                        e
                    ) or "Eigenvalues did not converge" in str(e):
                        # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                        with open(osp.join(output_root, "error_log.txt"), "a") as f:
                            f.write(f"Exception in sequence {seq}: {str(e)}\n")
                        print(f"Traj evaluation error in sequence {seq}, skipping.")
                    else:
                        raise e  # Rethrow if it's not an expected exception

            avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
            if bool(hydra_cfg.get("require_complete_eval", False)) and len(results) != len(seq_list):
                completed = {seq for seq, *_ in results}
                missing = [seq for seq in seq_list if seq not in completed]
                raise RuntimeError(
                    f"Incomplete evaluation for {model_keyname}/{dataset_name}: "
                    f"completed={len(results)}/{len(seq_list)}, missing={missing}"
                )
            if forward_timing_enabled(hydra_cfg):
                if not timing_results:
                    raise RuntimeError(f"No forward timing results for {model_keyname}/{dataset_name}")
                total_frames = sum(item["num_frames"] for item in timing_results)
                total_seconds = sum(item["forward_seconds"] for item in timing_results)
                timing_metrics = {
                    "model": output_slug,
                    "dataset": dataset_name,
                    "sequences": len(timing_results),
                    "num_frames": total_frames,
                    "forward_calls": sum(item["forward_calls"] for item in timing_results),
                    "forward_seconds": total_seconds,
                    "avg_forward_ms_per_frame": 1000.0 * total_seconds / total_frames,
                    "forward_fps": total_frames / total_seconds,
                }
                timing_csv = osp.join(output_root, "forward_timing_summary.csv")
                if osp.isfile(timing_csv):
                    os.remove(timing_csv)
                write_csv(timing_csv, timing_metrics)
                save_forward_timing(
                    osp.join(output_root, "forward_timing_summary.json"), timing_metrics
                )
                model_logger.info(
                    f"[{dataset_name}] aggregate forward-only: "
                    f"{timing_metrics['avg_forward_ms_per_frame']:.3f} ms/frame, "
                    f"{timing_metrics['forward_fps']:.3f} FPS"
                )

            dataset_metrics = {
                "model": output_slug,
                "ATE": avg_ate,
                "RPE trans": avg_rpe_trans,
                "RPE rot": avg_rpe_rot,
            }
            statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric-{output_slug}")
            if getattr(hydra_cfg, "save_suffix", None) is not None:
                statistics_file += f"-{hydra_cfg.save_suffix}"
            statistics_file += ".csv"
            write_csv(statistics_file, dataset_metrics)

            dataset_metrics.pop("model")  # Remove model name for logging
            model_logger.info(f"{dataset_name} - Average pose estimation metrics: {dataset_metrics}")

            # Free the model so the next dataset re-builds it with fresh
            # patches_per_frame / KV cache state.
            del model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    set_default_arg("evaluation", "relpose_stride1")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()
