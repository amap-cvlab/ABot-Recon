#!/usr/bin/env python3
"""Evaluate HorizonStream loop closure from cached no-loop trajectories."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random

import hydra
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from interfaces.horizonstream import run_horizon_loop_from_c2w
from relpose.evo_utils import (
    calculate_averages,
    eval_metrics,
    get_tum_poses,
    load_traj,
    plot_trajectory,
    save_tum_poses,
)
from relpose.long_pose_protocol import trajectory_length, validate_predicted_poses
from utils.files import _load_json_summary, get_all_sequences, list_imgs_a_sequence
from utils.messages import save_list_of_matrices, write_csv


DATASET_KEYS = {
    "kitti": "kitti-long",
    "vbr": "vbr-long",
    "oxford": "oxford_spires_processed-long",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_KEYS), required=True)
    parser.add_argument("--base-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--horizon-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", choices=["auto", "kitti", "vbr", "generic"], default="auto")
    parser.add_argument("--salad-ckpt", required=True)
    parser.add_argument("--salad-dino-weights", required=True)
    parser.add_argument("--resume-existing", choices=["true", "false"], default="true")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--max-eval-frames", type=int, default=0)
    return parser.parse_args()


def load_eval_config():
    config_dir = osp.join(str(root), "configs")
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        return compose(config_name="eval", overrides=["evaluation=relpose_stride1"])


def load_loop_config(args):
    config_path = osp.join(args.horizon_root, "configs", "horizonstream_infer.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        official = yaml.safe_load(handle) or {}
    loop_cfg = dict(official.get("online_loop", {}) or {})
    preset = args.preset
    if preset == "auto":
        preset = args.dataset if args.dataset in {"kitti", "vbr"} else "generic"
    if preset != "generic":
        groups = official.get("online_loop_groups", {}) or {}
        if preset not in groups:
            raise ValueError(f"Horizon loop preset is unavailable: {preset}")
        loop_cfg.update(groups[preset] or {})
    loop_cfg.update(
        methods=["salad"],
        salad_ckpt_path=osp.abspath(args.salad_ckpt),
        salad_dino_weights_path=osp.abspath(args.salad_dino_weights),
    )
    return preset, loop_cfg


def load_gt(dataset_info, dataset_name, seq, stride, max_frames=0):
    if dataset_info.img.get("source", None) == "json":
        summary = _load_json_summary(dataset_info.json_file)
        indices = summary["scenes"][seq]["gt_row_indices"][::stride]
        if max_frames:
            indices = indices[:max_frames]
        return load_traj(
            gt_traj_file=dataset_info.anno.path.format(seq=seq),
            traj_format=dataset_info.anno.format,
            frame_indices=indices,
        )
    return load_traj(
        gt_traj_file=dataset_info.anno.path.format(seq=seq),
        traj_format=dataset_info.anno.format,
        stride=stride,
        num_frames=max_frames or None,
    )


def main():
    args = parse_args()
    cfg = load_eval_config()
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset_name = DATASET_KEYS[args.dataset]
    dataset_info = cfg.data[dataset_name]
    stride = int(dataset_info.get("pose_eval_stride", 1))
    preset, loop_cfg = load_loop_config(args)

    model_info = cfg.model.horizonstream
    model_info.cfg.checkpoint = args.checkpoint
    model_info.cfg.horizonstream_root = args.horizon_root
    model_info.cfg.amp_dtype = "fp16"
    model = hydra.utils.instantiate(model_info.cfg).to(args.device).eval()

    output_root = osp.join(args.output_root, dataset_name)
    os.makedirs(output_root, exist_ok=True)
    seq_metrics_path = osp.join(output_root, "seq_metrics.csv")
    aggregate_path = osp.join(output_root, "aggregate_metrics.csv")
    for stale_summary in (seq_metrics_path, aggregate_path):
        if osp.isfile(stale_summary):
            os.remove(stale_summary)
    results = []
    sequences = get_all_sequences(dataset_info)
    if args.sequence:
        if args.sequence not in sequences:
            raise ValueError(f"Unknown sequence for {dataset_name}: {args.sequence}")
        sequences = [args.sequence]
    for seq in sequences:
        filelist = list_imgs_a_sequence(dataset_info, seq)[::stride]
        if args.max_eval_frames > 0:
            filelist = filelist[: args.max_eval_frames]
        base_seq_dir = osp.join(args.base_output_root, dataset_name, seq)
        base_pose_path = osp.join(base_seq_dir, "pred_poses.npy")
        if not osp.isfile(base_pose_path):
            raise FileNotFoundError(
                f"Missing Horizon no-loop pose cache: {base_pose_path}"
            )
        base_poses = validate_predicted_poses(
            torch.from_numpy(np.load(base_pose_path)), len(filelist)
        )

        seq_dir = osp.join(output_root, seq)
        os.makedirs(seq_dir, exist_ok=True)
        loop_pose_path = osp.join(seq_dir, "pred_poses.npy")
        manifest_path = osp.join(seq_dir, "runtime_manifest.json")
        use_cache = args.resume_existing == "true" and osp.isfile(loop_pose_path)
        if use_cache:
            if not osp.isfile(manifest_path):
                raise RuntimeError(
                    f"Loop pose cache has no runtime manifest: {manifest_path}"
                )
            with open(manifest_path, "r", encoding="utf-8") as handle:
                cached_manifest = json.load(handle)
            expected = {
                "preset": preset,
                "checkpoint": osp.abspath(args.checkpoint),
                "horizon_root": osp.abspath(args.horizon_root),
                "source_noloop_pose": osp.abspath(base_pose_path),
            }
            mismatches = {
                key: (cached_manifest.get(key), value)
                for key, value in expected.items()
                if cached_manifest.get(key) != value
            }
            cached_loop_cfg = cached_manifest.get("loop_config", {}) or {}
            for key in ("salad_ckpt_path", "salad_dino_weights_path"):
                expected_value = osp.abspath(str(loop_cfg[key]))
                if cached_loop_cfg.get(key) != expected_value:
                    mismatches[key] = (cached_loop_cfg.get(key), expected_value)
            if mismatches:
                raise RuntimeError(
                    f"Horizon loop cache provenance mismatch for {seq}: {mismatches}"
                )
            loop_poses = validate_predicted_poses(
                torch.from_numpy(np.load(loop_pose_path)), len(filelist)
            )
        else:
            loop_poses, metadata = run_horizon_loop_from_c2w(
                filelist,
                base_poses,
                model,
                cfg,
                loop_cfg,
                osp.join(seq_dir, "loop_artifacts"),
            )
            loop_poses = validate_predicted_poses(loop_poses, len(filelist))
            np.save(loop_pose_path, loop_poses.numpy())
            save_list_of_matrices(
                loop_poses.numpy().tolist(), osp.join(seq_dir, "pred_poses.json")
            )
            metadata.update(
                dataset=dataset_name,
                sequence=seq,
                preset=preset,
                checkpoint=osp.abspath(args.checkpoint),
                horizon_root=osp.abspath(args.horizon_root),
                source_noloop_pose=osp.abspath(base_pose_path),
            )
            with open(manifest_path, "w") as handle:
                json.dump(metadata, handle, indent=2)

        pred_traj = get_tum_poses(loop_poses)
        gt_traj = load_gt(
            dataset_info, dataset_name, seq, stride, args.max_eval_frames
        )
        if trajectory_length(gt_traj) != trajectory_length(pred_traj):
            raise ValueError(
                f"GT/loop pose mismatch for {dataset_name}/{seq}: "
                f"gt={trajectory_length(gt_traj)} pred={trajectory_length(pred_traj)}"
            )
        save_tum_poses(pred_traj, osp.join(seq_dir, "pred_traj.txt"), verbose=False)
        save_tum_poses(gt_traj, osp.join(seq_dir, "gt_traj.txt"), verbose=False)
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj,
            seq=seq,
            filename=osp.join(seq_dir, "eval_metric.txt"),
            verbose=False,
        )
        plot_trajectory(
            pred_traj,
            gt_traj,
            title=f"{seq} Horizon + Loop",
            filename=osp.join(seq_dir, "vis.png"),
            verbose=False,
        )
        metrics = {
            "model": "horizonstream_loop_salad",
            "dataset": dataset_name,
            "seq": seq,
            "ATE": ate,
            "RPE trans": rpe_trans,
            "RPE rot": rpe_rot,
        }
        write_csv(seq_metrics_path, metrics)
        results.append((seq, ate, rpe_trans, rpe_rot))

    if len(results) != len(sequences):
        raise RuntimeError(
            f"Incomplete Horizon loop evaluation: {len(results)}/{len(sequences)}"
        )
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
    aggregate = {
        "model": "horizonstream_loop_salad",
        "dataset": dataset_name,
        "loop_preset": preset,
        "ATE": avg_ate,
        "RPE trans": avg_rpe_trans,
        "RPE rot": avg_rpe_rot,
    }
    write_csv(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
