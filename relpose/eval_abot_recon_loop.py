#!/usr/bin/env python3
"""Evaluate ABot-Recon loop closure from cached no-loop trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import random

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from interfaces.abot_recon import (
    extract_abot_recon_loop_descriptors,
    run_abot_recon_loop_from_c2w,
)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=["auto", "paged", "sdpa"], default="paged")
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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gt(dataset_info, seq, stride, max_frames=0):
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
    cfg.device = args.device
    cfg.abot_recon_loop_enabled = False
    cfg.abot_recon_loop_salad_ckpt = osp.abspath(args.salad_ckpt)
    cfg.abot_recon_loop_dino_weights = osp.abspath(args.salad_dino_weights)

    model_info = cfg.model.abot_recon
    model_info.cfg.ckpt = args.checkpoint
    model_info.cfg.attention_backend = args.attention_backend
    model = hydra.utils.instantiate(model_info.cfg).to(args.device).eval()

    output_root = osp.join(args.output_root, dataset_name)
    os.makedirs(output_root, exist_ok=True)
    seq_metrics_path = osp.join(output_root, "seq_metrics.csv")
    aggregate_path = osp.join(output_root, "aggregate_metrics.csv")

    all_sequences = get_all_sequences(dataset_info)
    sequences = all_sequences
    if args.sequence:
        if args.sequence not in all_sequences:
            raise ValueError(f"Unknown sequence for {dataset_name}: {args.sequence}")
        sequences = [args.sequence]

    results = []
    for seq in sequences:
        filelist = list_imgs_a_sequence(dataset_info, seq)[::stride]
        if args.max_eval_frames > 0:
            filelist = filelist[: args.max_eval_frames]

        base_seq_dir = osp.join(args.base_output_root, dataset_name, seq)
        base_pose_path = osp.join(base_seq_dir, "pred_poses.npy")
        if not osp.isfile(base_pose_path):
            raise FileNotFoundError(
                f"Missing ABot-Recon no-loop pose cache: {base_pose_path}"
            )
        base_digest = file_sha256(base_pose_path)
        base_poses = validate_predicted_poses(
            torch.from_numpy(np.load(base_pose_path)), len(filelist)
        )

        seq_dir = osp.join(output_root, seq)
        os.makedirs(seq_dir, exist_ok=True)
        loop_pose_path = osp.join(seq_dir, "pred_poses.npy")
        manifest_path = osp.join(seq_dir, "runtime_manifest.json")
        use_cache = args.resume_existing == "true" and osp.isfile(loop_pose_path)
        expected = {
            "checkpoint": osp.abspath(args.checkpoint),
            "attention_backend": args.attention_backend,
            "source_noloop_pose": osp.abspath(base_pose_path),
            "source_noloop_sha256": base_digest,
            "salad_checkpoint": osp.abspath(args.salad_ckpt),
            "dino_checkpoint": osp.abspath(args.salad_dino_weights),
        }
        if use_cache:
            if not osp.isfile(manifest_path):
                raise RuntimeError(
                    f"Loop pose cache has no runtime manifest: {manifest_path}"
                )
            with open(manifest_path, "r", encoding="utf-8") as handle:
                cached_manifest = json.load(handle)
            mismatches = {
                key: (cached_manifest.get(key), value)
                for key, value in expected.items()
                if cached_manifest.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"ABot-Recon loop cache provenance mismatch for {seq}: {mismatches}"
                )
            loop_poses = validate_predicted_poses(
                torch.from_numpy(np.load(loop_pose_path)), len(filelist)
            )
        else:
            descriptors, descriptor_stats = extract_abot_recon_loop_descriptors(
                filelist, model, cfg
            )
            cfg.output_dir = seq_dir
            loop_poses, metadata = run_abot_recon_loop_from_c2w(
                filelist,
                base_poses,
                model,
                cfg,
                descriptors=descriptors,
                descriptor_stats=descriptor_stats,
                return_metadata=True,
            )
            loop_poses = validate_predicted_poses(loop_poses, len(filelist))
            np.save(loop_pose_path, loop_poses.numpy())
            save_list_of_matrices(
                loop_poses.numpy().tolist(), osp.join(seq_dir, "pred_poses.json")
            )
            metadata.update(dataset=dataset_name, sequence=seq, **expected)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

        pred_traj = get_tum_poses(loop_poses)
        gt_traj = load_gt(dataset_info, seq, stride, args.max_eval_frames)
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
            title=f"{seq} ABot-Recon + Loop",
            filename=osp.join(seq_dir, "vis_traj_error.png"),
            verbose=False,
        )
        metrics = {
            "model": "abot_recon_loop",
            "dataset": dataset_name,
            "seq": seq,
            "ATE": ate,
            "RPE trans": rpe_trans,
            "RPE rot": rpe_rot,
        }
        write_csv(
            seq_metrics_path,
            metrics,
            key_fields=("model", "dataset", "seq"),
        )
        results.append((seq, ate, rpe_trans, rpe_rot))

    if len(results) != len(sequences):
        raise RuntimeError(
            f"Incomplete ABot-Recon loop evaluation: {len(results)}/{len(sequences)}"
        )
    if sequences == all_sequences:
        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
        aggregate = {
            "model": "abot_recon_loop",
            "dataset": dataset_name,
            "ATE": avg_ate,
            "RPE trans": avg_rpe_trans,
            "RPE rot": avg_rpe_rot,
        }
        write_csv(
            aggregate_path,
            aggregate,
            key_fields=("model", "dataset"),
        )
        print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
