#!/usr/bin/env python3
"""Pose-only inference on a custom image sequence without ground truth."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import random
import re
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from PIL import Image

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from relpose.evo_utils import get_tum_poses, plot_trajectory, save_tum_poses
from relpose.long_pose_protocol import validate_predicted_poses
from utils.messages import save_list_of_matrices


METHOD_TO_MODEL = {
    "lingbot": "lingbot_map",
    "horizon": "horizonstream",
    "abot_recon": "abot_recon",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def list_images(
    image_dir: str,
    stride: int,
    max_frames: int,
    skip_invalid: bool,
    start_frame: int = 0,
    end_frame: int = -1,
) -> tuple[list[str], list[str]]:
    paths = sorted(
        (p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if end_frame >= 0 and end_frame < start_frame:
        raise ValueError("--end-frame must be >= --start-frame")
    # Frame bounds refer to the naturally sorted source sequence and are
    # inclusive. Stride is applied inside that selected source-frame range.
    stop = None if end_frame < 0 else end_frame + 1
    paths = paths[start_frame:stop:stride]
    if max_frames > 0:
        paths = paths[:max_frames]
    skipped = []
    if skip_invalid:
        valid = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                valid.append(path)
            except (OSError, ValueError) as exc:
                print(f"[custom-pose] skipping unreadable image: {path}: {exc}", flush=True)
                skipped.append(path)
        paths = valid
    if not paths:
        raise FileNotFoundError(f"No supported images found under {image_dir}")
    return [str(p.resolve()) for p in paths], [str(p.resolve()) for p in skipped]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=sorted(METHOD_TO_MODEL), required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sequence-name", default="wangjing_long_stride5")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=-1)
    p.add_argument("--save-reconstruction", action="store_true")
    p.add_argument("--save-point-maps", action="store_true")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--point-pixel-stride", type=int, default=2)
    p.add_argument("--point-depth-max", type=float, default=40.0)
    p.add_argument("--skip-invalid-images", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume-existing", action="store_true")
    p.add_argument("--loop-mode", choices=["auto", "off", "both"], default="auto")
    p.add_argument("--attention-backend", choices=["auto", "paged", "sdpa"], default="auto")
    p.add_argument("--lingbot-root", default="third_party/LingBot-Map")
    p.add_argument("--horizon-root", default="third_party/HorizonStream")
    p.add_argument(
        "--salad-ckpt",
        default="checkpoints/loop/dino_salad.ckpt",
    )
    p.add_argument(
        "--salad-dino-weights",
        default="checkpoints/loop/dinov2_vitb14_pretrain.pth",
    )
    p.add_argument("--loop-preset", choices=["generic", "kitti", "vbr"], default="generic")
    return p.parse_args()


def load_config():
    config_dir = osp.join(str(root), "configs")
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        return compose(config_name="eval", overrides=["evaluation=relpose_stride1"])


def configure_model(cfg, args):
    key = METHOD_TO_MODEL[args.method]
    info = cfg.model[key]
    if args.method == "lingbot":
        info.cfg.pretrained_model_name_or_path = osp.abspath(args.checkpoint)
        info.cfg.source_root = osp.abspath(args.lingbot_root)
        # The benchmark-only profile is defined only for KITTI/VBR/Oxford.
        # For arbitrary imagery, use LingBot's released aspect-preserving
        # area-budget preprocessing.
        info.cfg.preprocess_mode = "area_budget"
        info.cfg.max_frame_num = 22000
    elif args.method == "horizon":
        info.cfg.checkpoint = osp.abspath(args.checkpoint)
        info.cfg.horizonstream_root = osp.abspath(args.horizon_root)
        info.cfg.amp_dtype = "fp16"
    else:
        info.cfg.ckpt = osp.abspath(args.checkpoint)
        info.cfg.device = args.device
        info.cfg.attention_backend = args.attention_backend
        info.cfg.num_frames_cap = 22000
        info.cfg.max_frames = 22000
    return key, info


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def save_reconstruction_outputs(paths, poses, local, conf, pad, model, out_dir, args):
    from mv_recon.fov_preprocess import resize_width_crop_or_pad_mean, strip_vertical_pad_hwc

    conf_dir = out_dir / "confidence"
    map_dir = out_dir / "point_maps"
    conf_dir.mkdir(parents=True, exist_ok=True)
    if args.save_point_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    local = strip_vertical_pad_hwc(local, pad).numpy()
    conf = strip_vertical_pad_hwc(conf[..., None], pad)[..., 0].numpy()
    poses_np = poses.numpy().astype(np.float32)
    all_xyz, all_rgb = [], []
    pixel_stride = max(int(args.point_pixel_stride), 1)
    for index, path in enumerate(paths):
        np.save(conf_dir / f"frame_{index:06d}.npy", conf[index].astype(np.float16))
        with Image.open(path) as image:
            prepared = resize_width_crop_or_pad_mean(
                image.convert("RGB"),
                target_h=int(model.height),
                target_w=int(model.width),
                pad_rgb=getattr(model, "fov_pad_rgb", (0.485, 0.456, 0.406)),
            )
        rgb = prepared.image.permute(1, 2, 0)
        rgb = strip_vertical_pad_hwc(rgb, pad).numpy()
        rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)

        local_i = local[index]
        rotation = poses_np[index, :3, :3]
        translation = poses_np[index, :3, 3]
        world_i = local_i @ rotation.T + translation
        geometry_valid = (
            np.isfinite(world_i).all(axis=-1)
            & np.isfinite(local_i).all(axis=-1)
            & (local_i[..., 2] > 1e-4)
            & (local_i[..., 2] <= float(args.point_depth_max))
        )
        confidence_valid = conf[index] >= float(args.confidence_threshold)
        sampled = np.zeros_like(geometry_valid)
        sampled[::pixel_stride, ::pixel_stride] = True
        keep = geometry_valid & confidence_valid & sampled
        all_xyz.append(world_i[keep].astype(np.float32))
        all_rgb.append(rgb[keep])
        if args.save_point_maps:
            np.savez_compressed(
                map_dir / f"frame_{index:06d}.npz",
                local_points=local_i.astype(np.float16),
                world_points=world_i.astype(np.float32),
                confidence=conf[index].astype(np.float16),
                rgb=rgb,
                valid_mask=geometry_valid,
            )

    xyz = np.concatenate(all_xyz, axis=0) if all_xyz else np.empty((0, 3), np.float32)
    rgb = np.concatenate(all_rgb, axis=0) if all_rgb else np.empty((0, 3), np.uint8)
    write_binary_ply(out_dir / "fused_confident_points.ply", xyz, rgb)
    metadata = {
        "confidence_activation": "sigmoid",
        "confidence_threshold": float(args.confidence_threshold),
        "point_depth_max_m": float(args.point_depth_max),
        "point_pixel_stride": pixel_stride,
        "num_exported_points": int(len(xyz)),
        "per_frame_point_maps_saved": bool(args.save_point_maps),
    }
    (out_dir / "reconstruction_manifest.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def save_pose_outputs(poses: torch.Tensor, out_dir: Path, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = poses.detach().float().cpu()
    np.save(out_dir / "pred_poses.npy", poses.numpy())
    save_list_of_matrices(poses.numpy().tolist(), str(out_dir / "pred_poses.json"))
    traj = get_tum_poses(poses)
    save_tum_poses(traj, str(out_dir / "pred_traj.txt"), verbose=False)
    plot_trajectory(
        traj,
        None,
        title=title,
        filename=str(out_dir / "trajectory.png"),
        verbose=False,
    )


def load_horizon_loop_cfg(args):
    cfg_path = osp.join(args.horizon_root, "configs", "horizonstream_infer.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        official = yaml.safe_load(f) or {}
    loop_cfg = dict(official.get("online_loop", {}) or {})
    if args.loop_preset != "generic":
        groups = official.get("online_loop_groups", {}) or {}
        loop_cfg.update(groups[args.loop_preset] or {})
    loop_cfg.update(
        methods=["salad"],
        salad_ckpt_path=osp.abspath(args.salad_ckpt),
        salad_dino_weights_path=osp.abspath(args.salad_dino_weights),
    )
    return loop_cfg


def run_horizon_loop(paths, base_poses, model, cfg, args, loop_dir):
    from interfaces.horizonstream import run_horizon_loop_from_c2w

    poses, metadata = run_horizon_loop_from_c2w(
        paths,
        base_poses,
        model,
        cfg,
        load_horizon_loop_cfg(args),
        str(loop_dir / "loop_artifacts"),
    )
    with open(loop_dir / "loop_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return poses


def run_abot_recon_loop(paths, base_poses, model, cfg, args, loop_dir):
    from interfaces.abot_recon import run_abot_recon_loop_from_c2w

    with open_dict(cfg):
        cfg.output_dir = str(loop_dir)
        cfg.abot_recon_loop_salad_ckpt = osp.abspath(args.salad_ckpt)
        cfg.abot_recon_loop_dino_weights = osp.abspath(args.salad_dino_weights)
    return run_abot_recon_loop_from_c2w(paths, base_poses, model, cfg)


def main():
    args = parse_args()
    if args.loop_mode == "auto":
        args.loop_mode = "both" if args.method == "abot_recon" else "off"
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.loop_mode == "both" and args.method == "lingbot":
        raise ValueError("LingBot has no loop backend in this custom runner")
    if args.save_reconstruction and args.method != "abot_recon":
        raise ValueError("--save-reconstruction is currently supported for abot_recon only")
    if not osp.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if args.loop_mode == "both":
        for path in (args.salad_ckpt, args.salad_dino_weights):
            if not osp.isfile(path):
                raise FileNotFoundError(path)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    paths, skipped_paths = list_images(
        args.image_dir,
        args.stride,
        args.max_frames,
        args.skip_invalid_images,
        args.start_frame,
        args.end_frame,
    )
    print(
        f"[custom-pose] method={args.method}, images={len(paths)}, "
        f"input_stride={args.stride}, loop_mode={args.loop_mode}",
        flush=True,
    )
    out_root = Path(args.output_root).resolve() / args.method / args.sequence_name
    raw_dir = out_root / "noloop"
    loop_dir = out_root / "loop"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "image_paths.txt").write_text("\n".join(paths) + "\n")
    if skipped_paths:
        (out_root / "skipped_image_paths.txt").write_text("\n".join(skipped_paths) + "\n")

    cfg = load_config()
    cfg.device = args.device
    key, info = configure_model(cfg, args)
    infer_cfg = info.get(
        "infer_cameras_c2w",
        {"_target_": f"interfaces.{key}.infer_cameras_c2w", "_partial_": True},
    )
    infer = hydra.utils.instantiate(infer_cfg)
    print(f"[custom-pose] loading {args.method} checkpoint", flush=True)
    model = hydra.utils.instantiate(info.cfg).to(args.device).eval()
    print(f"[custom-pose] model ready on {args.device}", flush=True)

    pose_cache = raw_dir / "pred_poses.npy"
    reconstruction = None
    reconstruction_manifest = raw_dir / "reconstruction_manifest.json"
    can_resume = args.resume_existing and pose_cache.is_file()
    if args.save_reconstruction and not reconstruction_manifest.is_file():
        can_resume = False
    if can_resume:
        base_poses = validate_predicted_poses(torch.from_numpy(np.load(pose_cache)), len(paths))
        if args.save_reconstruction:
            reconstruction = json.loads(reconstruction_manifest.read_text())
    else:
        print("[custom-pose] starting no-loop pose inference", flush=True)
        if args.save_reconstruction:
            from interfaces.abot_recon import infer_custom_reconstruction

            base_poses, local, conf, pad = infer_custom_reconstruction(paths, model, cfg)
            reconstruction = save_reconstruction_outputs(
                paths, base_poses, local, conf, pad, model, raw_dir, args
            )
        else:
            base_poses, _ = infer(paths, model, cfg)
        base_poses = validate_predicted_poses(base_poses, len(paths))
        save_pose_outputs(base_poses, raw_dir, f"{args.sequence_name}: {args.method}")
        print(f"[custom-pose] no-loop outputs saved to {raw_dir}", flush=True)

    run_info = {
        "method": args.method,
        "sequence": args.sequence_name,
        "image_dir": osp.abspath(args.image_dir),
        "num_images": len(paths),
        "input_stride": args.stride,
        "source_start_frame": args.start_frame,
        "source_end_frame_inclusive": args.end_frame,
        "checkpoint": osp.abspath(args.checkpoint),
        "ground_truth": None,
        "outputs": ["noloop"],
    }
    if args.save_reconstruction:
        run_info["reconstruction"] = reconstruction
    if args.loop_mode == "both":
        loop_dir.mkdir(parents=True, exist_ok=True)
        loop_cache = loop_dir / "pred_poses.npy"
        if args.resume_existing and loop_cache.is_file():
            print(f"[custom-pose] reusing loop outputs from {loop_cache}", flush=True)
            loop_poses = validate_predicted_poses(torch.from_numpy(np.load(loop_cache)), len(paths))
        else:
            print("[custom-pose] starting loop-closure backend", flush=True)
            if args.method == "horizon":
                loop_poses = run_horizon_loop(paths, base_poses, model, cfg, args, loop_dir)
            else:
                loop_poses = run_abot_recon_loop(paths, base_poses, model, cfg, args, loop_dir)
            loop_poses = validate_predicted_poses(loop_poses, len(paths))
            save_pose_outputs(loop_poses, loop_dir, f"{args.sequence_name}: {args.method} w/ LC")
            print(f"[custom-pose] loop outputs saved to {loop_dir}", flush=True)
        run_info["outputs"].append("loop")

    with open(out_root / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
