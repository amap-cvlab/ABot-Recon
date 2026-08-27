from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .api import ABotRecon
from .checkpoint import DEFAULT_MODEL_ID


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ABot-Recon streaming inference")
    parser.add_argument("--image-dir", type=Path, default=Path("examples/images"))
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_MODEL_ID,
        help="local checkpoint path or Hugging Face repo ID",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--attention-backend", choices=("auto", "paged", "sdpa"), default="auto")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-frames", type=int, default=22_000)
    parser.add_argument("--dense-stride", type=int, default=1)
    parser.add_argument("--save-points", action="store_true")
    parser.add_argument("--save-confidence", action="store_true")
    parser.add_argument(
        "--loop-closure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable sparse loop closure (default: enabled)",
    )
    parser.add_argument(
        "--loop-salad-checkpoint",
        type=Path,
        default=Path("checkpoints/loop/dino_salad.ckpt"),
    )
    parser.add_argument(
        "--loop-dino-checkpoint",
        type=Path,
        default=Path("checkpoints/loop/dinov2_vitb14_pretrain.pth"),
    )
    parser.add_argument("--loop-output-dir", type=Path, default=Path("outputs/loop"))
    return parser


def _collect_images(directory: Path, start: int, end: int | None, stride: int):
    if stride <= 0:
        raise ValueError("stride must be positive")
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return paths[start:end:stride]


def _save_pose_outputs(output_dir: Path, result) -> None:
    np.save(output_dir / "camera_poses.npy", result.camera_poses.cpu().numpy())
    np.save(output_dir / "relative_poses.npy", result.relative_poses.cpu().numpy())
    np.save(output_dir / "camera_poses_noloop.npy", result.camera_poses_noloop.cpu().numpy())
    np.save(output_dir / "relative_poses_noloop.npy", result.relative_poses_noloop.cpu().numpy())
    if result.camera_poses_loop is not None:
        np.save(output_dir / "camera_poses_loop.npy", result.camera_poses_loop.cpu().numpy())
        np.save(output_dir / "relative_poses_loop.npy", result.relative_poses_loop.cpu().numpy())


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = _collect_images(args.image_dir, args.start, args.end, args.stride)
    if args.dense_stride <= 0:
        raise ValueError("dense-stride must be positive")
    model = ABotRecon.from_pretrained(
        args.checkpoint,
        device=args.device,
        amp_dtype=args.amp_dtype,
        attention_backend=args.attention_backend,
        max_frames=args.max_frames,
        output_points=args.save_points,
        output_confidence=args.save_confidence,
        loop_closure=args.loop_closure,
        loop_salad_checkpoint=args.loop_salad_checkpoint,
        loop_dino_checkpoint=args.loop_dino_checkpoint,
        loop_output_dir=args.loop_output_dir,
    )
    dense_indices = None
    if args.dense_stride > 1 and (args.save_points or args.save_confidence):
        dense_indices = range(0, len(paths), args.dense_stride)
    result = model.infer(paths, dense_output_indices=dense_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_pose_outputs(args.output_dir, result)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)
    if result.local_points is not None:
        torch.save(result.local_points.cpu(), args.output_dir / "local_points.pt")
    if result.world_points is not None:
        torch.save(result.world_points.cpu(), args.output_dir / "world_points.pt")
    if result.confidence is not None:
        torch.save(result.confidence.cpu(), args.output_dir / "confidence.pt")


if __name__ == "__main__":
    main()
