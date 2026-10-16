"""Minimal, readable ABot-Recon inference example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from abot_recon import ABotRecon
from abot_recon.preprocessing import preprocess_image


MODEL_ID = "acvlab/ABot-Recon"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABot-Recon on an RGB video")
    parser.add_argument("--image-dir", type=Path, default=Path("examples/images"))
    parser.add_argument(
        "--checkpoint",
        default=MODEL_ID,
        help="Hugging Face repository ID or local checkpoint path",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("auto", "paged", "sdpa"), default="auto")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--dense-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=22_000)
    parser.add_argument(
        "--loop-closure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable optional loop-closure refinement (default: enabled)",
    )
    parser.add_argument(
        "--save-local-points",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-world-points",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
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
    return parser.parse_args()


def collect_images(directory: Path, start: int, end: int | None, stride: int) -> list[Path]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    images = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    selected = images[start:end:stride]
    if not selected:
        raise ValueError(f"No input images selected from {directory}")
    return selected


def save_result(output_dir: Path, result, images: list[Path], dense_indices) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "camera_poses.npy", result.camera_poses.cpu().numpy())
    np.save(output_dir / "relative_poses.npy", result.relative_poses.cpu().numpy())
    np.save(output_dir / "camera_poses_noloop.npy", result.camera_poses_noloop.cpu().numpy())
    np.save(output_dir / "relative_poses_noloop.npy", result.relative_poses_noloop.cpu().numpy())
    if result.camera_poses_loop is not None:
        np.save(output_dir / "camera_poses_loop.npy", result.camera_poses_loop.cpu().numpy())
        np.save(output_dir / "relative_poses_loop.npy", result.relative_poses_loop.cpu().numpy())
    if result.local_points is not None:
        torch.save(result.local_points.cpu(), output_dir / "local_points.pt")
    if result.world_points is not None:
        torch.save(result.world_points.cpu(), output_dir / "world_points.pt")
    if result.confidence is not None:
        torch.save(result.confidence.cpu(), output_dir / "confidence.pt")
    if result.confidence_mask is not None:
        torch.save(result.confidence_mask.cpu(), output_dir / "confidence_mask.pt")
    if result.local_points is not None or result.world_points is not None:
        indices = range(len(images)) if dense_indices is None else dense_indices
        colors = []
        for index in indices:
            with Image.open(images[index]) as image:
                tensor, _ = preprocess_image(image)
            colors.append(
                (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
            )
        torch.save(torch.stack(colors), output_dir / "colors.pt")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.dense_stride <= 0:
        raise ValueError("--dense-stride must be positive")
    images = collect_images(args.image_dir, args.start, args.end, args.stride)

    model = ABotRecon.from_pretrained(
        args.checkpoint,
        device=args.device,
        attention_backend=args.attention_backend,
        max_frames=args.max_frames,
        loop_closure=args.loop_closure,
        loop_salad_checkpoint=args.loop_salad_checkpoint,
        loop_dino_checkpoint=args.loop_dino_checkpoint,
        loop_output_dir=args.loop_output_dir,
    )

    dense_indices = None
    if args.save_local_points or args.save_world_points or args.save_confidence:
        dense_indices = range(0, len(images), args.dense_stride)
    result = model.infer(
        images,
        output_local_points=args.save_local_points,
        output_world_points=args.save_world_points,
        output_confidence=args.save_confidence,
        confidence_threshold=args.confidence_threshold,
        loop_closure=args.loop_closure,
        dense_output_indices=dense_indices,
    )
    save_result(args.output_dir, result, images, dense_indices)
    print(f"Processed {len(images)} frames; results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
