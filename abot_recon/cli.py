from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .api import ABotRecon
from .checkpoint import DEFAULT_MODEL_ID
from .preprocessing import preprocess_image


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
    parser.add_argument(
        "--save-local-points", action=argparse.BooleanOptionalAction, default=True,
        help="save per-frame local point maps (default: enabled)",
    )
    parser.add_argument(
        "--save-world-points", action=argparse.BooleanOptionalAction, default=False,
        help="save point maps transformed by the final trajectory (default: disabled)",
    )
    parser.add_argument(
        "--save-confidence", action=argparse.BooleanOptionalAction, default=True,
        help="save per-pixel confidence maps (default: enabled)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="mask points below this confidence in [0,1] (default: 0, no filtering)",
    )
    parser.add_argument("--save-points", action="store_true", help=argparse.SUPPRESS)
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


def _save_dense_colors(output_dir: Path, paths: list[Path], dense_indices) -> None:
    indices = range(len(paths)) if dense_indices is None else dense_indices
    colors = []
    for index in indices:
        with Image.open(paths[index]) as image:
            tensor, _ = preprocess_image(image)
        colors.append(
            (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        )
    torch.save(torch.stack(colors), output_dir / "colors.pt")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = _collect_images(args.image_dir, args.start, args.end, args.stride)
    if args.dense_stride <= 0:
        raise ValueError("dense-stride must be positive")
    save_local_points = args.save_local_points or args.save_points
    save_world_points = args.save_world_points or args.save_points
    model = ABotRecon.from_pretrained(
        args.checkpoint,
        device=args.device,
        amp_dtype=args.amp_dtype,
        attention_backend=args.attention_backend,
        max_frames=args.max_frames,
        output_local_points=save_local_points,
        output_world_points=save_world_points,
        output_confidence=args.save_confidence,
        confidence_threshold=args.confidence_threshold,
        loop_closure=args.loop_closure,
        loop_salad_checkpoint=args.loop_salad_checkpoint,
        loop_dino_checkpoint=args.loop_dino_checkpoint,
        loop_output_dir=args.loop_output_dir,
    )
    dense_indices = None
    if args.dense_stride > 1 and (
        save_local_points or save_world_points or args.save_confidence
    ):
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
    if result.confidence_mask is not None:
        torch.save(result.confidence_mask.cpu(), args.output_dir / "confidence_mask.pt")
    if result.local_points is not None or result.world_points is not None:
        _save_dense_colors(args.output_dir, paths, dense_indices)


if __name__ == "__main__":
    main()
