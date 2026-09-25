from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _camera(width: int, height: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
    focal = 0.8 * max(width, height)
    intrinsics = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 0.03 * frame
    return intrinsics, pose


def _image(width: int, height: int, frame: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None].repeat(height, axis=0)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None].repeat(width, axis=1)
    return np.stack((np.roll(x, frame * 3, axis=1), y, 255 - x), axis=-1)


def make_dl3dv(root: Path, frames: int, width: int, height: int) -> None:
    dense = root / "demo_bucket" / "demo_scene" / "dense"
    for name in ("rgb", "depth", "cam", "sky_mask", "outlier_mask"):
        (dense / name).mkdir(parents=True, exist_ok=True)
    (root / "dl3dv_geometry_blacklist_20260810.txt").write_text("", encoding="utf-8")
    for frame in range(frames):
        stem = f"{frame:05d}"
        intrinsics, pose = _camera(width, height, frame)
        Image.fromarray(_image(width, height, frame)).save(dense / "rgb" / f"{stem}.png")
        Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(
            dense / "sky_mask" / f"{stem}.png"
        )
        Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(
            dense / "outlier_mask" / f"{stem}.png"
        )
        np.save(dense / "depth" / f"{stem}.npy", np.full((height, width), 2.0, np.float32))
        np.savez(dense / "cam" / f"{stem}.npz", intrinsic=intrinsics, pose=pose)


def make_tartanground(root: Path, frames: int, width: int, height: int) -> None:
    sequence = root / "DemoEnv__omni__P0001__rcam_front"
    for name in ("images", "depths", "cameras", "masks"):
        (sequence / name).mkdir(parents=True, exist_ok=True)
    for frame in range(frames):
        stem = f"{frame:05d}"
        intrinsics, pose = _camera(width, height, frame)
        Image.fromarray(_image(width, height, frame)).save(
            sequence / "images" / f"{stem}.jpg", quality=95
        )
        np.save(sequence / "depths" / f"{stem}.npy", np.full((height, width), 2.0, np.float32))
        np.save(sequence / "masks" / f"{stem}.npy", np.zeros((height, width), dtype=bool))
        np.savez(
            sequence / "cameras" / f"{stem}.npz",
            camera_intrinsics=intrinsics,
            camera_pose=pose,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny processed training datasets")
    parser.add_argument("--output", type=Path, default=Path("examples/training_data"))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--height", type=int, default=280)
    args = parser.parse_args()
    if args.frames < 2 or args.width < 14 or args.height < 14:
        parser.error("frames must be >=2 and image dimensions must be >=14")
    make_dl3dv(args.output / "dl3dv", args.frames, args.width, args.height)
    make_tartanground(args.output / "tartanground", args.frames, args.width, args.height)
    print(f"Training samples written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
