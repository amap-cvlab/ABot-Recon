#!/usr/bin/env python3
"""Offline visualization of published ABot-Recon outputs. No inference or pose refinement.

Coordinate conventions: saved poses are camera-to-world, local +Z is forward,
local +X is right, and local +Y is down. Depth limits use local Z, not distance
from the rendering camera. All distances are in reconstruction units, not meters.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import torch


# ------------------------------ Configuration ------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("Input and frame selection (indices refer to INFERENCE frames)")
    g.add_argument("--result-dir", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--poses", choices=("final", "noloop", "loop"), default="final")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--end", type=int, help="exclusive; defaults to all poses")
    g.add_argument("--ground-frames", type=int, default=600, help="maximum L; never repeat frames to fill")
    g.add_argument("--view-step", type=int, default=1, help="S: inference frames advanced per video frame")
    g.add_argument("--cloud-step", type=int, default=1, help="P: select saved dense frames by (pose_index-start) %% P == 0")
    g.add_argument("--global-scope", choices=("all", "segment"), default="all", help="bird images use all poses or [start,end)")
    g.add_argument("--image-dir", type=Path, help="original inference input images; optional in native mode")
    g.add_argument("--image-start", type=int, default=0, help="same start used by inference CLI")
    g.add_argument("--image-end", type=int, help="same exclusive end used by inference CLI")
    g.add_argument("--image-stride", type=int, default=1, help="same stride used by inference CLI")
    g.add_argument("--frame-list", type=Path, help="alternative: JSON list of exactly N ordered source image paths")
    g = p.add_argument_group("Geometry and filtering")
    g.add_argument("--resolution", choices=("native", "source"), default="native")
    g.add_argument("--upsample", choices=("bilinear", "nearest"), default="bilinear")
    g.add_argument("--pixel-step", type=int, default=1, help="sample both image axes; 2 keeps about 1/4")
    g.add_argument("--confidence", type=float, default=0.15)
    g.add_argument("--min-depth", type=float, default=0.0)
    g.add_argument("--max-depth", type=float, default=6.0, help="0 disables upper bound")
    g.add_argument("--depth-edge", type=float, default=0.15, help="local (max-min)/Z threshold; 0 disables")
    g.add_argument("--edge-kernel", type=int, default=3)
    g.add_argument("--sky", action=argparse.BooleanOptionalAction, default=True, help="use precomputed sky masks; missing/incompatible masks are an error")
    g.add_argument("--sky-mask-dir", type=Path, help="defaults to result-dir/sky_masks; generate with generate_sky_masks.py first")
    g.add_argument("--history", type=int, default=0, help="0=all, else retain the last N inference frames")
    g.add_argument("--prune-every", type=int, default=10, help="permanently discard off-frustum points every N video frames; 0=off")
    g.add_argument("--bird-cloud-step", type=int, default=5)
    g.add_argument("--bird-pixel-step", type=int, default=8)
    g = p.add_argument_group("Virtual cameras and appearance (never change reconstructed poses)")
    g.add_argument("--width", type=int, default=1920)
    g.add_argument("--height", type=int, default=1080)
    g.add_argument("--background", type=int, nargs=3, default=[32, 32, 32], metavar=("R", "G", "B"))
    g.add_argument("--point-size", type=int, default=2, help="exact square diameter in output pixels")
    g.add_argument("--ground-fov", type=float, default=60, help="vertical FoV in degrees, matching the Wangjing renderer")
    g.add_argument("--bird-fov", type=float, default=60, help="vertical FoV in degrees")
    g.add_argument("--bird-margin", type=float, default=1.12, help="margin around the full trajectory in bird view")
    g.add_argument("--follow-distance", type=float, default=0, help="0=auto, positive=fixed backward distance")
    g.add_argument("--distance-ratio", type=float, default=0.12)
    g.add_argument("--distance-min", type=float, default=0.6)
    g.add_argument("--distance-max", type=float, default=2.0)
    g.add_argument("--height-ratio", type=float, default=0.2, help="eye raised by ratio * follow distance")
    g.add_argument("--look-ahead", type=float, default=0.35, help="target ahead by ratio * follow distance")
    g.add_argument("--smooth-radius", type=int, default=6, help="center/forward centered smoothing radius; 6 gives 13 frames")
    g.add_argument("--smooth-frames", type=int, default=6, help="centered up-direction smoothing window")
    g.add_argument("--frustum-step", type=int, default=1, help="0 disables; otherwise every N inference poses")
    g.add_argument("--frustum-size", type=float, default=0.05, help="reconstruction-camera frustum Z length")
    g.add_argument("--trajectory-width", type=int, default=2)
    g.add_argument("--frustum-width", type=int, default=1)
    g.add_argument("--center-size", type=int, default=2, help="diameter; 0 disables camera-center dots")
    g = p.add_argument_group("Outputs and runtime")
    g.add_argument("--fps", type=float, default=30, help="output rate, independent of original video FPS")
    g.add_argument("--transition-frames", type=int, default=120, help="0 disables; freeze scene time, animate virtual camera only")
    g.add_argument("--outputs", nargs="+", choices=("video", "trajectory", "pointcloud", "rgb"), default=["video", "trajectory", "pointcloud"])
    g.add_argument("--device", default="cpu", help="rasterizer: cpu or cuda:N")
    g.add_argument("--chunk-points", type=int, default=200000)
    g.add_argument("--max-resident-points", type=int, default=20000000, help="fail instead of silently downsampling or exhausting memory")
    g.add_argument("--crf", type=int, default=20, help="H.264 CRF")
    g.add_argument("--preset", default="medium", choices=("ultrafast", "fast", "medium", "slow"))
    g.add_argument("--overwrite", action="store_true", help="overwrite this script's outputs, not the input results")
    return p


def validate_args(a):
    for name, value in vars(a).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in ("view_step", "cloud_step", "pixel_step", "image_stride", "bird_cloud_step", "bird_pixel_step", "width", "height", "point_size", "smooth_frames", "chunk_points", "max_resident_points", "trajectory_width", "frustum_width"):
        if getattr(a, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("start", "image_start", "ground_frames", "transition_frames", "history", "prune_every", "frustum_step", "center_size", "smooth_radius"):
        if getattr(a, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not 0 <= a.confidence <= 1:
        raise ValueError("confidence must be in [0,1]")
    if a.edge_kernel <= 0 or a.edge_kernel % 2 != 1 or a.depth_edge < 0:
        raise ValueError("edge-kernel must be positive odd; depth-edge must be nonnegative")
    if a.min_depth < 0 or (a.max_depth != 0 and a.max_depth <= a.min_depth):
        raise ValueError("invalid depth interval")
    if a.fps <= 0 or not 1 < a.ground_fov < 175 or not 1 < a.bird_fov < 175:
        raise ValueError("invalid FPS or FoV")
    if a.bird_margin < 1 or a.height_ratio < 0:
        raise ValueError("bird-margin must be >= 1 and height-ratio must be nonnegative")
    if not 0 < a.distance_min <= a.distance_max or a.follow_distance < 0 or a.distance_ratio <= 0 or a.frustum_size <= 0:
        raise ValueError("invalid camera distance or size")
    if not 0 <= a.crf <= 51 or any(not 0 <= c <= 255 for c in a.background):
        raise ValueError("invalid CRF or background")
    if a.image_dir and a.frame_list:
        raise ValueError("choose image-dir OR frame-list")
    if (a.resolution == "source" or "rgb" in a.outputs) and not (a.image_dir or a.frame_list):
        raise ValueError("source resolution and original RGB require source images")
    if a.sky:
        a.sky_mask_dir = a.sky_mask_dir or a.result_dir / "sky_masks"
        if not (a.sky_mask_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"Sky filtering requires precomputed masks at {a.sky_mask_dir}. Run visualization/generate_sky_masks.py first, provide --sky-mask-dir PATH, or explicitly use --no-sky.")
    if any(x in a.outputs for x in ("video", "rgb")):
        if a.ground_frames == 0 or a.width % 2 or a.height % 2:
            raise ValueError("video requires ground-frames > 0 and even output dimensions")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg executable is required for video output")
    if a.output_dir.resolve() == a.result_dir.resolve():
        raise ValueError("output-dir must differ from result-dir")
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.overwrite:
        raise FileExistsError("output-dir is not empty; use a fresh directory or --overwrite")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


# --------------------- Published result and source geometry -----------------

def read_tensor(path: Path) -> torch.Tensor:
    # Public CLI uses zip-format torch.save, so tensors can be file-mapped.
    # weights_only avoids executing arbitrary pickled application objects.
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}")
    return value


class ReconstructionReader:
    def __init__(self, a):
        self.root = a.result_dir
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        filename = {"final": "camera_poses.npy", "noloop": "camera_poses_noloop.npy", "loop": "camera_poses_loop.npy"}[a.poses]
        self.poses = np.load(self.root / filename, allow_pickle=False).astype(np.float64)
        self.final_poses = np.load(self.root / "camera_poses.npy", allow_pickle=False).astype(np.float64)
        n = len(self.poses)
        for poses in (self.poses, self.final_poses):
            if poses.shape != (n, 4, 4) or not n or not np.isfinite(poses).all():
                raise ValueError("poses must be finite [N,4,4]")
            rot = poses[:, :3, :3]
            if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-4) or not np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=0.02) or np.any(np.linalg.det(rot) < 0.98):
                raise ValueError("poses must be proper rigid C2W matrices, not reflections or Sim3")
        if self.metadata.get("frames") != n:
            raise ValueError("metadata.frames does not match pose count")
        self.local = (self.root / "local_points.pt").is_file()
        self.points = read_tensor(self.root / ("local_points.pt" if self.local else "world_points.pt"))
        if self.points.ndim != 4 or self.points.shape[-1] != 3 or not self.points.is_floating_point():
            raise ValueError("points must be floating [M,H,W,3]")
        self.m, self.h, self.w, _ = self.points.shape
        if not self.m or min(self.h, self.w) < 2:
            raise ValueError("empty or degenerate point maps")
        mapping = self.metadata.get("dense_output_indices")
        self.indices = np.arange(n) if mapping is None else np.asarray(mapping)
        if self.indices.dtype.kind not in "iu" or self.indices.shape != (self.m,) or np.any(np.diff(self.indices) <= 0) or self.indices[0] < 0 or self.indices[-1] >= n:
            raise ValueError("dense_output_indices must map every dense tensor row to a unique increasing pose index")
        self.colors = read_tensor(self.root / "colors.pt")
        if self.colors.shape != self.points.shape or self.colors.dtype != torch.uint8:
            raise ValueError("colors must be uint8 [M,H,W,3], matching points")
        conf_path = self.root / "confidence.pt"
        self.confidence = read_tensor(conf_path) if conf_path.exists() else None
        if a.confidence > 0 and self.confidence is None:
            raise ValueError("confidence.pt missing; explicitly use --confidence 0 to disable filtering")
        if self.confidence is not None and self.confidence.shape != self.points.shape[:-1]:
            raise ValueError("confidence shape must be [M,H,W]")
        if self.metadata.get("confidence_threshold", 0) > a.confidence:
            warnings.warn("Inference already discarded low-confidence geometry; lowering the rendering threshold cannot restore it.")
        self.paths = None
        if a.frame_list:
            listed = json.loads(a.frame_list.read_text())
            if not isinstance(listed, list) or not all(isinstance(p, str) for p in listed):
                raise ValueError("frame-list must be a JSON list of paths")
            self.paths = [Path(p) if Path(p).is_absolute() else a.frame_list.parent / p for p in listed]
        elif a.image_dir:
            suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.paths = sorted(p for p in a.image_dir.iterdir() if p.suffix.lower() in suffixes)[a.image_start:a.image_end:a.image_stride]
        self.source_hw = None
        if self.paths is not None:
            if len(self.paths) != n or not all(p.is_file() for p in self.paths):
                raise ValueError("source selection must contain exactly N existing images in inference order")
            with Image.open(self.paths[0]) as im:
                self.source_hw = (im.height, im.width)
            # Verify a few associations against the actual published preprocessor.
            # A matching count alone cannot catch an incorrect image-start/stride.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from abot_recon.preprocessing import preprocess_image
            for j in sorted({0, self.m // 2, self.m - 1}):
                with Image.open(self.paths[int(self.indices[j])]) as im:
                    expected, _ = preprocess_image(im, height=self.h, width=self.w)
                expected = (expected.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
                error = np.abs(expected.astype(float) - self.colors[j].numpy()).mean()
                if error > 1.0:
                    raise ValueError(f"Source image does not match colors.pt at dense row {j} (MAE {error:.2f}); check source selection")
        else:
            warnings.warn("No source images: original padding cannot be identified. Native points/colors are used as saved.")
        self.geometry = None if self.source_hw is None else source_geometry(self.source_hw, (self.h, self.w), maps=a.resolution == "source")

    def local_points(self, row):
        points = self.points[row].float().numpy()
        if not self.local:
            # world_points.pt was generated using FINAL poses, even when the user
            # selects raw poses for display. Undo that transform exactly once.
            pose = self.final_poses[self.indices[row]]
            points = (points - pose[:3, 3]) @ pose[:3, :3]
        return points.astype(np.float32, copy=False)

    def rgb(self, pose_index):
        with Image.open(self.paths[pose_index]) as im:
            rgb = np.asarray(im.convert("RGB"))
        if rgb.shape[:2] != self.source_hw:
            raise ValueError("source geometry changes within the sequence")
        return rgb


def source_geometry(source_hw, native_hw, *, maps=True):
    """Pixel-center mapping matching preprocessing.py's rounded width-lock resize.

    Crop/pad changes the Y origin. X and Y scale can differ slightly because
    resized_height is rounded. Source areas removed by the crop stay invalid.
    """
    sh, sw = source_hw
    h, w = native_hw
    rh = max(1, round(sh * w / sw))
    crop = round((rh - h) * 0.5) if rh > h else 0
    pad = (h - rh) // 2 if rh < h else 0
    mx = my = None
    if maps:
        yy, xx = np.mgrid[:sh, :sw].astype(np.float32)
        mx = (xx + 0.5) * (w / sw) - 0.5
        my = (yy + 0.5) * (rh / sh) - 0.5 - crop + pad
    valid = np.zeros((h, w), bool)
    valid[pad:pad + min(rh, h)] = True
    return mx, my, valid, rh, crop, pad


def native_depth_valid(points, confidence, a):
    z = points[..., 2]
    valid = np.isfinite(points).all(-1) & (z > max(a.min_depth, 1e-6))
    if a.max_depth:
        valid &= z <= a.max_depth
    if confidence is not None:
        if not np.all(np.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)):
            raise ValueError("confidence must contain finite sigmoid scores in [0,1]")
        valid &= confidence >= a.confidence
    if a.depth_edge:
        finite = np.isfinite(z) & (z > 0)
        kernel = np.ones((a.edge_kernel, a.edge_kernel), np.uint8)
        high = cv2.dilate(np.where(finite, z, -1e20), kernel)
        low = cv2.erode(np.where(finite, z, 1e20), kernel)
        valid &= (high - low) <= a.depth_edge * np.maximum(z, 1e-6)
    return valid


class SkyMasks:
    """Read-only, precomputed masks. No model inference or mask generation here.

    Validate every saved dense frame before creating render outputs. RGB hashes
    bind masks to the actual input frames, not just filenames or tensor shapes.
    Masks already include thresholding/dilation; neither is repeated at render.
    """
    def __init__(self, a, reader):
        self.root = a.sky_mask_dir or a.result_dir / "sky_masks"
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if not isinstance(m, dict) or type(m.get("schema_version")) is not int or m["schema_version"] != 1 or m.get("mask_kind") != "sky_rejected" or m.get("status") != "complete":
            raise ValueError("Sky masks are incomplete or incompatible; run generate_sky_masks.py first")
        self.hw = (reader.h, reader.w)
        for name in ("native_hw", "dense_output_indices"):
            values = m.get(name)
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise ValueError(f"Sky mask {name} must contain integer indices/dimensions")
        if type(m.get("frames")) is not int:
            raise ValueError("Sky mask frames must be an integer")
        if m.get("native_hw") != list(self.hw) or m.get("frames") != len(reader.poses) or m.get("dense_output_indices") != reader.indices.tolist():
            raise ValueError("Sky mask geometry/frame mapping does not match this inference result")
        for name in ("color_sha256", "mask_sha256"):
            values = m.get(name)
            if not isinstance(values, list) or len(values) != reader.m or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values):
                raise ValueError(f"Invalid sky mask manifest field: {name}")
        self.rows = {int(index): row for row, index in enumerate(reader.indices)}
        for row, index in enumerate(reader.indices):
            digest = hashlib.sha256(reader.colors[row].numpy().tobytes()).hexdigest()
            if digest != m["color_sha256"][row]:
                raise ValueError(f"Sky mask RGB mismatch at inference frame {index}; regenerate masks for this result")
            self.rejected(int(index))
            if reader.m >= 200 and (row+1) % 200 == 0:
                print(f"Validated sky masks: {row+1}/{reader.m}", flush=True)

    def rejected(self, pose_index):
        row = self.rows[pose_index]
        path = self.root / f"{pose_index:08d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing sky mask: {path}; regenerate masks before rendering")
        # Recheck the PNG at use time too, rather than silently accepting a
        # file replaced while a long render is running. Masks are small files.
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.manifest["mask_sha256"][row]:
            raise ValueError(f"Sky mask checksum mismatch: {path}")
        with Image.open(BytesIO(data)) as im:
            if im.format != "PNG" or im.mode != "L":
                raise ValueError(f"Sky mask must be a grayscale PNG: {path}")
            mask = np.asarray(im)
        if mask.shape != self.hw or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
            raise ValueError(f"Sky mask must be a native-resolution 0/255 grayscale PNG: {path}")
        return mask == 255


def prepare_cloud(reader, row, a, pixel_step, sky=None):
    """One shared filter/backprojection path for ground AND global bird clouds."""
    pose_index = int(reader.indices[row])
    points = reader.local_points(row)
    conf = None if reader.confidence is None else reader.confidence[row].float().numpy()
    valid = native_depth_valid(points, conf, a)
    stats = {"pose_index": pose_index, "native_pixels": valid.size, "after_geometry_conf_edge": int(valid.sum())}
    colors = reader.colors[row].numpy()
    rgb = reader.rgb(pose_index) if a.resolution == "source" else None
    geometry = reader.geometry
    if geometry is not None:
        valid &= geometry[2]
    if sky is not None:
        valid &= ~sky.rejected(pose_index)
    stats["after_padding_sky"] = int(valid.sum())
    if a.resolution == "source":
        mx, my = geometry[:2]
        method = cv2.INTER_LINEAR if a.upsample == "bilinear" else cv2.INTER_NEAREST
        # Interpolate camera rays and Z separately: do NOT refit K and alter the
        # original predicted geometry. Only fully valid interpolation footprints
        # survive; invalid/sky/NaN pixels cannot leak back into the reconstruction.
        z = points[..., 2]
        rays = np.divide(points, z[..., None], out=np.zeros_like(points), where=np.isfinite(z[..., None]) & (z[..., None] > 1e-6))
        depth = cv2.remap(np.where(valid, z, 0), mx, my, method, borderMode=cv2.BORDER_REPLICATE)
        rays = cv2.remap(np.where(valid[..., None], rays, 0), mx, my, method, borderMode=cv2.BORDER_REPLICATE)
        coverage = cv2.remap(valid.astype(np.float32), mx, my, method, borderMode=cv2.BORDER_REPLICATE)
        h, w = valid.shape
        valid = (coverage > 1 - 1e-6) & (mx >= -0.5) & (mx < w - 0.5) & (my >= -0.5) & (my < h - 0.5)
        points = depth[..., None] * rays
        colors = rgb
    mask = valid[::pixel_step, ::pixel_step]
    local = points[::pixel_step, ::pixel_step][mask]
    color = colors[::pixel_step, ::pixel_step][mask].copy()
    pose = reader.poses[pose_index]
    world = (local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    stats["output_points"] = len(world)
    return world, color, stats


# --------------------- Virtual cameras and trajectory geometry --------------

def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("Cannot normalize a degenerate camera axis")
    return np.asarray(vector, np.float64) / length


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float                         # Vertical FoV, degrees (legacy convention).

    @property
    def rotation(self):
        forward = unit(self.target - self.eye)
        cross = np.cross(forward, self.up)
        if np.linalg.norm(cross) < 1e-8:
            # A degenerate observer-up can occur at a 180-degree camera turn.
            # Choose a perpendicular observing axis; do not modify any poses.
            cross = np.cross(forward, np.eye(3)[np.argmin(np.abs(forward))])
        right = unit(cross)
        true_up = unit(np.cross(right, forward))
        return np.column_stack((right, -true_up, forward))  # C2W, Y down.

    def project(self, points, width, height):
        local = (points - self.eye) @ self.rotation
        f = height / (2 * math.tan(math.radians(self.fov) / 2))
        xy = local[:, :2] * (f / np.maximum(local[:, 2:3], 1e-6))
        xy += [(width - 1) / 2, (height - 1) / 2]
        return xy, local[:, 2]

    def as_dict(self):
        return {"eye": self.eye.tolist(), "target": self.target.tolist(), "up": self.up.tolist(), "vertical_fov": self.fov, "rotation_c2w": self.rotation.tolist()}


def follow_camera(poses, index, a, distance):
    # Wangjing uses a centered radius-6 mean for center/forward (13 samples),
    # but a centered 6-sample window for up. These are observer-only changes.
    window = poses[max(0, index - a.smooth_radius):index + a.smooth_radius + 1]
    forward = window[:, :3, 2].mean(0)
    lo = max(0, index - a.smooth_frames // 2)
    hi = min(len(poses), lo + a.smooth_frames)
    lo = max(0, hi - a.smooth_frames)
    up = -poses[lo:hi, :3, 1].mean(0)
    forward = unit(forward if np.linalg.norm(forward) > 1e-6 else poses[index, :3, 2])
    up = unit(up if np.linalg.norm(up) > 1e-6 else -poses[index, :3, 1])
    # Only the observing camera is smoothed. Every reconstructed camera keeps
    # its original rotation AND translation for point transformations/frustums.
    center = window[:, :3, 3].mean(0)
    return Camera(center - forward * distance + up * distance * a.height_ratio,
                  center + forward * distance * a.look_ahead, up, a.ground_fov)


def bird_camera(poses, reference, a):
    up = np.median(-poses[:, :3, 1], axis=0)
    up = unit(up if np.linalg.norm(up) > 1e-6 else reference.up)
    # Match Wangjing's deterministic PCA orientation and trajectory-only fit.
    # Point outliers must not move the common bird PNG / transition camera.
    centers = poses[:, :3, 3]
    planar = centers - np.median(centers, axis=0)
    planar -= (planar @ up)[:, None] * up
    eigenvalues, eigenvectors = np.linalg.eigh(planar.T @ planar / len(planar))
    right = eigenvectors[:, np.argmax(eigenvalues)]
    right -= up * np.dot(right, up)
    if np.linalg.norm(right) < 1e-6 or eigenvalues.max() < 1e-12:
        right = reference.rotation[:, 0] - up * np.dot(reference.rotation[:, 0], up)
        if np.linalg.norm(right) < 1e-6:
            axis = np.eye(3)[np.argmin(np.abs(up))]
            right = axis - up * np.dot(axis, up)
    right = unit(right)
    first_motion = planar[min(len(planar)-1, max(1, len(planar)//20))]
    if np.dot(right, first_motion) < 0:
        right = -right
    image_up = unit(np.cross(right, -up))
    x, y, z = centers @ right, centers @ image_up, centers @ up
    z_mid = np.median(z)
    center = right * (x.min()+x.max())/2 + image_up * (y.min()+y.max())/2 + up * z_mid
    half_x, half_y = max(np.ptp(x)/2, 1e-3), max(np.ptp(y)/2, 1e-3)
    tany = math.tan(math.radians(a.bird_fov)/2)
    distance = a.bird_margin * max(half_y/tany, half_x/(tany*a.width/a.height))
    distance += max(z.max()-z_mid, 0)
    return Camera(center + up * distance, center, image_up, a.bird_fov)


def transition_camera(start, end, fraction):
    """Animate the virtual camera, NOT images or reconstructed poses."""
    if fraction <= 0:
        return start
    if fraction >= 1:
        return end
    t = fraction * fraction * (3 - 2 * fraction)
    rotations = Rotation.from_matrix(np.stack([start.rotation, end.rotation]))
    basis = Slerp([0, 1], rotations)(t).as_matrix()
    d0, d1 = np.linalg.norm(start.eye - start.target), np.linalg.norm(end.eye - end.target)
    distance = math.exp((1 - t) * math.log(d0) + t * math.log(d1))
    # Couple target travel to zoom, avoiding an early large lateral fly-out.
    q = t if abs(d1 - d0) < 1e-8 else (distance - d0) / (d1 - d0)
    target = (1 - q) * start.target + q * end.target
    return Camera(target - basis[:, 2] * distance, target, -basis[:, 1], (1 - t) * start.fov + t * end.fov)


def time_colors(count):
    stops = np.array([0, .16, .32, .48, .65, .82, 1])
    rgb = np.array([[37, 99, 235], [56, 189, 248], [6, 182, 212], [34, 197, 94], [234, 179, 8], [249, 115, 22], [239, 68, 68]])
    t = np.linspace(0, 1, count)
    return np.column_stack([np.interp(t, stops, rgb[:, c]) for c in range(3)]).round().astype(np.uint8)


def fit_intrinsics(points):
    """Approximate pinhole intrinsics for DISPLAY FRUSTUMS ONLY, never geometry."""
    h, w = points.shape[:2]
    step = max(1, min(h, w) // 40)
    yy, xx = np.mgrid[:h:step, :w:step]
    samples = points[::step, ::step].reshape(-1, 3)
    valid = np.isfinite(samples).all(1) & (samples[:, 2] > 1e-6)
    if valid.sum() < 8:
        return None
    rays = samples[valid, :2] / samples[valid, 2:3]
    params, errors = [], []
    for axis, pixel in enumerate((xx.ravel()[valid], yy.ravel()[valid])):
        design = np.column_stack((rays[:, axis], np.ones(len(rays))))
        keep = np.ones(len(rays), bool)
        for _ in range(3):
            coeff = np.linalg.lstsq(design[keep], pixel[keep], rcond=None)[0]
            err = np.abs(design @ coeff - pixel)
            keep = err <= max(3 * np.median(err), 0.5)
        if coeff[0] <= 0 or not np.isfinite(coeff).all():
            return None
        params.extend(coeff)
        errors.append(float(np.percentile(err, 90)))
    # fx,cx,fy,cy; learned rays need not be an exact pinhole camera.
    return np.asarray(params), max(errors)


def trajectory_geometry(reader, a):
    n = len(reader.poses)
    colors = time_colors(n)
    if not a.frustum_step:
        return colors, None, []
    fitted = [fit_intrinsics(reader.local_points(j)) for j in range(reader.m)]
    usable = [j for j, value in enumerate(fitted) if value is not None]
    if not usable:
        raise ValueError("Cannot estimate camera FoV; use --frustum-step 0 to draw trajectory only")
    known_indices = reader.indices[usable]
    values = np.stack([fitted[j][0] for j in usable])
    params = np.column_stack([np.interp(np.arange(n), known_indices, values[:, c]) for c in range(4)])
    errors = [fitted[j][1] for j in usable]
    if max(errors) > 3:
        warnings.warn(f"Display-frustum pinhole fit P90 residual reaches {max(errors):.2f}px. Point geometry is NOT refitted.")
    z = a.frustum_size
    corners = []
    for fx, cx, fy, cy in params:
        pixels = np.array([[-.5, -.5], [reader.w-.5, -.5], [reader.w-.5, reader.h-.5], [-.5, reader.h-.5]])
        corners.append(np.column_stack(((pixels[:, 0]-cx)/fx*z, (pixels[:, 1]-cy)/fy*z, np.full(4, z))))
    corners = np.asarray(corners)
    corners = np.einsum("nij,nkj->nki", reader.poses[:, :3, :3], corners) + reader.poses[:, None, :3, 3]
    report = [{"pose_index": int(reader.indices[j]), "fx_cx_fy_cy": fitted[j][0].tolist(), "fit_p90_pixels": fitted[j][1]} for j in usable]
    return colors, corners, report


# ----------------------- Bounded point-splat rasterizer ----------------------

class Rasterizer:
    """Small CPU/CUDA z-buffer. Streams chunks, without OpenGL/display servers.

    Occlusion here only chooses what a frame displays. It NEVER compresses or
    deletes historical geometry. Equal-depth ties keep the first point, making
    CPU/GPU and differently chunked outputs deterministic.
    """
    def __init__(self, a):
        self.w, self.h, self.size = a.width, a.height, a.point_size
        self.device = torch.device(a.device)
        self.chunk = a.chunk_points
        self.clear()

    def clear(self):
        self.depth = torch.full((self.h*self.w,), float("inf"), device=self.device)
        self.rgb = torch.zeros((self.h*self.w, 3), dtype=torch.uint8, device=self.device)
        self.owner = torch.full((self.h*self.w,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=self.device)
        self.point_offset = 0

    def add(self, points, colors, camera):
        f = self.h / (2 * math.tan(math.radians(camera.fov) / 2))
        rot = torch.as_tensor(camera.rotation, dtype=torch.float32, device=self.device)
        eye = torch.as_tensor(camera.eye, dtype=torch.float32, device=self.device)
        offsets = range(-(self.size // 2), self.size - self.size // 2)
        point_offset = self.point_offset
        self.point_offset += len(points)
        for begin in range(0, len(points), self.chunk):
            p = torch.as_tensor(points[begin:begin+self.chunk], dtype=torch.float32, device=self.device)
            c = torch.as_tensor(colors[begin:begin+self.chunk], dtype=torch.uint8, device=self.device)
            local = (p - eye) @ rot
            source_ids = torch.arange(len(p), device=self.device) + point_offset + begin
            visible = torch.isfinite(local).all(1) & (local[:, 2] > 1e-5)
            local, c, source_ids = local[visible], c[visible], source_ids[visible]
            xy = local[:, :2] / local[:, 2:3] * f
            xy += torch.tensor([(self.w-1)/2, (self.h-1)/2], device=self.device)
            # Exclude far-offscreen coordinates before integer conversion.
            visible = (xy[:, 0] >= -self.size) & (xy[:, 0] < self.w+self.size) & (xy[:, 1] >= -self.size) & (xy[:, 1] < self.h+self.size)
            xy, local, c, source_ids = xy[visible].round().long(), local[visible], c[visible], source_ids[visible]
            if not len(xy):
                continue
            for dy in offsets:
                for dx in offsets:
                    x, y = xy[:, 0]+dx, xy[:, 1]+dy
                    ok = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
                    pixel = y[ok] * self.w + x[ok]
                    z, color = local[ok, 2], c[ok]
                    if not len(z):
                        continue
                    nearest = torch.full_like(self.depth, float("inf"))
                    nearest.scatter_reduce_(0, pixel, z, reduce="amin", include_self=True)
                    ids = torch.arange(len(z), device=self.device)
                    candidates = torch.where(z == nearest[pixel], ids, len(z))
                    winner = torch.full((self.h*self.w,), len(z), dtype=torch.long, device=self.device)
                    winner.scatter_reduce_(0, pixel, candidates, reduce="amin", include_self=True)
                    hit = winner < len(z)
                    owner = torch.full_like(self.owner, torch.iinfo(torch.int64).max)
                    owner[hit] = source_ids[ok][winner[hit]]
                    changed = hit & ((nearest < self.depth) | ((nearest == self.depth) & (owner < self.owner)))
                    self.rgb[changed] = color[winner[changed]]
                    self.depth[changed] = nearest[changed]
                    self.owner[changed] = owner[changed]

    def rgba(self):
        rgb = self.rgb.reshape(self.h, self.w, 3).cpu().numpy()
        alpha = torch.isfinite(self.depth).reshape(self.h, self.w).cpu().numpy().astype(np.uint8) * 255
        return np.dstack((rgb, alpha))


def in_frustum(points, camera, width, height, margin=0):
    xy, z = camera.project(points, width, height)
    return (z > 1e-5) & np.isfinite(xy).all(1) & (xy[:, 0] >= -margin) & (xy[:, 0] < width+margin) & (xy[:, 1] >= -margin) & (xy[:, 1] < height+margin)


def trajectory_layer(poses, colors, corners, camera, indices, a):
    """RGB premultiplied by coverage plus a separate alpha channel.

    Lines are intentionally overlays so the complete historical path remains
    legible. They do not participate in point-cloud occlusion.
    """
    layer = np.zeros((a.height, a.width, 4), np.uint8)
    basis = camera.rotation
    f = a.height / (2 * math.tan(math.radians(camera.fov) / 2))

    def line(p0, p1, color, thickness):
        q = (np.stack([p0, p1]) - camera.eye) @ basis
        if np.all(q[:, 2] <= 1e-5):
            return
        # Clip a crossing segment at the near plane before projection.
        for i in range(2):
            if q[i, 2] < 1e-5:
                j = 1-i
                t = (1e-5-q[i, 2])/(q[j, 2]-q[i, 2])
                q[i] += t*(q[j]-q[i])
        xy = q[:, :2]/q[:, 2:3]*f + [(a.width-1)/2, (a.height-1)/2]
        xy = np.clip(np.rint(xy), -10000000, 10000000).astype(int)
        ok, p0, p1 = cv2.clipLine((0, 0, a.width, a.height), tuple(xy[0]), tuple(xy[1]))
        if ok:
            cv2.line(layer, p0, p1, (*map(int, color), 255), thickness, cv2.LINE_AA)

    centers = poses[:, :3, 3]
    for previous, current in zip(indices[:-1], indices[1:]):
        line(centers[previous], centers[current], colors[current], a.trajectory_width)
    xy, z = camera.project(centers[indices], a.width, a.height)
    for k, index in enumerate(indices):
        if a.frustum_step and index % a.frustum_step == 0:
            for j in range(4):
                line(centers[index], corners[index, j], colors[index], a.frustum_width)
                line(corners[index, j], corners[index, (j+1) % 4], colors[index], a.frustum_width)
        if a.center_size and z[k] > 1e-5 and np.isfinite(xy[k]).all() and -a.center_size <= xy[k, 0] < a.width+a.center_size and -a.center_size <= xy[k, 1] < a.height+a.center_size:
            x, y = np.rint(xy[k]).astype(int)
            d = a.center_size
            cv2.rectangle(layer, (x-d//2, y-d//2), (x-d//2+d-1, y-d//2+d-1), (*map(int, colors[index]), 255), -1)
    return layer


def straight_alpha(layer):
    alpha = layer[..., 3:4].astype(np.float32) / 255
    rgb = np.divide(layer[..., :3], alpha, out=np.zeros_like(layer[..., :3], dtype=np.float32), where=alpha > 0)
    return np.dstack((np.clip(np.rint(rgb), 0, 255).astype(np.uint8), layer[..., 3]))


def composite(cloud, trajectory, background, opacity=1):
    base = np.broadcast_to(np.asarray(background, np.float32), cloud.shape[:2]+(3,)).copy()
    alpha = cloud[..., 3:4].astype(np.float32) / 255 * opacity
    base = cloud[..., :3] * alpha + base * (1-alpha)
    ta = trajectory[..., 3:4].astype(np.float32) / 255
    return np.clip(np.rint(trajectory[..., :3] + base * (1-ta)), 0, 255).astype(np.uint8)


class VideoWriter:
    def __init__(self, path, width, height, a):
        # Edge-pad a possible odd original RGB size by <=1 pixel, never stretch.
        self.log = path.with_suffix(".ffmpeg.log").open("wb")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(a.fps), "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-threads", "4", "-preset", a.preset, "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.log)
        self.shape = (height, width, 3)

    def write(self, rgb):
        if rgb.shape != self.shape or rgb.dtype != np.uint8:
            raise ValueError("video frame size/dtype changed")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError(f"ffmpeg exited with {code}; inspect the adjacent .ffmpeg.log")


# ---------------------------- Output orchestration --------------------------

def frame_plan(a, count):
    end = count if a.end is None else a.end
    if not 0 <= a.start < end <= count:
        raise ValueError("require 0 <= start < end <= inference frame count")
    return np.arange(a.start, end, a.view_step, dtype=int)[:a.ground_frames], end


def selected_rows(reader, start, end, step):
    # Step is in inference-index space, NOT dense-array-row space. Missing dense
    # frames are never fabricated. The origin is the selected interval's start.
    indices = reader.indices
    return np.flatnonzero((indices >= start) & (indices < end) & ((indices-start) % step == 0))


def render(a):
    validate_args(a)
    reader = ReconstructionReader(a)
    # Missing/corrupt/mismatched masks fail before creating or overwriting any
    # render artifacts, including when only a subsection will be visualized.
    sky = SkyMasks(a, reader) if a.sky else None
    views, end = frame_plan(a, len(reader.poses))
    scope_start, scope_end = (0, len(reader.poses)) if a.global_scope == "all" else (a.start, end)
    need_bird = "trajectory" in a.outputs or "pointcloud" in a.outputs or ("video" in a.outputs and a.transition_frames > 0)
    global_indices = np.arange(scope_start, scope_end)
    global_rows = selected_rows(reader, scope_start, scope_end, a.bird_cloud_step)
    if need_bird and len(global_rows) == 0:
        raise ValueError("No saved dense frames match bird-cloud-step in selected scope")
    ground_rows = selected_rows(reader, a.start, end, a.cloud_step)
    if "video" in a.outputs and not len(ground_rows):
        raise ValueError("No saved dense frames match cloud-step in selected interval")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # Only our named output artifacts may be replaced; never remove input data.
    if a.overwrite:
        for name in ("ground_pointcloud.mp4", "ground_rgb.mp4", "last_frame.png", "last_frame_transparent.png", "birdseye_trajectory.png", "birdseye_pointcloud.png", "render_manifest.json", "filter_stats.jsonl", "frustum_intrinsics.json"):
            (a.output_dir / name).unlink(missing_ok=True)
    started = time.monotonic()
    manifest = {
        "status": "running", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "input_metadata": reader.metadata, "pose_count": len(reader.poses),
        "dense_indices": reader.indices.tolist(), "ground_view_indices": views.tolist(),
        "source_size_hw": reader.source_hw, "ground_cloud_indices": [],
        "global_cloud_indices": reader.indices[global_rows].tolist(),
        "transition_timeline": "freeze last reconstruction frame; animate virtual camera only",
        "frustum_intrinsics": "estimated from point-map rays; interpolated for missing dense frames; visualization only",
        "sky_masks": None if sky is None else {"directory": str(sky.root), "manifest": sky.manifest},
    }
    def save_manifest():
        (a.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_manifest()
    writers = []
    try:
        colors, corners, fits = trajectory_geometry(reader, a) if "video" in a.outputs or "trajectory" in a.outputs else (None, None, [])
        (a.output_dir / "frustum_intrinsics.json").write_text(json.dumps(fits, indent=2) + "\n")
        centers = reader.poses[:, :3, 3]
        radius = max(float(np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 90)), 0.5)
        distance = a.follow_distance or float(np.clip(a.distance_ratio * radius, a.distance_min, a.distance_max))
        last_index = int(views[-1]) if len(views) else end-1
        last_ground = follow_camera(reader.poses, last_index, a, distance)

        with (a.output_dir / "filter_stats.jsonl").open("w") as stats_file:
            def record(phase, stats):
                stats_file.write(json.dumps({"phase": phase, **stats}) + "\n")

            # One trajectory-fitted camera is shared by both PNGs and the
            # transition endpoint. No point-cloud bounds pass is needed.
            bird_point_count = 0
            bird = bird_camera(reader.poses[global_indices], last_ground, a) if need_bird else last_ground
            manifest.update(bird_camera=bird.as_dict() if need_bird else None, follow_distance=distance, global_point_count=bird_point_count)
            save_manifest()
            raster = Rasterizer(a)

            video = rgb_video = None
            if "video" in a.outputs:
                video = VideoWriter(a.output_dir / "ground_pointcloud.mp4", a.width, a.height, a)
                writers.append(video)
            if "rgb" in a.outputs:
                h, w = reader.source_hw
                rgb_video = VideoWriter(a.output_dir / "ground_rgb.mp4", w, h, a)
                writers.append(rgb_video)

            clouds, cursor, resident = deque(), 0, 0
            frame_log = []
            for number, pose_index in enumerate(views if video or rgb_video else []):
                pose_index = int(pose_index)
                camera = follow_camera(reader.poses, pose_index, a, distance)
                added = []
                if video:
                    # S=2,P=1 adds BOTH intervening dense frames before drawing.
                    while cursor < len(ground_rows) and reader.indices[ground_rows[cursor]] <= pose_index:
                        row = ground_rows[cursor]
                        xyz, rgb, stats = prepare_cloud(reader, row, a, a.pixel_step, sky)
                        index = int(reader.indices[row])
                        record("ground", stats)
                        clouds.append((index, xyz, rgb))
                        resident += len(xyz)
                        if resident > a.max_resident_points:
                            raise MemoryError(f"Resident cloud {resident:,} exceeds budget while adding frames; reduce sampling/history or raise max-resident-points")
                        added.append(index)
                        cursor += 1
                    if a.history:
                        while clouds and clouds[0][0] <= pose_index - a.history:
                            resident -= len(clouds.popleft()[1])
                    if a.prune_every and number % a.prune_every == 0:
                        kept = deque()
                        for index, xyz, rgb in clouds:
                            mask = in_frustum(xyz, camera, a.width, a.height, a.point_size)
                            kept.append((index, xyz[mask], rgb[mask]))
                        clouds = kept
                        resident = sum(len(item[1]) for item in clouds)
                    if resident > a.max_resident_points:
                        raise MemoryError(f"Resident cloud {resident:,} exceeds budget; change sampling/history/pruning or explicitly raise max-resident-points")
                    raster.clear()
                    for _, xyz, rgb in clouds:
                        raster.add(xyz, rgb, camera)
                    partial = np.arange(scope_start, pose_index+1)
                    layer = trajectory_layer(reader.poses, colors, corners, camera, partial, a)
                    frame = composite(raster.rgba(), layer, a.background)
                    video.write(frame)
                    frame_log.append({"pose_index": pose_index, "added_dense_pose_indices": added, "resident_points": resident})
                    manifest["ground_cloud_indices"].extend(added)
                if rgb_video:
                    rgb_video.write(reader.rgb(pose_index))
                if number % 20 == 0 or number+1 == len(views):
                    print(f"Ground: {number+1}/{len(views)}, pose {pose_index}, resident {resident:,} points", flush=True)

            # The scene stops advancing. No optical-flow/image interpolation,
            # no new geometry, and no irreversible pruning during the turn.
            for number in range(a.transition_frames if video or rgb_video else 0):
                fraction = (number+1)/a.transition_frames
                camera = transition_camera(last_ground, bird, fraction)
                if video:
                    raster.clear()
                    if fraction < 1:
                        for _, xyz, rgb in clouds:
                            raster.add(xyz, rgb, camera)
                    layer = trajectory_layer(reader.poses, colors, corners, camera, np.arange(scope_start, last_index+1), a)
                    frame = composite(raster.rgba(), layer, a.background, opacity=1-fraction*fraction*(3-2*fraction))
                    video.write(frame)
                if rgb_video:
                    rgb_video.write(reader.rgb(last_index))
                if number % 30 == 0:
                    print(f"Transition: {number+1}/{a.transition_frames}", flush=True)
            if video:
                Image.fromarray(frame).save(a.output_dir / "last_frame.png")
                if a.transition_frames:
                    Image.fromarray(straight_alpha(layer)).save(a.output_dir / "last_frame_transparent.png")
            if "trajectory" in a.outputs:
                layer = trajectory_layer(reader.poses, colors, corners, bird, global_indices, a)
                Image.fromarray(straight_alpha(layer)).save(a.output_dir / "birdseye_trajectory.png")
            if "pointcloud" in a.outputs:
                raster.clear()
                for number, row in enumerate(global_rows):
                    xyz, rgb, stats = prepare_cloud(reader, row, a, a.bird_pixel_step, sky)
                    record("bird_image", stats)
                    bird_point_count += len(xyz)
                    raster.add(xyz, rgb, bird)
                    if number % 100 == 0:
                        print(f"Bird image: {number+1}/{len(global_rows)} dense frames", flush=True)
                Image.fromarray(raster.rgba()).save(a.output_dir / "birdseye_pointcloud.png")
        for writer in writers:
            writer.close()
        writers.clear()
        manifest.update(status="complete", elapsed_seconds=time.monotonic()-started,
                        global_point_count=bird_point_count,
                        ground_frame_count=len(views) if video or rgb_video else 0,
                        transition_frame_count=a.transition_frames if video or rgb_video else 0,
                        ground_frame_log=frame_log)
        save_manifest()
        print(f"Complete: {a.output_dir} ({manifest['elapsed_seconds']:.1f}s)", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        save_manifest()
        for writer in writers:
            try:
                writer.close()
            except (OSError, ValueError, RuntimeError):
                pass
        raise


if __name__ == "__main__":
    render(parser().parse_args())
