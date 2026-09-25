#!/usr/bin/env python3
"""Preprocess ScanNet++ as one collection of ordered device sequences.

This is the sequence-oriented counterpart of ``preprocess_scannetpp.py``.  It
does not consume precomputed image pairs.  Every training scene produces up to
two independent sequences:

* ``<scene_id>_iphone``: raw video frames sampled by the
  original frame number (default: ``frame_id % 30 == 0``), using ScanNet++'s
  per-frame ``aligned_pose`` in the common scan world.
* ``<scene_id>_dslr``: the longest continuous DSLR run with valid COLMAP
  calibration.  Images are ordered by capture prefix/id and adjacent views
  must pass co-visibility, translation, and rotation checks.

For both streams the script applies the same undistortion as the CUT3R
ScanNet++ pair preprocessor, resizes to a fixed width while preserving aspect
ratio, center-crops only the vertical dimension when needed, renders metric
depth from the aligned mesh, and writes the common sequence training layout.

The script is restart-safe: files are written atomically, each sequence has a
configuration/source-selection fingerprint, and ``.complete.json`` is created
only after every expected frame is valid.  ``--verify`` deeply validates an
existing completed sequence; ``--validate_only`` never renders or rewrites
frame data.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


SCRIPT_VERSION = "2.0.0"
SCHEMA_VERSION = 2
SEQUENCE_LAYOUT = "flat_scene_stream_v2"
SEQUENCE_STREAMS = ("iphone", "dslr")

OPENGL_TO_OPENCV = np.asarray(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

IPHONE_RE = re.compile(r"^frame_(?P<frame_id>\d+)\.(?:jpg|JPG)$")
DSLR_RE = re.compile(
    r"^(?:(?P<prefix>.+)_)?DSC(?P<frame_id>\d+)\.(?:jpe?g)$",
    re.IGNORECASE,
)
NUMERIC_OUTPUT_RE = re.compile(r"^\d{5}\.(?:jpg|npy|npz)$")


@dataclass(frozen=True)
class FrameInfo:
    source_name: str
    frame_id: int
    camera_id: int
    intrinsics: Tuple[Any, ...]
    camera_pose: np.ndarray
    timestamp: Optional[float] = None
    pose_source: str = "colmap"
    colmap_image_id: Optional[int] = None
    capture_prefix: str = ""


class SequenceFiltered(RuntimeError):
    """A source sequence intentionally omitted by deterministic selection rules."""

    def __init__(self, message: str, stats: Dict[str, Any]):
        super().__init__(message)
        self.stats = stats


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sequence_id(scene_id: str, stream: str) -> str:
    if stream not in SEQUENCE_STREAMS:
        raise ValueError(f"Unsupported stream: {stream}")
    return f"{scene_id}_{stream}"


# CUT3R/DUSt3R intrinsics conventions; derived portions retain CC BY-NC-SA 4.0.
# Copyright (C) 2024-present Naver Corporation. See THIRD_PARTY_NOTICES.md.
def colmap_to_opencv_intrinsics(k: np.ndarray) -> np.ndarray:
    """COLMAP pixel centers are offset by +0.5 relative to OpenCV."""
    k = k.copy()
    k[0, 2] -= 0.5
    k[1, 2] -= 0.5
    return k


def opencv_to_colmap_intrinsics(k: np.ndarray) -> np.ndarray:
    """Inverse pixel-center conversion, used by the mesh renderer."""
    k = k.copy()
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    return k


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def temp_path(path: str) -> str:
    return f"{path}.tmp-{os.getpid()}-{time.time_ns()}"


def atomic_write_bytes(path: str, payload: bytes) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = temp_path(path)
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def atomic_write_text(path: str, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def atomic_save_npy(path: str, value: np.ndarray) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = temp_path(path)
    try:
        with open(tmp, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def atomic_save_npz(path: str, compressed: bool = True, **values: np.ndarray) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = temp_path(path)
    try:
        with open(tmp, "wb") as handle:
            if compressed:
                np.savez_compressed(handle, **values)
            else:
                np.savez(handle, **values)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def atomic_save_pickle(path: str, value: Any) -> None:
    atomic_write_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_save_jpeg(path: str, rgb: np.ndarray, quality: int) -> None:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    tmp = temp_path(path)
    os.makedirs(osp.dirname(path), exist_ok=True)
    try:
        with open(tmp, "wb") as handle:
            image.save(handle, format="JPEG", quality=quality, subsampling=0)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def pose_from_qwxyz_txyz(values: Sequence[str]) -> np.ndarray:
    qw, qx, qy, qz, tx, ty, tz = map(float, values)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    w2c[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(w2c).astype(np.float32)


def load_colmap_cameras(path: str) -> Dict[int, Tuple[Any, ...]]:
    cameras: Dict[int, Tuple[Any, ...]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed COLMAP camera line in {path}: {line}")
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = tuple(float(v) for v in parts[4:])
            cameras[camera_id] = (model, width, height, *params)
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def frame_number(source_name: str, stream: str) -> int:
    matcher = IPHONE_RE if stream == "iphone" else DSLR_RE
    match = matcher.match(osp.basename(source_name))
    if match is None:
        raise ValueError(f"Unexpected {stream} image name: {source_name}")
    return int(match.group("frame_id"))


def capture_prefix(source_name: str, stream: str) -> str:
    if stream != "dslr":
        return ""
    match = DSLR_RE.match(osp.basename(source_name))
    if match is None:
        raise ValueError(f"Unexpected DSLR image name: {source_name}")
    return match.group("prefix") or ""


def load_colmap_images(
    path: str, cameras: Dict[int, Tuple[Any, ...]], stream: str
) -> List[FrameInfo]:
    """Load COLMAP image poses while intentionally skipping the points2D rows."""
    frames: List[FrameInfo] = []
    seen_names = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Image records have: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME.
            # The alternating points2D records never end in a JPEG filename.
            if len(parts) < 10 or not re.search(r"\.(?:jpg|JPG)$", parts[-1]):
                continue
            source_name = parts[-1]
            if source_name in seen_names:
                raise ValueError(f"Duplicate COLMAP image name: {source_name}")
            camera_id = int(parts[-2])
            if camera_id not in cameras:
                raise KeyError(f"Image {source_name} references camera {camera_id}")
            frames.append(
                FrameInfo(
                    source_name=source_name,
                    frame_id=frame_number(source_name, stream),
                    camera_id=camera_id,
                    intrinsics=cameras[camera_id],
                    camera_pose=pose_from_qwxyz_txyz(parts[1:8]),
                    colmap_image_id=int(parts[0]),
                    capture_prefix=capture_prefix(source_name, stream),
                )
            )
            seen_names.add(source_name)
    if not frames:
        raise ValueError(f"No {stream} image poses found in {path}")
    frames.sort(
        key=lambda item: (item.capture_prefix, item.frame_id, item.source_name)
    )
    return frames


def source_dirs(scene_dir: str, stream: str) -> Tuple[str, str, str]:
    if stream == "iphone":
        base = osp.join(scene_dir, "iphone")
        return (
            osp.join(base, "colmap"),
            osp.join(base, "rgb"),
            osp.join(base, "rgb_masks"),
        )
    base = osp.join(scene_dir, "dslr")
    return (
        osp.join(base, "colmap"),
        osp.join(base, "resized_images"),
        osp.join(base, "resized_anon_masks"),
    )


def source_paths(
    rgb_dir: str, mask_dir: str, source_name: str
) -> Tuple[str, str]:
    mask_name = osp.splitext(source_name)[0] + ".png"
    return osp.join(rgb_dir, source_name), osp.join(mask_dir, mask_name)


def list_source_jpegs(rgb_dir: str, stream: str) -> Dict[str, str]:
    matcher = IPHONE_RE if stream == "iphone" else DSLR_RE
    result: Dict[str, str] = {}
    for entry in os.scandir(rgb_dir):
        if not entry.is_file():
            continue
        match = matcher.match(entry.name)
        if match is not None:
            result[entry.name] = entry.name
    return result


def estimate_undistorted_intrinsics(intrinsics: Tuple[Any, ...]) -> np.ndarray:
    """Return the same undistorted OpenCV K used by image preprocessing."""
    camera_type = str(intrinsics[0])
    width, height = int(intrinsics[1]), int(intrinsics[2])
    fx, fy, cx, cy = map(float, intrinsics[3:7])
    distortion = np.asarray(intrinsics[7:], dtype=np.float64)
    k_colmap = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    k_opencv = colmap_to_opencv_intrinsics(k_colmap)
    if camera_type == "OPENCV_FISHEYE":
        if distortion.size != 4:
            raise ValueError(
                f"OPENCV_FISHEYE expects 4 coefficients, got {distortion.size}"
            )
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k_opencv,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        new_k[0, 2] = width / 2.0
        new_k[1, 2] = height / 2.0
    elif camera_type in {"OPENCV", "FULL_OPENCV"}:
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            k_opencv,
            distortion,
            (width, height),
            1,
            (width, height),
            True,
        )
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera_type}")
    return np.asarray(new_k, dtype=np.float64)


def resized_cropped_geometry(
    intrinsics_opencv: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_max_height: int,
) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Scale to an exact width, then center-crop height and update K."""
    output_width = int(target_width)
    output_height_before_crop = max(
        1, int(round(float(source_height) * output_width / float(source_width)))
    )
    sx = output_width / float(source_width)
    sy = output_height_before_crop / float(source_height)
    output_k = np.asarray(intrinsics_opencv, dtype=np.float64).copy()
    output_k[0, :] *= sx
    output_k[1, :] *= sy
    output_height = min(output_height_before_crop, int(target_max_height))
    crop_top = max(0, (output_height_before_crop - output_height) // 2)
    output_k[1, 2] -= crop_top
    output_k[2] = (0.0, 0.0, 1.0)
    return output_k, (output_height, output_width), crop_top


def final_camera_fov_degrees(
    intrinsics: Tuple[Any, ...], target_width: int, target_max_height: int
) -> Tuple[float, float]:
    source_width, source_height = int(intrinsics[1]), int(intrinsics[2])
    output_k, (height, width), _ = resized_cropped_geometry(
        estimate_undistorted_intrinsics(intrinsics),
        source_width,
        source_height,
        target_width,
        target_max_height,
    )
    fx, fy = float(output_k[0, 0]), float(output_k[1, 1])
    cx, cy = float(output_k[0, 2]), float(output_k[1, 2])
    fov_x = np.degrees(np.arctan2(cx, fx) + np.arctan2(width - cx, fx))
    fov_y = np.degrees(np.arctan2(cy, fy) + np.arctan2(height - cy, fy))
    return float(fov_x), float(fov_y)


def adjacent_covisibility_from_colmap_tracks(
    points3d_path: str, frames: Sequence[FrameInfo]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count raw COLMAP track overlap for each adjacent ordered image pair."""
    positions: Dict[int, int] = {}
    for position, frame in enumerate(frames):
        if frame.colmap_image_id is None:
            raise ValueError(f"Missing COLMAP image id for {frame.source_name}")
        image_id = int(frame.colmap_image_id)
        if image_id in positions:
            raise ValueError(f"Duplicate COLMAP image id: {image_id}")
        positions[image_id] = position

    observations = np.zeros(len(frames), dtype=np.int64)
    shared = np.zeros(max(0, len(frames) - 1), dtype=np.int64)
    with open(points3d_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            track_image_ids = {int(value) for value in parts[8::2]}
            track_positions = {
                positions[image_id]
                for image_id in track_image_ids
                if image_id in positions
            }
            for position in track_positions:
                observations[position] += 1
                if position + 1 in track_positions:
                    shared[position] += 1
    denominator = np.maximum(observations[:-1], observations[1:])
    scores = np.divide(
        shared,
        denominator,
        out=np.zeros_like(shared, dtype=np.float64),
        where=denominator > 0,
    )
    return observations, shared, scores


def select_longest_dslr_segment(
    frames: Sequence[FrameInfo],
    points3d_path: str,
    target_width: int,
    target_max_height: int,
    min_segment_frames: int,
    min_shared_points: int,
    min_overlap_score: float,
    max_translation: float,
    rotation_fov_fraction: float,
) -> Tuple[List[FrameInfo], Dict[str, Any]]:
    if not frames:
        raise ValueError("No posed DSLR frames")
    frames = sorted(
        frames,
        key=lambda item: (item.capture_prefix, item.frame_id, item.source_name),
    )
    observations, shared, scores = adjacent_covisibility_from_colmap_tracks(
        points3d_path, frames
    )
    fov_cache: Dict[Tuple[Any, ...], Tuple[float, float]] = {}
    edges: List[Dict[str, Any]] = []
    break_reason_counts = {
        "capture_prefix": 0,
        "shared_points": 0,
        "overlap_score": 0,
        "translation": 0,
        "rotation": 0,
    }
    for edge_index, (first, second) in enumerate(zip(frames, frames[1:])):
        for frame in (first, second):
            if frame.intrinsics not in fov_cache:
                fov_cache[frame.intrinsics] = final_camera_fov_degrees(
                    frame.intrinsics, target_width, target_max_height
                )
        fov_limit = rotation_fov_fraction * min(
            *fov_cache[first.intrinsics], *fov_cache[second.intrinsics]
        )
        translation = float(
            np.linalg.norm(first.camera_pose[:3, 3] - second.camera_pose[:3, 3])
        )
        relative_rotation = first.camera_pose[:3, :3].T @ second.camera_pose[:3, :3]
        rotation_degrees = float(
            np.degrees(Rotation.from_matrix(relative_rotation).magnitude())
        )
        reasons = []
        if first.capture_prefix != second.capture_prefix:
            reasons.append("capture_prefix")
        if int(shared[edge_index]) < min_shared_points:
            reasons.append("shared_points")
        if float(scores[edge_index]) < min_overlap_score:
            reasons.append("overlap_score")
        if translation > max_translation:
            reasons.append("translation")
        if rotation_degrees > fov_limit:
            reasons.append("rotation")
        for reason in reasons:
            break_reason_counts[reason] += 1
        edges.append(
            {
                "valid": not reasons,
                "shared_points": int(shared[edge_index]),
                "overlap_score": float(scores[edge_index]),
                "translation_m": translation,
                "rotation_deg": rotation_degrees,
                "rotation_limit_deg": float(fov_limit),
                "frame_id_gap": int(second.frame_id - first.frame_id),
                "reasons": reasons,
            }
        )

    segments: List[Tuple[int, int]] = []
    segment_start = 0
    for edge_index, edge in enumerate(edges):
        if not edge["valid"]:
            segments.append((segment_start, edge_index + 1))
            segment_start = edge_index + 1
    segments.append((segment_start, len(frames)))

    def segment_rank(segment: Tuple[int, int]) -> Tuple[int, float, int]:
        start, end = segment
        segment_scores = [
            edges[index]["overlap_score"] for index in range(start, end - 1)
        ]
        mean_score = float(np.mean(segment_scores)) if segment_scores else 0.0
        return end - start, mean_score, -start

    selected_start, selected_end = max(segments, key=segment_rank)
    selected = list(frames[selected_start:selected_end])
    selected_edges = edges[selected_start : selected_end - 1]

    def metric_summary(key: str) -> Dict[str, Optional[float]]:
        values = [float(edge[key]) for edge in selected_edges]
        if not values:
            return {"min": None, "mean": None, "max": None}
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    stats: Dict[str, Any] = {
        "ordering": "capture_prefix_then_dsc_id_v1",
        "covisibility": "raw_colmap_track_shared_over_max_observations_v1",
        "posed_physical_images": len(frames),
        "segment_count": len(segments),
        "segment_lengths": [end - start for start, end in segments],
        "longest_segment_frames": len(selected),
        "discarded_images": len(frames) - len(selected),
        "selected_start_name": selected[0].source_name,
        "selected_end_name": selected[-1].source_name,
        "selected_start_frame_id": int(selected[0].frame_id),
        "selected_end_frame_id": int(selected[-1].frame_id),
        "break_reason_counts": break_reason_counts,
        "selected_shared_points": metric_summary("shared_points"),
        "selected_overlap_score": metric_summary("overlap_score"),
        "selected_translation_m": metric_summary("translation_m"),
        "selected_rotation_deg": metric_summary("rotation_deg"),
        "selected_rotation_limit_deg": metric_summary("rotation_limit_deg"),
        "selected_observations": {
            "min": int(np.min(observations[selected_start:selected_end])),
            "mean": float(np.mean(observations[selected_start:selected_end])),
            "max": int(np.max(observations[selected_start:selected_end])),
        },
    }
    if len(selected) < min_segment_frames:
        raise SequenceFiltered(
            f"longest DSLR segment has {len(selected)} frames, below "
            f"--dslr_min_segment_frames={min_segment_frames}",
            stats,
        )
    return selected, stats


def select_frames(
    scene_dir: str,
    stream: str,
    iphone_stride: int,
    iphone_offset: int,
    target_width: int,
    target_max_height: int,
    dslr_min_segment_frames: int,
    dslr_min_shared_points: int,
    dslr_min_overlap_score: float,
    dslr_max_translation: float,
    dslr_rotation_fov_fraction: float,
    max_frames: Optional[int],
    allow_missing_colmap: bool,
) -> Tuple[List[FrameInfo], Dict[str, Any]]:
    sfm_dir, rgb_dir, mask_dir = source_dirs(scene_dir, stream)
    required = [
        osp.join(sfm_dir, "cameras.txt"),
        osp.join(sfm_dir, "images.txt"),
        rgb_dir,
        mask_dir,
    ]
    if stream == "dslr":
        required.append(osp.join(sfm_dir, "points3D.txt"))
    for path in required:
        if not osp.exists(path):
            raise FileNotFoundError(path)

    cameras = load_colmap_cameras(osp.join(sfm_dir, "cameras.txt"))
    colmap_frames = load_colmap_images(
        osp.join(sfm_dir, "images.txt"), cameras, stream
    )
    colmap_frame_ids = {item.frame_id for item in colmap_frames}

    if stream == "iphone":
        # ScanNet++ supplies a per-frame ARKit pose transformed into the aligned
        # scan world.  This covers the complete video, unlike SfM/COLMAP, which
        # deliberately filters a small subset of frames.  Use aligned_pose for
        # strict 0,30,60,... sampling while retaining COLMAP's calibrated
        # OPENCV distortion model for image undistortion.
        pose_path = osp.join(scene_dir, "iphone", "pose_intrinsic_imu.json")
        if not osp.isfile(pose_path):
            raise FileNotFoundError(pose_path)
        with open(pose_path, "r", encoding="utf-8") as handle:
            pose_metadata = json.load(handle)
        if not isinstance(pose_metadata, dict) or not pose_metadata:
            raise ValueError(f"Invalid per-frame iPhone metadata: {pose_path}")

        used_camera_ids = [item.camera_id for item in colmap_frames]
        camera_id = max(set(used_camera_ids), key=used_camera_ids.count)
        calibrated_intrinsics = cameras[camera_id]
        selected = []
        candidate_ids = []
        missing_aligned_pose = []
        for key, metadata in pose_metadata.items():
            match = re.match(r"^frame_(\d+)$", key)
            if match is None:
                continue
            current_id = int(match.group(1))
            if (current_id - iphone_offset) % iphone_stride != 0:
                continue
            candidate_ids.append(current_id)
            if not isinstance(metadata, dict) or "aligned_pose" not in metadata:
                missing_aligned_pose.append(current_id)
                continue
            pose = np.asarray(metadata["aligned_pose"], dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                missing_aligned_pose.append(current_id)
                continue
            rotation = np.asarray(pose[:3, :3], dtype=np.float64)
            determinant = float(np.linalg.det(rotation))
            if (
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3)
                or not 0.95 < determinant < 1.05
            ):
                missing_aligned_pose.append(current_id)
                continue
            pose = pose.copy()
            pose[3] = np.asarray([0, 0, 0, 1], dtype=np.float32)
            selected.append(
                FrameInfo(
                    source_name=key + ".jpg",
                    frame_id=current_id,
                    camera_id=camera_id,
                    intrinsics=calibrated_intrinsics,
                    camera_pose=pose,
                    timestamp=float(metadata["timestamp"]) if "timestamp" in metadata else None,
                    pose_source="aligned_pose",
                )
            )
        if missing_aligned_pose:
            raise ValueError(
                f"{len(missing_aligned_pose)} sampled iPhone frames lack a valid aligned_pose; "
                f"first ids: {sorted(missing_aligned_pose)[:10]}"
            )
        selected.sort(key=lambda item: (item.frame_id, item.source_name))
        wanted_ids = set(candidate_ids)
        physical_count = len(pose_metadata)
        missing_colmap = wanted_ids - colmap_frame_ids
        segment_stats: Dict[str, Any] = {}
    else:
        physical = list_source_jpegs(rgb_dir, stream)
        wanted_names = set(physical)
        colmap_names = {item.source_name for item in colmap_frames}
        missing_colmap = wanted_names - colmap_names
        if missing_colmap and not allow_missing_colmap:
            preview = sorted(missing_colmap)[:10]
            raise ValueError(
                f"DSLR has {len(missing_colmap)} physical images without COLMAP pose; "
                f"first names: {preview}. Use --allow_missing_colmap to process only posed images."
            )
        selected = [item for item in colmap_frames if item.source_name in wanted_names]
        physical_count = len(physical)
        selected, segment_stats = select_longest_dslr_segment(
            selected,
            osp.join(sfm_dir, "points3D.txt"),
            target_width,
            target_max_height,
            dslr_min_segment_frames,
            dslr_min_shared_points,
            dslr_min_overlap_score,
            dslr_max_translation,
            dslr_rotation_fov_fraction,
        )
        wanted_ids = wanted_names

    missing_files = []
    for item in selected:
        rgb_path, mask_path = source_paths(rgb_dir, mask_dir, item.source_name)
        if not osp.isfile(rgb_path) or not osp.isfile(mask_path):
            missing_files.append((rgb_path, mask_path))
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} selected {stream} frames are missing RGB/mask files; "
            f"first: {missing_files[0]}"
        )

    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError(f"No frames selected for {osp.basename(scene_dir)}/{stream}")

    stats = {
        "physical_images": physical_count,
        "colmap_images": len(colmap_frames),
        "wanted_images": len(wanted_ids),
        "wanted_without_colmap": len(missing_colmap),
        "selected_images": len(selected),
    }
    stats.update(segment_stats)
    return selected, stats


def undistort_image_and_mask(
    intrinsics: Tuple[Any, ...], rgb: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_type = str(intrinsics[0])
    width, height = int(intrinsics[1]), int(intrinsics[2])
    fx, fy, cx, cy = map(float, intrinsics[3:7])
    distortion = np.asarray(intrinsics[7:], dtype=np.float64)

    if rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB shape {rgb.shape[:2]} disagrees with COLMAP {(height, width)}"
        )
    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"Mask shape {mask.shape[:2]} disagrees with COLMAP {(height, width)}"
        )

    k_colmap = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    k_opencv = colmap_to_opencv_intrinsics(k_colmap)

    if camera_type == "OPENCV_FISHEYE":
        if distortion.size != 4:
            raise ValueError(
                f"OPENCV_FISHEYE expects 4 coefficients, got {distortion.size}"
            )
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k_opencv,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        new_k[0, 2] = width / 2.0
        new_k[1, 2] = height / 2.0
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            k_opencv,
            distortion,
            np.eye(3),
            new_k,
            (width, height),
            cv2.CV_32FC1,
        )
    elif camera_type in {"OPENCV", "FULL_OPENCV"}:
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            k_opencv,
            distortion,
            (width, height),
            1,
            (width, height),
            True,
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            k_opencv,
            distortion,
            np.eye(3),
            new_k,
            (width, height),
            cv2.CV_32FC1,
        )
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {camera_type}")

    undistorted_rgb = cv2.remap(
        rgb,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    undistorted_mask = cv2.remap(
        mask,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return undistorted_rgb, undistorted_mask, np.asarray(new_k, dtype=np.float64)


def rescale_image_and_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    target_width: int,
    target_max_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_height, source_width = rgb.shape[:2]
    resized_k, output_shape, crop_top = resized_cropped_geometry(
        intrinsics,
        source_width,
        source_height,
        target_width,
        target_max_height,
    )
    output_height, output_width = output_shape
    height_before_crop = max(
        1, int(round(float(source_height) * output_width / float(source_width)))
    )
    image_interpolation = cv2.INTER_AREA if output_width < source_width else cv2.INTER_LINEAR
    resized_rgb = cv2.resize(
        rgb, (output_width, height_before_crop), interpolation=image_interpolation
    )
    resized_mask = cv2.resize(
        mask, (output_width, height_before_crop), interpolation=cv2.INTER_LINEAR
    )
    if height_before_crop > output_height:
        resized_rgb = resized_rgb[crop_top : crop_top + output_height]
        resized_mask = resized_mask[crop_top : crop_top + output_height]
    return (
        np.asarray(resized_rgb, dtype=np.uint8),
        np.asarray(resized_mask),
        np.asarray(resized_k, dtype=np.float32),
    )


def expected_output_shape(
    intrinsics: Tuple[Any, ...], target_width: int, target_max_height: int
) -> Tuple[int, int]:
    width, height = int(intrinsics[1]), int(intrinsics[2])
    resized_height = max(1, int(round(float(height) * target_width / float(width))))
    return min(resized_height, int(target_max_height)), int(target_width)


class MeshDepthRenderer:
    def __init__(self, mesh_path: str, platform: str, znear: float, zfar: float):
        if platform:
            os.environ["PYOPENGL_PLATFORM"] = platform
        import pyrender
        import trimesh
        import trimesh.exchange.ply

        self.pyrender = pyrender
        self.znear = znear
        self.zfar = zfar
        with open(mesh_path, "rb") as handle:
            mesh_kwargs = trimesh.exchange.ply.load_ply(handle)
        mesh_scene = trimesh.Trimesh(**mesh_kwargs)
        mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
        self.scene = pyrender.Scene()
        self.scene.add(mesh)
        self.renderer = pyrender.OffscreenRenderer(1, 1)

    def render(
        self, intrinsics_opencv: np.ndarray, camera_pose: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        k_render = opencv_to_colmap_intrinsics(intrinsics_opencv)
        self.renderer.viewport_width = int(width)
        self.renderer.viewport_height = int(height)
        camera = self.pyrender.camera.IntrinsicsCamera(
            float(k_render[0, 0]),
            float(k_render[1, 1]),
            float(k_render[0, 2]),
            float(k_render[1, 2]),
            znear=self.znear,
            zfar=self.zfar,
        )
        node = self.scene.add(
            camera,
            pose=np.asarray(camera_pose, dtype=np.float32) @ OPENGL_TO_OPENCV,
        )
        try:
            depth = self.renderer.render(
                self.scene, flags=self.pyrender.RenderFlags.DEPTH_ONLY
            )
        finally:
            self.scene.remove_node(node)
        return np.asarray(depth, dtype=np.float32)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.renderer.delete()


def build_invalid_mask(
    depth: np.ndarray,
    anonymization_mask: np.ndarray,
    max_depth: float,
) -> np.ndarray:
    invalid = (
        (anonymization_mask < 255)
        | ~np.isfinite(depth)
        | (depth <= 0.0)
        | (depth > max_depth)
    )
    return np.asarray(invalid, dtype=np.bool_)


def output_paths(sequence_dir: str, local_index: int) -> Tuple[str, str, str, str]:
    stem = f"{local_index:05d}"
    return (
        osp.join(sequence_dir, "images", stem + ".jpg"),
        osp.join(sequence_dir, "depths", stem + ".npy"),
        osp.join(sequence_dir, "masks", stem + ".npy"),
        osp.join(sequence_dir, "cameras", stem + ".npz"),
    )


def validate_frame_outputs(
    paths: Tuple[str, str, str, str],
    expected_shape: Tuple[int, int],
    deep: bool,
) -> Tuple[bool, str]:
    image_path, depth_path, mask_path, camera_path = paths
    for path in paths:
        try:
            if osp.getsize(path) <= 0:
                return False, f"empty file: {path}"
        except OSError as exc:
            return False, str(exc)
    try:
        with Image.open(image_path) as image:
            if image.size != expected_shape[::-1]:
                return False, f"image shape {image.size[::-1]} != {expected_shape}"
            if image.mode != "RGB":
                return False, f"image mode is {image.mode}, expected RGB"
            if deep:
                image.load()

        depth = np.load(depth_path, mmap_mode=None if deep else "r", allow_pickle=False)
        if depth.dtype != np.float32 or depth.shape != expected_shape:
            return False, f"depth dtype/shape is {depth.dtype}/{depth.shape}"
        if deep and not np.isfinite(depth).all():
            return False, "depth contains NaN/Inf"

        mask = np.load(mask_path, mmap_mode=None if deep else "r", allow_pickle=False)
        if mask.dtype != np.bool_ or mask.shape != expected_shape:
            return False, f"mask dtype/shape is {mask.dtype}/{mask.shape}"

        with np.load(camera_path, allow_pickle=False) as camera:
            if set(camera.files) != {"camera_intrinsics", "camera_pose"}:
                return False, f"unexpected camera keys: {camera.files}"
            k = camera["camera_intrinsics"]
            pose = camera["camera_pose"]
        if k.dtype != np.float32 or k.shape != (3, 3):
            return False, f"invalid intrinsics dtype/shape: {k.dtype}/{k.shape}"
        if pose.dtype != np.float32 or pose.shape != (4, 4):
            return False, f"invalid pose dtype/shape: {pose.dtype}/{pose.shape}"
        if not np.isfinite(k).all() or not np.isfinite(pose).all():
            return False, "camera contains NaN/Inf"
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            return False, "invalid camera intrinsic values"
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
            return False, "invalid C2W homogeneous row"
    except Exception as exc:
        return False, f"validation error: {exc}"
    return True, ""


def validate_sequence_outputs(
    sequence_dir: str,
    frames: Sequence[FrameInfo],
    deep: bool,
) -> Tuple[bool, str]:
    target_width, target_max_height = _manifest_target_geometry(sequence_dir)
    for local_index, frame in enumerate(frames):
        valid, reason = validate_frame_outputs(
            output_paths(sequence_dir, local_index),
            expected_output_shape(
                frame.intrinsics, target_width, target_max_height
            ),
            deep,
        )
        if not valid:
            return False, f"frame {local_index:05d}: {reason}"
    return True, ""


def _manifest_target_geometry(sequence_dir: str) -> Tuple[int, int]:
    manifest = read_json(osp.join(sequence_dir, "sequence_manifest.json"))
    config = manifest["config"]
    return int(config["target_width"]), int(config["target_max_height"])


def sequence_config(
    scene_id: str,
    stream: str,
    frames: Sequence[FrameInfo],
    selection_stats: Dict[str, Any],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    selection_hasher = hashlib.sha256()
    for item in frames:
        selection_hasher.update(f"{item.frame_id}\t{item.source_name}\t{item.pose_source}\n".encode("utf-8"))
        selection_hasher.update(np.asarray(item.camera_pose, dtype=np.float32).tobytes())
        selection_hasher.update(stable_json_bytes(list(item.intrinsics)))
    selection_hash = selection_hasher.hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "scene_id": scene_id,
        "stream": stream,
        "pose_source": frames[0].pose_source,
        "selection_hash": selection_hash,
        "frame_count": len(frames),
        "first_source_frame_id": int(frames[0].frame_id),
        "last_source_frame_id": int(frames[-1].frame_id),
        "iphone_stride": int(args["iphone_stride"]),
        "iphone_offset": int(args["iphone_offset"]),
        "target_width": int(args["target_width"]),
        "target_max_height": int(args["target_max_height"]),
        "dslr_min_segment_frames": int(args["dslr_min_segment_frames"]),
        "dslr_min_shared_points": int(args["dslr_min_shared_points"]),
        "dslr_min_overlap_score": float(args["dslr_min_overlap_score"]),
        "dslr_max_translation": float(args["dslr_max_translation"]),
        "dslr_rotation_fov_fraction": float(args["dslr_rotation_fov_fraction"]),
        "max_depth": float(args["max_depth"]),
        "depth_percentile": args["depth_percentile"],
        "znear": float(args["znear"]),
        "zfar": float(args["zfar"]),
        "jpeg_quality": int(args["jpeg_quality"]),
        "max_frames": args["max_frames"],
        "selection_stats": selection_stats,
    }


def sequence_fingerprint(config: Dict[str, Any]) -> str:
    return sha256_json(config)


def write_sequence_manifest(
    sequence_dir: str,
    config: Dict[str, Any],
    frames: Sequence[FrameInfo],
) -> str:
    fingerprint = sequence_fingerprint(config)
    manifest_path = osp.join(sequence_dir, "sequence_manifest.json")
    manifest = {
        "fingerprint": fingerprint,
        "config": config,
        "source_frames": [
            {
                "output_index": i,
                "source_frame_id": int(item.frame_id),
                "source_name": item.source_name,
                "timestamp": item.timestamp,
                "pose_source": item.pose_source,
                "colmap_image_id": item.colmap_image_id,
                "capture_prefix": item.capture_prefix,
            }
            for i, item in enumerate(frames)
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return fingerprint


def remove_numeric_outputs(sequence_dir: str) -> int:
    removed = 0
    for folder in ("images", "depths", "masks", "cameras"):
        folder_path = osp.join(sequence_dir, folder)
        if not osp.isdir(folder_path):
            continue
        for entry in os.scandir(folder_path):
            if entry.is_file() and NUMERIC_OUTPUT_RE.match(entry.name):
                os.unlink(entry.path)
                removed += 1
    for name in (".complete.json", "sequence_metadata.npz"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(osp.join(sequence_dir, name))
    return removed


def output_size_bytes(sequence_dir: str, frame_count: int) -> int:
    total = 0
    for index in range(frame_count):
        for path in output_paths(sequence_dir, index):
            total += osp.getsize(path)
    for name in ("sequence_manifest.json", "sequence_metadata.npz"):
        path = osp.join(sequence_dir, name)
        if osp.isfile(path):
            total += osp.getsize(path)
    return total


def prepare_sequence(
    scene_dir: str,
    scene_id: str,
    stream: str,
    output_root: str,
    args: Dict[str, Any],
) -> Tuple[str, List[FrameInfo], Dict[str, Any], str, Dict[str, Any]]:
    frames, selection_stats = select_frames(
        scene_dir=scene_dir,
        stream=stream,
        iphone_stride=int(args["iphone_stride"]),
        iphone_offset=int(args["iphone_offset"]),
        target_width=int(args["target_width"]),
        target_max_height=int(args["target_max_height"]),
        dslr_min_segment_frames=int(args["dslr_min_segment_frames"]),
        dslr_min_shared_points=int(args["dslr_min_shared_points"]),
        dslr_min_overlap_score=float(args["dslr_min_overlap_score"]),
        dslr_max_translation=float(args["dslr_max_translation"]),
        dslr_rotation_fov_fraction=float(args["dslr_rotation_fov_fraction"]),
        max_frames=args["max_frames"],
        allow_missing_colmap=bool(args["allow_missing_colmap"]),
    )
    sequence_dir = osp.join(output_root, sequence_id(scene_id, stream))
    os.makedirs(sequence_dir, exist_ok=True)
    for folder in ("images", "depths", "masks", "cameras"):
        os.makedirs(osp.join(sequence_dir, folder), exist_ok=True)

    config = sequence_config(scene_id, stream, frames, selection_stats, args)
    fingerprint = sequence_fingerprint(config)
    manifest_path = osp.join(sequence_dir, "sequence_manifest.json")
    write_manifest = not osp.exists(manifest_path)
    if osp.exists(manifest_path):
        try:
            previous = read_json(manifest_path)
        except Exception as exc:
            if not args["force"]:
                raise RuntimeError(
                    f"Unreadable historical manifest for {scene_id}/{stream}: {exc}; "
                    "use --force to rebuild this derived sequence"
                ) from exc
            remove_numeric_outputs(sequence_dir)
            write_manifest = True
        else:
            if previous.get("fingerprint") != fingerprint:
                if not args["force"]:
                    raise RuntimeError(
                        f"Historical fingerprint mismatch for {scene_id}/{stream}; "
                        "use the original parameters or pass --force"
                    )
                remove_numeric_outputs(sequence_dir)
                write_manifest = True
    elif any(directory_has_entries(osp.join(sequence_dir, folder)) for folder in ("images", "depths", "masks", "cameras")):
        if not args["force"]:
            raise RuntimeError(
                f"Historical outputs exist without a manifest: {sequence_dir}; use --force"
            )
        remove_numeric_outputs(sequence_dir)

    if write_manifest:
        if args["validate_only"]:
            raise RuntimeError(f"Missing historical manifest: {manifest_path}")
        write_sequence_manifest(sequence_dir, config, frames)
    return sequence_dir, frames, config, fingerprint, selection_stats


def directory_has_entries(path: str) -> bool:
    if not osp.isdir(path):
        return False
    with os.scandir(path) as entries:
        return next(entries, None) is not None


def completed_marker_matches(
    sequence_dir: str,
    fingerprint: str,
    frame_count: int,
) -> bool:
    path = osp.join(sequence_dir, ".complete.json")
    if not osp.isfile(path):
        return False
    try:
        marker = read_json(path)
    except Exception:
        return False
    marker_matches = (
        marker.get("fingerprint") == fingerprint
        and int(marker.get("frame_count", -1)) == frame_count
    )
    return marker_matches and sequence_metadata_valid(sequence_dir, frame_count)


def sequence_metadata_valid(sequence_dir: str, frame_count: int) -> bool:
    path = osp.join(sequence_dir, "sequence_metadata.npz")
    if not osp.isfile(path):
        return False
    required = {
        "frame_indices",
        "source_frame_indices",
        "source_image_names",
        "timestamps",
        "pose_sources",
        "scene_id",
        "stream",
    }
    try:
        with np.load(path, allow_pickle=False) as metadata:
            if not required.issubset(metadata.files):
                return False
            indices = np.asarray(metadata["frame_indices"], dtype=np.int32)
            if not np.array_equal(indices, np.arange(frame_count, dtype=np.int32)):
                return False
            for key in (
                "source_frame_indices",
                "source_image_names",
                "timestamps",
                "pose_sources",
            ):
                if len(metadata[key]) != frame_count:
                    return False
    except Exception:
        return False
    return True


def write_sequence_metadata(
    sequence_dir: str,
    scene_id: str,
    stream: str,
    frames: Sequence[FrameInfo],
) -> None:
    frame_count = len(frames)
    atomic_save_npz(
        osp.join(sequence_dir, "sequence_metadata.npz"),
        frame_indices=np.arange(frame_count, dtype=np.int32),
        source_frame_indices=np.asarray([item.frame_id for item in frames], dtype=np.int64),
        source_image_names=np.asarray([item.source_name for item in frames]),
        timestamps=np.asarray(
            [np.nan if item.timestamp is None else item.timestamp for item in frames],
            dtype=np.float64,
        ),
        pose_sources=np.asarray([item.pose_source for item in frames]),
        scene_id=np.asarray(scene_id),
        stream=np.asarray(stream),
    )


def process_one_sequence(
    scene_dir: str,
    scene_id: str,
    stream: str,
    output_root: str,
    args: Dict[str, Any],
    renderer: Optional[MeshDepthRenderer],
    prepared: Optional[
        Tuple[str, List[FrameInfo], Dict[str, Any], str, Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    if prepared is None:
        prepared = prepare_sequence(scene_dir, scene_id, stream, output_root, args)
    sequence_dir, frames, config, fingerprint, selection_stats = prepared
    complete = completed_marker_matches(sequence_dir, fingerprint, len(frames))

    # A force rewrite must invalidate the O(1) completion signal before the
    # first frame is touched.  Otherwise a crash during rewriting could leave
    # a stale marker that a later normal resume would incorrectly trust.
    if args["force"] and not args["validate_only"]:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(osp.join(sequence_dir, ".complete.json"))
        complete = False

    if args["validate_only"]:
        if not complete:
            raise RuntimeError("sequence has no matching completion marker")
        valid, reason = validate_sequence_outputs(sequence_dir, frames, bool(args["verify"]))
        if not valid:
            raise RuntimeError(reason)
        return {
            "scene_id": scene_id,
            "stream": stream,
            "status": "validated",
            "frame_count": len(frames),
            "processed": 0,
            "skipped": len(frames),
            "repaired": 0,
            "selection_stats": selection_stats,
        }

    if complete and not args["verify"] and not args["force"]:
        return {
            "scene_id": scene_id,
            "stream": stream,
            "status": "complete_skip",
            "frame_count": len(frames),
            "processed": 0,
            "skipped": len(frames),
            "repaired": 0,
            "selection_stats": selection_stats,
        }

    if complete and args["verify"] and not args["force"]:
        valid, reason = validate_sequence_outputs(sequence_dir, frames, True)
        if valid:
            return {
                "scene_id": scene_id,
                "stream": stream,
                "status": "verified_skip",
                "frame_count": len(frames),
                "processed": 0,
                "skipped": len(frames),
                "repaired": 0,
                "selection_stats": selection_stats,
            }
        with contextlib.suppress(FileNotFoundError):
            os.unlink(osp.join(sequence_dir, ".complete.json"))
        print(f"[{scene_id}/{stream}] verification found damage: {reason}; repairing")

    if renderer is None:
        raise RuntimeError("renderer was not initialized")

    _, rgb_dir, mask_dir = source_dirs(scene_dir, stream)
    processed = 0
    skipped = 0
    repaired = 0
    output_shapes = []

    for local_index, frame in enumerate(frames):
        paths = output_paths(sequence_dir, local_index)
        expected_shape = expected_output_shape(
            frame.intrinsics,
            int(args["target_width"]),
            int(args["target_max_height"]),
        )
        output_shapes.append(expected_shape)
        had_partial = any(osp.isfile(path) for path in paths)
        if not args["force"]:
            valid, _ = validate_frame_outputs(paths, expected_shape, bool(args["verify"]))
            if valid:
                skipped += 1
                continue
        if had_partial:
            repaired += 1

        rgb_path, source_mask_path = source_paths(rgb_dir, mask_dir, frame.source_name)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(source_mask_path) as image:
            source_mask = np.asarray(image.convert("L"), dtype=np.uint8)

        rgb, source_mask, k_opencv = undistort_image_and_mask(
            frame.intrinsics, rgb, source_mask
        )
        rgb, source_mask, k_opencv = rescale_image_and_mask(
            rgb,
            source_mask,
            k_opencv,
            int(args["target_width"]),
            int(args["target_max_height"]),
        )
        height, width = rgb.shape[:2]
        if (height, width) != expected_shape:
            raise RuntimeError(
                f"Unexpected resized shape {(height, width)} != {expected_shape}"
            )
        depth = renderer.render(k_opencv, frame.camera_pose, width, height)
        depth[~np.isfinite(depth)] = 0.0
        invalid_mask = build_invalid_mask(
            depth,
            source_mask,
            float(args["max_depth"]),
        )
        depth[invalid_mask] = 0.0

        atomic_save_jpeg(paths[0], rgb, int(args["jpeg_quality"]))
        atomic_save_npy(paths[1], np.asarray(depth, dtype=np.float32))
        atomic_save_npy(paths[2], invalid_mask)
        atomic_save_npz(
            paths[3],
            camera_intrinsics=np.asarray(k_opencv, dtype=np.float32),
            camera_pose=np.asarray(frame.camera_pose, dtype=np.float32),
        )
        valid, reason = validate_frame_outputs(paths, expected_shape, False)
        if not valid:
            raise RuntimeError(f"Fresh frame {local_index:05d} failed validation: {reason}")
        processed += 1

    write_sequence_metadata(sequence_dir, scene_id, stream, frames)
    size_bytes = output_size_bytes(sequence_dir, len(frames))
    marker = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "scene_id": scene_id,
        "stream": stream,
        "fingerprint": fingerprint,
        "frame_count": len(frames),
        "output_bytes": size_bytes,
        "completed_at": utc_now(),
        "config": config,
    }
    atomic_write_json(osp.join(sequence_dir, ".complete.json"), marker)
    return {
        "scene_id": scene_id,
        "stream": stream,
        "status": "processed",
        "frame_count": len(frames),
        "processed": processed,
        "skipped": skipped,
        "repaired": repaired,
        "output_bytes": size_bytes,
        "selection_stats": selection_stats,
    }


def process_scene_task(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    scene_id = task["scene_id"]
    scene_dir = osp.join(task["scannetpp_dir"], "data", scene_id)
    streams: List[str] = task["streams"]
    args = task["args"]
    output_root = task["output_root"]
    egl_devices = [str(device) for device in args.get("egl_devices", ["0"])]
    process_identity = mp.current_process()._identity
    worker_ordinal = int(process_identity[0] - 1) if process_identity else 0
    egl_device_id = egl_devices[worker_ordinal % len(egl_devices)]
    if str(args["pyopengl_platform"]).lower() == "egl":
        os.environ["EGL_DEVICE_ID"] = egl_device_id
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    renderer: Optional[MeshDepthRenderer] = None
    prepared: Dict[
        str, Tuple[str, List[FrameInfo], Dict[str, Any], str, Dict[str, Any]]
    ] = {}
    filtered_streams = set()

    try:
        if not osp.isdir(scene_dir):
            raise FileNotFoundError(scene_dir)

        # Prepare first to detect completed/invalid history before loading the mesh.
        pending = False
        for stream in streams:
            try:
                prepared[stream] = prepare_sequence(
                    scene_dir, scene_id, stream, output_root, args
                )
                sequence_dir, frames, _, fingerprint, _ = prepared[stream]
                if args["validate_only"] or args["verify"] or args["force"]:
                    pending = pending or not args["validate_only"]
                elif not completed_marker_matches(sequence_dir, fingerprint, len(frames)):
                    pending = True
            except SequenceFiltered as exc:
                filtered_streams.add(stream)
                results.append(
                    {
                        "scene_id": scene_id,
                        "stream": stream,
                        "status": "filtered_short",
                        "frame_count": 0,
                        "processed": 0,
                        "skipped": 0,
                        "repaired": 0,
                        "reason": str(exc),
                        "selection_stats": exc.stats,
                    }
                )
            except Exception as exc:
                failures.append(
                    {"scene_id": scene_id, "stream": stream, "error": str(exc)}
                )

        if pending and prepared:
            mesh_path = osp.join(scene_dir, "scans", "mesh_aligned_0.05.ply")
            if not osp.isfile(mesh_path):
                raise FileNotFoundError(mesh_path)
            renderer = MeshDepthRenderer(
                mesh_path=mesh_path,
                platform=str(args["pyopengl_platform"]),
                znear=float(args["znear"]),
                zfar=float(args["zfar"]),
            )

        failed_streams = {item["stream"] for item in failures}
        for stream in streams:
            if stream in failed_streams or stream in filtered_streams:
                continue
            try:
                stream_result = process_one_sequence(
                    scene_dir,
                    scene_id,
                    stream,
                    output_root,
                    args,
                    renderer,
                    prepared=prepared[stream],
                )
                stream_result["egl_device_id"] = egl_device_id
                results.append(stream_result)
            except Exception as exc:
                failures.append(
                    {
                        "scene_id": scene_id,
                        "stream": stream,
                        "error": f"{exc}\n{traceback.format_exc(limit=12)}",
                    }
                )
    except Exception as exc:
        failures.append(
            {
                "scene_id": scene_id,
                "stream": "scene",
                "error": f"{exc}\n{traceback.format_exc(limit=12)}",
            }
        )
    finally:
        if renderer is not None:
            renderer.close()

    return {"scene_id": scene_id, "results": results, "failures": failures}


def load_train_scenes(scannetpp_dir: str, split_file: Optional[str]) -> Tuple[List[str], str]:
    if split_file is None:
        split_file = osp.join(scannetpp_dir, "splits", "nvs_sem_train.txt")
    if not osp.isfile(split_file):
        raise FileNotFoundError(split_file)
    with open(split_file, "r", encoding="utf-8") as handle:
        scenes = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    scenes = list(dict.fromkeys(scenes))
    if not scenes:
        raise ValueError(f"No scenes in split file: {split_file}")
    return scenes, split_file


def collect_complete_metadata(output_root: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    compact: Dict[str, np.ndarray] = {}
    rich: Dict[str, Any] = {}
    if not osp.isdir(output_root):
        return compact, rich
    for entry in sorted(os.scandir(output_root), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        complete_path = osp.join(entry.path, ".complete.json")
        metadata_path = osp.join(entry.path, "sequence_metadata.npz")
        manifest_path = osp.join(entry.path, "sequence_manifest.json")
        if not all(osp.isfile(path) for path in (complete_path, metadata_path, manifest_path)):
            continue
        try:
            marker = read_json(complete_path)
            manifest = read_json(manifest_path)
            if marker.get("fingerprint") != manifest.get("fingerprint"):
                continue
            stream = str(marker["stream"])
            scene_id_value = str(marker["scene_id"])
            if stream not in SEQUENCE_STREAMS:
                continue
            if entry.name != sequence_id(scene_id_value, stream):
                continue
            frame_count = int(marker["frame_count"])
            if not sequence_metadata_valid(entry.path, frame_count):
                continue
            with np.load(metadata_path, allow_pickle=False) as metadata:
                indices = np.asarray(metadata["frame_indices"], dtype=np.int32)
                source_indices = np.asarray(metadata["source_frame_indices"], dtype=np.int64)
                timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
            if len(indices) != frame_count or not np.array_equal(
                indices, np.arange(frame_count, dtype=np.int32)
            ):
                continue
        except Exception:
            continue
        compact[entry.name] = indices
        rich[entry.name] = {
            "frame_indices": indices,
            "source_frame_indices": source_indices,
            "timestamps": timestamps,
            "data_path": entry.name,
            "stream": stream,
            "scene_id": scene_id_value,
        }
    return compact, rich


def generate_metadata(output_root: str) -> Dict[str, Any]:
    compact, rich = collect_complete_metadata(output_root)
    atomic_save_npz(osp.join(output_root, "all_metadata.npz"), **compact)
    atomic_save_pickle(osp.join(output_root, "all_metadata.pkl"), rich)
    summary: Dict[str, Any] = {
        "sequences": len(compact),
        "frames": int(sum(len(value) for value in compact.values())),
    }
    for stream in SEQUENCE_STREAMS:
        stream_keys = [key for key, value in rich.items() if value["stream"] == stream]
        summary[stream] = {
            "sequences": len(stream_keys),
            "frames": int(sum(len(compact[key]) for key in stream_keys)),
        }
    return summary


def aggregate_completed_stats(output_root: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {
        stream: {"sequences": 0, "frames": 0, "bytes": 0, "scene_ids": []}
        for stream in SEQUENCE_STREAMS
    }
    if not osp.isdir(output_root):
        return result
    for entry in os.scandir(output_root):
        marker_path = osp.join(entry.path, ".complete.json") if entry.is_dir() else ""
        if not marker_path or not osp.isfile(marker_path):
            continue
        try:
            marker = read_json(marker_path)
            stream = str(marker["stream"])
            scene_id_value = str(marker["scene_id"])
            if stream not in SEQUENCE_STREAMS:
                continue
            if entry.name != sequence_id(scene_id_value, stream):
                continue
            result[stream]["sequences"] += 1
            result[stream]["frames"] += int(marker.get("frame_count", 0))
            result[stream]["bytes"] += int(marker.get("output_bytes", 0))
            result[stream]["scene_ids"].append(scene_id_value)
        except Exception:
            continue
    for stream in SEQUENCE_STREAMS:
        result[stream]["scene_ids"] = sorted(set(result[stream]["scene_ids"]))
    return result


def render_readme(output_root: str, stats: Dict[str, Dict[str, Any]], args: Dict[str, Any]) -> None:
    total_sequences = sum(item["sequences"] for item in stats.values())
    total_frames = sum(item["frames"] for item in stats.values())
    total_bytes = sum(item["bytes"] for item in stats.values())
    average = total_frames / total_sequences if total_sequences else 0.0
    physical_scenes = len(
        set(stats["iphone"]["scene_ids"]) | set(stats["dslr"]["scene_ids"])
    )
    percentile_text = (
        "关闭" if args["depth_percentile"] is None else f"最远 {100.0 - float(args['depth_percentile']):g}%"
    )
    readme = f"""# ScanNet++ sequential preprocessing

## 1.1 数据集简介

本目录由 ScanNet++ 训练集生成。每个原始场景被拆成互不混合的 iPhone 视频流序列与 DSLR 拍摄序列；两者共享同一个 ScanNet++ 对齐世界坐标系。iPhone 默认严格按原始视频帧号每 {args['iphone_stride']} 帧采一张，位姿使用官方逐帧 `aligned_pose`，图像去畸变使用官方 COLMAP 标定；DSLR 按拍摄前缀与编号排序，以 COLMAP 共视、相邻平移和相邻旋转切段，只保留最长且不少于 {args['dslr_min_segment_frames']} 帧的一段。

## 1.2 统计摘要

- 场景类型：真实室内场景
- 已完成物理场景数量：{physical_scenes}（完整训练集应为 856）
- 已完成序列数量：{total_sequences}（iPhone {stats['iphone']['sequences']}，DSLR {stats['dslr']['sequences']}）
- 总图像数：{total_frames}（iPhone {stats['iphone']['frames']}，DSLR {stats['dslr']['frames']}）
- 平均每个序列图像数：{average:.2f}
- 已完成数据量：{total_bytes / (1024 ** 3):.3f} GiB
- 采集设备：iPhone、DSLR；深度来自对齐激光扫描 mesh 的离屏渲染
- 采集场景：住宅、办公室及其他真实室内空间

## 2. 目录结构

```text
dataset_root/
├── README.md
├── dataset_config.json
├── all_metadata.npz
├── all_metadata.pkl
├── <scene_id>_iphone/
│   ├── images/00000.jpg
│   ├── depths/00000.npy
│   ├── masks/00000.npy
│   ├── cameras/00000.npz
│   ├── sequence_metadata.npz
│   ├── sequence_manifest.json
│   └── .complete.json
├── <scene_id>_dslr/
│   └── ...
└── ...
```

## 3. 数据模态详解

### 3.1 元数据 (Metadata)

- 根目录只有一份 `all_metadata.npz`：key 为序列目录名（例如 `<scene_id>_iphone`），value 为连续输出帧序号 `int32` 数组。
- 根目录只有一份 `all_metadata.pkl`：每条序列包含 `frame_indices`、`source_frame_indices`、`data_path`、`scene_id` 和 `stream`。
- `sequence_metadata.npz`：保存输出序号、原始帧号、原始图像名、iPhone 时间戳与位姿来源。
- `sequence_manifest.json`：保存源帧选择与处理参数指纹；`.complete.json` 只在整条序列完成后生成，用于断点续跑及历史结果校验。
- 根目录 `dataset_config.json`：锁定会影响输出的统一参数，阻止不同分批任务把不同配置混入同一个数据集。
- 模型训练时的序列 interval 应根据 iPhone 与 DSLR 两类序列分别设计；预处理不跨场景、也不混合两种设备。

### 3.2 图像数据 (Images)

- 格式：JPEG，RGB，命名 `{{frame_id:05d}}.jpg`。
- 已去畸变，并与深度、mask、相机内参联合缩放。
- 图像保持宽高比缩放到固定宽 `{args['target_width']}`；若缩放后高度超过 `{args['target_max_height']}`，仅做垂直居中裁切到 `{args['target_max_height']}`，相机内参同步缩放并减去裁切偏移。

### 3.3 深度图 (Depths)

- 格式：NumPy `.npy`，`float32`，与 RGB 同分辨率。
- 深度由 `mesh_aligned_0.05.ply` 渲染，不归一化；数值与 C2W 平移处于相同的米制尺度。
- 命名 `{{frame_id:05d}}.npy`。无效像素由 mask 控制。

### 3.4 掩码 (Masks)

- 格式：NumPy `.npy`，`bool`，与 RGB 同分辨率；`True` 表示无效。
- 无效条件：匿名化区域、深度 NaN/Inf、深度小于等于 0、深度超过 {args['max_depth']} m，以及统计过滤（{percentile_text}）。
- ScanNet++ 原始数据没有可直接复用的天空/动态物体/水面语义 mask，因此这些类别只有在上述几何或匿名化条件触发时才会被屏蔽。

### 3.5 相机参数 (Camera Parameters)

- 每帧一个压缩 NPZ，包含 `camera_intrinsics` (`float32`, 3x3) 和 `camera_pose` (`float32`, 4x4)。
- 内参采用 OpenCV 像素中心约定；相机坐标为 RDF 右手系（x 右、y 下、z 前）。
- `camera_pose` 为 `T_cam_to_world` (C2W)，所有相机共享原始 ScanNet++ 场景的对齐世界坐标系。
"""
    atomic_write_text(osp.join(output_root, "README.md"), readme)


@contextlib.contextmanager
def dataset_lock(output_root: str) -> Iterable[None]:
    os.makedirs(output_root, exist_ok=True)
    lock_path = osp.join(output_root, ".preprocess_scannet_seq.lock")
    with open(lock_path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another preprocess_scannet_seq.py process holds {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess ScanNet++ train scenes as ordered iPhone/DSLR sequences."
    )
    parser.add_argument("--scannetpp_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--stream", choices=("both", "iphone", "dslr"), default="both")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--single_scene", default=None)
    parser.add_argument("--iphone_stride", type=int, default=30)
    parser.add_argument("--iphone_offset", type=int, default=0)
    parser.add_argument(
        "--target_width",
        "--target_resolution",
        dest="target_width",
        type=int,
        default=504,
        help="Exact output width; --target_resolution is kept as a compatibility alias.",
    )
    parser.add_argument("--target_max_height", type=int, default=280)
    parser.add_argument("--dslr_min_segment_frames", type=int, default=100)
    parser.add_argument("--dslr_min_shared_points", type=int, default=50)
    parser.add_argument("--dslr_min_overlap_score", type=float, default=0.05)
    parser.add_argument("--dslr_max_translation", type=float, default=1.0)
    parser.add_argument(
        "--dslr_rotation_fov_fraction",
        type=float,
        default=2.0 / 3.0,
        help="Maximum adjacent rotation as a fraction of the smaller final FOV.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_depth", type=float, default=40.0)
    parser.add_argument(
        "--depth_percentile",
        type=float,
        default=0.0,
        help="Compatibility option; percentile clipping was removed and only 0 is accepted.",
    )
    parser.add_argument("--znear", type=float, default=0.05)
    parser.add_argument("--zfar", type=float, default=40.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--pyopengl_platform", default="egl")
    parser.add_argument(
        "--egl_devices",
        default="0",
        help="Comma-separated EGL device ids distributed round-robin across workers.",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow_missing_colmap", action="store_true")
    parser.add_argument("--rebuild_metadata_only", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser


def normalize_args(namespace: argparse.Namespace) -> Tuple[Dict[str, Any], List[str], str]:
    args = vars(namespace).copy()
    if args["iphone_stride"] <= 0:
        raise ValueError("--iphone_stride must be positive")
    if (
        args["target_width"] <= 0
        or args["target_max_height"] <= 0
        or args["num_workers"] <= 0
    ):
        raise ValueError(
            "--target_width, --target_max_height and --num_workers must be positive"
        )
    if args["dslr_min_segment_frames"] <= 0 or args["dslr_min_shared_points"] < 0:
        raise ValueError("Invalid DSLR segment/shared-point thresholds")
    if not 0 <= args["dslr_min_overlap_score"] <= 1:
        raise ValueError("--dslr_min_overlap_score must be in [0, 1]")
    if args["dslr_max_translation"] <= 0:
        raise ValueError("--dslr_max_translation must be positive")
    if not 0 < args["dslr_rotation_fov_fraction"] <= 1:
        raise ValueError("--dslr_rotation_fov_fraction must be in (0, 1]")
    if not (1 <= args["jpeg_quality"] <= 100):
        raise ValueError("--jpeg_quality must be in [1, 100]")
    if args["max_depth"] <= 0 or args["znear"] <= 0 or args["zfar"] <= args["znear"]:
        raise ValueError("Invalid depth/near/far settings")
    if args["depth_percentile"] != 0:
        raise ValueError(
            "Per-frame depth percentile clipping has been removed; "
            "use --depth_percentile 0"
        )
    args["depth_percentile"] = None
    if args["single_scene"] and args["scenes"]:
        raise ValueError("Use only one of --single_scene and --scenes")
    if args["validate_only"] and args["force"]:
        raise ValueError("--validate_only and --force are mutually exclusive")
    try:
        args["egl_devices"] = [
            int(value.strip()) for value in str(args["egl_devices"]).split(",") if value.strip()
        ]
    except ValueError as exc:
        raise ValueError("--egl_devices must be comma-separated non-negative integers") from exc
    if not args["egl_devices"] or any(value < 0 for value in args["egl_devices"]):
        raise ValueError("--egl_devices must contain at least one non-negative id")

    scenes, split_file = load_train_scenes(args["scannetpp_dir"], args["split_file"])
    requested = [args["single_scene"]] if args["single_scene"] else args["scenes"]
    if requested:
        train_set = set(scenes)
        missing = [scene for scene in requested if scene not in train_set]
        if missing:
            raise ValueError(f"Requested scenes are not in the train split: {missing}")
        requested_set = set(requested)
        scenes = [scene for scene in scenes if scene in requested_set]
    return args, scenes, split_file


def write_run_summary(output_root: str, summary: Dict[str, Any]) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    atomic_write_json(osp.join(output_root, f"processing_summary_{stamp}.json"), summary)
    atomic_write_json(osp.join(output_root, "latest_summary.json"), summary)


def ensure_dataset_config(
    output_root: str, args: Dict[str, Any], split_file: str
) -> Dict[str, Any]:
    """Prevent separate invocations from silently mixing output parameters."""
    with open(split_file, "rb") as handle:
        split_sha256 = hashlib.sha256(handle.read()).hexdigest()
    config = {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "sequence_layout": SEQUENCE_LAYOUT,
        "scannetpp_dir": osp.realpath(args["scannetpp_dir"]),
        "split_file": osp.realpath(split_file),
        "split_sha256": split_sha256,
        "iphone_stride": args["iphone_stride"],
        "iphone_offset": args["iphone_offset"],
        "target_width": args["target_width"],
        "target_max_height": args["target_max_height"],
        "dslr_min_segment_frames": args["dslr_min_segment_frames"],
        "dslr_min_shared_points": args["dslr_min_shared_points"],
        "dslr_min_overlap_score": args["dslr_min_overlap_score"],
        "dslr_max_translation": args["dslr_max_translation"],
        "dslr_rotation_fov_fraction": args["dslr_rotation_fov_fraction"],
        "max_depth": args["max_depth"],
        "depth_percentile": args["depth_percentile"],
        "znear": args["znear"],
        "zfar": args["zfar"],
        "jpeg_quality": args["jpeg_quality"],
        "max_frames": args["max_frames"],
        "allow_missing_colmap": args["allow_missing_colmap"],
    }
    record = {"fingerprint": sha256_json(config), "config": config}
    path = osp.join(output_root, "dataset_config.json")
    if osp.isfile(path):
        previous = read_json(path)
        if previous.get("fingerprint") != record["fingerprint"]:
            raise RuntimeError(
                f"Dataset-level processing configuration differs from {path}. "
                "Use a different output directory rather than mixing configurations."
            )
    else:
        if args["validate_only"]:
            raise RuntimeError(f"Missing dataset-level configuration: {path}")
        atomic_write_json(path, record)
    return record


def main() -> int:
    namespace = get_parser().parse_args()
    args, scenes, split_file = normalize_args(namespace)
    output_root = osp.abspath(args["output_dir"])
    streams = ["iphone", "dslr"] if args["stream"] == "both" else [args["stream"]]

    with dataset_lock(output_root):
        dataset_config_record = ensure_dataset_config(output_root, args, split_file)

        print("=== ScanNet++ sequential preprocessing ===")
        print(f"Source       : {args['scannetpp_dir']}")
        print(f"Train split  : {split_file} ({len(scenes)} scenes selected)")
        print(f"Output       : {output_root}")
        print(f"Streams      : {', '.join(streams)}")
        print(f"iPhone stride: {args['iphone_stride']} (offset {args['iphone_offset']})")
        print(f"Workers      : {args['num_workers']}")
        print(f"EGL devices  : {args['egl_devices']}")
        print(f"Verify       : {args['verify']}; validate-only: {args['validate_only']}")

        if args["rebuild_metadata_only"]:
            metadata_summary = generate_metadata(output_root)
            stats = aggregate_completed_stats(output_root)
            render_readme(output_root, stats, args)
            print(f"Metadata rebuilt: {metadata_summary}")
            return 0

        task_args = {
            key: args[key]
            for key in (
                "iphone_stride",
                "iphone_offset",
                "target_width",
                "target_max_height",
                "dslr_min_segment_frames",
                "dslr_min_shared_points",
                "dslr_min_overlap_score",
                "dslr_max_translation",
                "dslr_rotation_fov_fraction",
                "max_depth",
                "depth_percentile",
                "znear",
                "zfar",
                "jpeg_quality",
                "pyopengl_platform",
                "egl_devices",
                "max_frames",
                "verify",
                "validate_only",
                "force",
                "allow_missing_colmap",
            )
        }
        tasks = [
            {
                "scene_id": scene_id,
                "scannetpp_dir": args["scannetpp_dir"],
                "output_root": output_root,
                "streams": streams,
                "args": task_args,
            }
            for scene_id in scenes
        ]

        all_results: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []
        started = time.time()

        if args["num_workers"] == 1:
            iterator = (process_scene_task(task) for task in tasks)
            for result in tqdm(iterator, total=len(tasks), desc="Scenes"):
                all_results.extend(result["results"])
                failures.extend(result["failures"])
                if args["fail_fast"] and result["failures"]:
                    break
        else:
            # Explicit spawn keeps EGL/OpenGL state worker-local and avoids
            # inheriting a graphics context through Linux fork.
            with ProcessPoolExecutor(
                max_workers=args["num_workers"], mp_context=mp.get_context("spawn")
            ) as executor:
                futures = {executor.submit(process_scene_task, task): task["scene_id"] for task in tasks}
                with tqdm(total=len(futures), desc="Scenes") as progress:
                    for future in as_completed(futures):
                        scene_id = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "results": [],
                                "failures": [{"scene_id": scene_id, "stream": "scene", "error": str(exc)}],
                            }
                        all_results.extend(result["results"])
                        failures.extend(result["failures"])
                        progress.update(1)
                        progress.set_postfix_str(scene_id)
                        if args["fail_fast"] and result["failures"]:
                            for pending in futures:
                                pending.cancel()
                            break

        metadata_summary = generate_metadata(output_root)
        completed_stats = aggregate_completed_stats(output_root)
        render_readme(output_root, completed_stats, args)

        summary = {
            "script_version": SCRIPT_VERSION,
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "finished_at": utc_now(),
            "elapsed_seconds": time.time() - started,
            "source": args["scannetpp_dir"],
            "split_file": split_file,
            "selected_scene_count": len(scenes),
            "streams": streams,
            "results": all_results,
            "failures": failures,
            "metadata_summary": metadata_summary,
            "completed_stats": completed_stats,
            "args": task_args,
            "dataset_config": dataset_config_record,
        }
        write_run_summary(output_root, summary)

        processed = sum(int(item.get("processed", 0)) for item in all_results)
        skipped = sum(int(item.get("skipped", 0)) for item in all_results)
        repaired = sum(int(item.get("repaired", 0)) for item in all_results)
        print(
            f"Done: processed={processed}, skipped={skipped}, repaired={repaired}, "
            f"failed_sequences={len(failures)}, elapsed={time.time() - started:.1f}s"
        )
        if failures:
            print("First failures:")
            for failure in failures[:20]:
                print(f"  {failure['scene_id']}/{failure['stream']}: {failure['error'].splitlines()[0]}")
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

