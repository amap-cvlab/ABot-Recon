# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""ARKitScenes high-resolution depth in the unmodified CUT3R export layout."""

from __future__ import annotations

import re

import cv2
import numpy as np

from .base import MultiViewDataset
from .foldback import foldback_from_start_long
from .io import imread_cv2


def _timestamp(name):
    stem = str(name).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        try:
            return float(parts[1])
        except ValueError:
            pass
    numbers = re.findall(r"\d+(?:\.\d+)?", stem)
    if not numbers:
        raise ValueError(f"Cannot parse ARKit timestamp: {name!r}")
    return float(numbers[1] if len(numbers) >= 2 else numbers[0])


def _segments(names, gap_threshold):
    if not len(names):
        return []
    segments = [[0]]
    previous = _timestamp(names[0])
    for index, name in enumerate(names[1:], 1):
        current = _timestamp(name)
        if current <= previous or current - previous > gap_threshold:
            segments.append([])
        segments[-1].append(index)
        previous = current
    return segments


class ARKitHR(MultiViewDataset):
    """Read Training/Validation scene_metadata, vga_wide and highres_depth.

    CUT3R's high-resolution preprocessor produces every required index. No
    generate_set pass or low-resolution ARKit directory is needed. Timestamp
    gaps break sequences; foldback never crosses those boundaries.
    """

    dataset_name = "arkit_hr"

    def __init__(
        self, root, *, timestamp_gap_threshold=1.0, min_interval=1,
        max_interval=1, forward_only=False, recent_stride_memory=8,
        fix_interval_prob=0.5, repeat_min_unique_divisor=None, **kwargs,
    ):
        self.timestamp_gap_threshold = float(timestamp_gap_threshold)
        if not np.isfinite(self.timestamp_gap_threshold) or self.timestamp_gap_threshold <= 0:
            raise ValueError("timestamp_gap_threshold must be positive and finite")
        self.min_interval, self.max_interval = int(min_interval), int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("require 1 <= min_interval <= max_interval")
        self.forward_only = bool(forward_only)
        self.recent_stride_memory = int(recent_stride_memory)
        self.fix_interval_prob = float(fix_interval_prob)
        self.repeat_min_unique_divisor = repeat_min_unique_divisor
        if repeat_min_unique_divisor is not None and int(repeat_min_unique_divisor) < 1:
            raise ValueError("repeat_min_unique_divisor must be positive")
        super().__init__(root=root, **kwargs)
        if self.split not in ("train", "test"):
            raise ValueError("ARKitHR split must be 'train' or 'test'")
        directory = self.root / ("Training" if self.split == "train" else "Validation")
        cutoff = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        minimum = self.num_views
        if self.allow_repeat and repeat_min_unique_divisor is not None:
            minimum = max(self.num_views // int(repeat_min_unique_divisor), 3)
        self.scenes, self.segments, self.starts = [], [], []
        for scene in sorted(path for path in directory.iterdir() if path.is_dir()):
            with np.load(scene / "scene_metadata.npz", allow_pickle=False) as metadata:
                order = np.argsort(metadata["images"], kind="stable")
                names = [str(value) for value in metadata["images"][order]]
                if len(names) < cutoff:
                    continue
                values = np.asarray(metadata["intrinsics"][order], dtype=np.float32)
                poses = np.asarray(metadata["trajectories"][order], dtype=np.float32)
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], len(names), axis=0)
            intrinsics[:, 0, 0], intrinsics[:, 1, 1] = values[:, 2], values[:, 3]
            intrinsics[:, 0, 2], intrinsics[:, 1, 2] = values[:, 4], values[:, 5]
            selected = [ids for ids in _segments(names, self.timestamp_gap_threshold) if len(ids) >= minimum]
            if not selected:
                continue
            scene_id = len(self.scenes)
            self.scenes.append((scene, names, intrinsics, poses))
            for ids in selected:
                segment_id = len(self.segments)
                self.segments.append((scene_id, ids))
                # The strictly forward sampler cannot start near the tail;
                # unlike foldback, it needs num_views-1 frames of headroom.
                starts = ids[:max(0, len(ids) - self.num_views + 1)] if self.forward_only else ids
                self.starts.extend((segment_id, index) for index in starts)

    def __len__(self):
        return len(self.starts)

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        segment_id, start = self.starts[idx]
        scene_id, ids = self.segments[segment_id]
        positions, _ = foldback_from_start_long(
            num_views=num_views, id_ref=start, ids_all=ids, rng=rng,
            min_interval=self.min_interval, max_interval=self.max_interval,
            forward_only=self.forward_only, recent_stride_memory=self.recent_stride_memory,
            fix_interval_prob=self.fix_interval_prob,
        )
        scene, names, intrinsics, poses = self.scenes[scene_id]
        views = []
        for position in positions:
            index = ids[position]
            name = names[index]
            image = imread_cv2(scene / "vga_wide" / name.replace(".png", ".jpg"))
            depth = imread_cv2(scene / "highres_depth" / name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            image, depth, K = self._crop_resize_if_necessary(
                image, depth, intrinsics[index].copy(), resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug, info=str(scene / name),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=K,
                camera_pose=poses[index].copy(), dataset=self.dataset_name,
                label=f"{scene.name}/{name}",
            ))
        return views
