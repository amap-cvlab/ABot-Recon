# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""ARKitScenes low-resolution depth after CUT3R's two preprocessing passes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .arkit_hr import _segments
from .base import MultiViewDataset
from .foldback import foldback_from_start_long
from .io import imread_cv2


class ARKitScenes(MultiViewDataset):
    """Read final Training/Test metadata, RGB JPG and millimetre depth PNG.

    ``preprocess_arkitscenes.py`` produces ``all_metadata.npz`` and the media;
    ``generate_set_arkitscenes.py`` produces ``new_scene_metadata.npz``.
    Only load trusted, locally generated metadata: its ``image_collection``
    member is the upstream pickled dictionary used for scene eligibility.

    The high-resolution scene names are excluded, as in development. By
    default they are read from ``root + '_highres'`` (Training/Validation).
    Poses and intrinsics already incorporate the producer's image rotation.
    ``camera_only=True`` avoids reading depth and marks the unit-depth
    placeholder as camera-only supervision through the shared release path.
    """

    dataset_name = "arkit"

    def __init__(
        self, root, *, timestamp_gap_threshold=1.0, min_interval=1,
        max_interval=1, forward_only=False, recent_stride_memory=8,
        fix_interval_prob=0.5, repeat_min_unique_divisor=None,
        camera_only=False, highres_root=None, **kwargs,
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
        if self.recent_stride_memory < 1:
            raise ValueError("recent_stride_memory must be positive")
        if not 0 <= self.fix_interval_prob <= 1:
            raise ValueError("fix_interval_prob must be in [0, 1]")
        self.repeat_min_unique_divisor = (
            None if repeat_min_unique_divisor is None else int(repeat_min_unique_divisor)
        )
        if self.repeat_min_unique_divisor is not None and self.repeat_min_unique_divisor < 1:
            raise ValueError("repeat_min_unique_divisor must be positive")
        super().__init__(root=root, **kwargs)
        self.camera_only = bool(camera_only)
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        if self.split not in ("train", "test"):
            raise ValueError("ARKitScenes split must be 'train' or 'test'")
        directory = self.root / ("Training" if self.split == "train" else "Test")
        self.highres_root = (
            Path(highres_root).expanduser() if highres_root is not None
            else Path(str(self.root) + "_highres")
        )
        highres_split = self.highres_root / (
            "Training" if self.split == "train" else "Validation"
        )
        if not highres_split.is_dir():
            raise FileNotFoundError(
                f"Missing ARKit high-resolution exclusion directory: {highres_split}. "
                "Prepare the high-resolution Training/Validation scene directories "
                "or set highres_root to their root; exclusion is not silently disabled."
            )
        excluded = {path.name for path in highres_split.iterdir()}
        with np.load(directory / "all_metadata.npz", allow_pickle=False) as metadata:
            scene_names = metadata["scenes"]
            if scene_names.ndim != 1:
                raise ValueError("ARKit all_metadata scenes must be one-dimensional")
            scene_names = sorted(set(str(value) for value in scene_names) - excluded)

        cutoff = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        minimum = self.num_views
        if self.allow_repeat and self.repeat_min_unique_divisor is not None:
            minimum = max(self.num_views // self.repeat_min_unique_divisor, 3)
        self.scenes, self.segments, self.starts = [], [], []
        for scene_name in scene_names:
            scene = directory / scene_name
            metadata_path = scene / "new_scene_metadata.npz"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing {metadata_path}; run CUT3R generate_set_arkitscenes.py "
                    "after preprocess_arkitscenes.py"
                )
            with np.load(metadata_path, allow_pickle=True) as metadata:
                names = [str(value) for value in metadata["images"]]
                if len(names) < cutoff:
                    continue
                collection = metadata["image_collection"].item()
                if not isinstance(collection, dict):
                    raise ValueError(f"Invalid image_collection: {metadata_path}")
                if not any(len(group) + 1 >= cutoff for group in collection.values()):
                    continue
                values = np.asarray(metadata["intrinsics"], dtype=np.float32)
                poses = np.asarray(metadata["trajectories"], dtype=np.float32)
            if values.shape != (len(names), 6) or not np.isfinite(values).all():
                raise ValueError(f"Invalid ARKit intrinsics: {metadata_path}")
            if poses.shape != (len(names), 4, 4) or not np.isfinite(poses).all():
                raise ValueError(f"Invalid ARKit camera poses: {metadata_path}")
            if (values[:, 2:4] <= 0).any():
                raise ValueError(f"Non-positive focal length: {metadata_path}")
            if any(not name.startswith(scene_name + "_") or not name.endswith(".png")
                   or Path(name).name != name for name in names):
                raise ValueError(f"Invalid ARKit image names: {metadata_path}")
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], len(names), axis=0)
            intrinsics[:, 0, 0], intrinsics[:, 1, 1] = values[:, 2], values[:, 3]
            intrinsics[:, 0, 2], intrinsics[:, 1, 2] = values[:, 4], values[:, 5]
            selected = [ids for ids in _segments(names, self.timestamp_gap_threshold)
                        if len(ids) >= minimum]
            if not selected:
                continue
            scene_id = len(self.scenes)
            self.scenes.append((scene, names, intrinsics, poses))
            for ids in selected:
                segment_id = len(self.segments)
                self.segments.append((scene_id, ids))
                # Match the release HR adapter's forward-only headroom guard.
                starts = ids[:max(0, len(ids) - self.num_views + 1)] if self.forward_only else ids
                self.starts.extend((segment_id, index) for index in starts)

    def __len__(self):
        return len(self.starts)

    def get_image_num(self):
        # Development counts the retained scenes, including unsampled short segments.
        return sum(len(names) for _, names, _, _ in self.scenes)

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        segment_id, start = self.starts[idx]
        scene_id, ids = self.segments[segment_id]
        if self.forward_only and len(ids) - ids.index(start) < num_views:
            raise ValueError("requested num_views exceeds the forward-only segment tail")
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
            if self.camera_only:
                depth = np.ones(image.shape[:2], dtype=np.float32)
            else:
                depth = imread_cv2(scene / "lowres_depth" / name, cv2.IMREAD_UNCHANGED)
                if depth.ndim != 2 or depth.shape != image.shape[:2]:
                    raise ValueError(f"RGB/depth shape mismatch: {scene}/{name}")
                depth = depth.astype(np.float32) / 1000.0
                depth[~np.isfinite(depth)] = 0
            image, depth, K = self._crop_resize_if_necessary(
                image, depth, intrinsics[index].copy(), resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug, info=str(scene / name),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=K,
                camera_pose=poses[index].copy(), dataset=self.dataset_name,
                label=f"{scene.name}/{name}", camera_only=self.camera_only,
            ))
        return views
