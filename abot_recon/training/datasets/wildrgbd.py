# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""WildRGB-D's original CUT3R export with development foldback sampling."""

from __future__ import annotations

import json
from collections import deque

import cv2
import numpy as np

from .base import MultiViewDataset
from .foldback import foldback_from_start_long
from .io import image_open, imread_cv2


class WildRGBD(MultiViewDataset):
    """Use selected_seqs_train/test.json and category/sequence payloads.

    Masks are foreground-valid (255), not invalid masks. Depth is stored in
    millimetres. The development loader's 90th-percentile depth cleanup and
    optional sequence-wide background masking are retained.
    """

    dataset_name = "wildrgbd"

    def __init__(self, root, *, mask_bg="rand", min_interval=1, max_interval=4, **kwargs):
        if mask_bg not in (True, False, "rand"):
            raise ValueError("mask_bg must be True, False or 'rand'")
        self.mask_bg = mask_bg
        self.min_stride, self.max_stride = int(min_interval), int(max_interval)
        if not 1 <= self.min_stride <= self.max_stride:
            raise ValueError("require 1 <= min_interval <= max_interval")
        super().__init__(root=root, **kwargs)
        if self.split not in ("train", "test"):
            raise ValueError("WildRGBD split must be 'train' or 'test'")
        with (self.root / f"selected_seqs_{self.split}.json").open(encoding="utf-8") as handle:
            inventory = json.load(handle)
        self.cutoff = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        self.scenes, self.starts = [], []
        for category, sequences in inventory.items():
            for sequence, frame_ids in sequences.items():
                if (category, sequence) == ("box", "scenes/scene_257") or len(frame_ids) < self.cutoff:
                    continue
                scene_id = len(self.scenes)
                self.scenes.append((self.root / category / sequence, [int(x) for x in frame_ids]))
                self.starts.extend((scene_id, index) for index in range(len(frame_ids) - self.cutoff + 1))
        self._invalid_frames = {}

    def __len__(self):
        return len(self.starts)

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        scene_id, start = self.starts[idx]
        scene, frame_ids = self.scenes[scene_id]
        ids = list(range(len(frame_ids)))
        positions, _ = foldback_from_start_long(
            num_views=num_views, id_ref=start, ids_all=ids, rng=rng,
            min_interval=self.min_stride, max_interval=self.max_stride,
            forward_only=False, recent_stride_memory=1,
        )
        mask_bg = self.mask_bg is True or (self.mask_bg == "rand" and bool(rng.choice(2, p=[0.9, 0.1])))
        invalid = self._invalid_frames.setdefault((scene_id, tuple(resolution)), set())
        views, used, pending = [], [], deque(positions)
        attempts = 0
        while pending:
            # Match development: enqueue a newly invalid frame at the tail,
            # process later requested frames first, then replace it nearby.
            # Each failure marks a distinct frame; bound total work explicitly
            # and leave whole-sample refetching to the shared base.
            attempts += 1
            if attempts > num_views + len(frame_ids):
                raise RuntimeError(f"Depth-frame retry limit exceeded in {scene}")
            if len(frame_ids) - len(invalid) < self.cutoff:
                raise RuntimeError(f"Too few valid depth frames in {scene}")
            position = pending.popleft()
            if position in invalid:
                direction = int(2 * rng.choice(2) - 1)
                position = next(
                    (position + direction * offset) % len(frame_ids)
                    for offset in range(1, len(frame_ids))
                    if (position + direction * offset) % len(frame_ids) not in invalid
                )
            basename = f"{frame_ids[position]:05d}"
            with np.load(scene / "metadata" / f"{basename}.npz", allow_pickle=False) as camera:
                K = np.asarray(camera["camera_intrinsics"], dtype=np.float32).copy()
                pose = np.asarray(camera["camera_pose"], dtype=np.float32).copy()
            image = image_open(scene / "rgb" / f"{basename}.jpg")
            depth = imread_cv2(scene / "depth" / f"{basename}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            if mask_bg:
                foreground = imread_cv2(scene / "masks" / f"{basename}.png", cv2.IMREAD_UNCHANGED)
                depth *= foreground.astype(np.float32) / 255.0 > 0.1
            positive = depth > 0
            if positive.any():
                depth[depth > np.percentile(depth[positive], 90)] = 0
            image, depth, K = self._crop_resize_if_necessary(
                image, depth, K, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(scene / "rgb" / f"{basename}.jpg"),
            )
            if not (depth > 0).any():
                invalid.add(position)
                pending.append(position)
                continue
            used.append(position)
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=K,
                camera_pose=pose, dataset=self.dataset_name,
                label=f"{scene.relative_to(self.root)}/{basename}",
            ))
        if num_views > 1 and len(set(used)) < 2:
            raise RuntimeError(f"Only one distinct valid frame sampled in {scene}")
        return views
