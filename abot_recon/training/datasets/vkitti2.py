# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""Virtual KITTI 2 in CUT3R's officially distributed, processed layout."""

from __future__ import annotations

import cv2
import numpy as np

from .base import MultiViewDataset
from .io import image_open, imread_cv2, np_load
from .sequence import SequenceIndex


class VirtualKITTI2(MultiViewDataset):
    """Read ``scene/variant/Camera_{0,1}/NNNNN_{rgb,depth,cam}`` files.

    CUT3R distributes the completed export rather than a VKITTI preprocessing
    script. RGB is ``_rgb.jpg``, centimeter depth is ``_depth.png``, and
    ``_cam.npz`` contains ``camera_intrinsics`` and camera-to-world
    ``camera_pose``. Raw VKITTI archives are not this layout.

    ``train`` excludes the last lexicographically sorted scene and ``test``
    selects it; ``None`` uses every scene, matching development training.
    Each camera/variant remains a separate temporal stream. Intervals adapt to
    the remaining tail, and ``allow_repeat`` controls eligible tail lengths.
    """

    dataset_name = "vkitti2"

    def __init__(
        self, root, *, split=None, min_interval=1, max_interval=5,
        allow_repeat=False, **kwargs,
    ):
        self.min_interval, self.max_interval = int(min_interval), int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("require 1 <= min_interval <= max_interval")
        super().__init__(root=root, split=split, allow_repeat=allow_repeat, **kwargs)
        self.split = split
        scenes = sorted(path for path in self.root.iterdir() if path.is_dir())
        if self.split == "train":
            scenes = scenes[:-1]
        elif self.split == "test":
            scenes = scenes[-1:]
        sequences = []
        for scene in scenes:
            for variant in sorted(path for path in scene.iterdir() if path.is_dir()):
                for camera in ("Camera_0", "Camera_1"):
                    directory = variant / camera
                    # Both camera directories are part of the official export.
                    # Do not mistake raw VKITTI folders for processed sequences.
                    names = sorted(
                        path.name[:-8] for path in directory.iterdir()
                        if path.is_file() and path.name.endswith("_rgb.jpg")
                    )
                    sequences.append((directory, names))
        self._index = SequenceIndex(sequences, self.num_views, self.allow_repeat)

    def __len__(self):
        return len(self._index)

    def get_image_num(self):
        return self._index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        directory, names, start = self._index.resolve(idx)
        if num_views < 1:
            raise ValueError("num_views must be positive")
        remaining = len(names) - 1 - start
        if not self.allow_repeat and remaining < num_views - 1:
            raise ValueError("requested num_views exceeds the non-repeating sequence tail")
        if num_views == 1:
            positions = [start]
        else:
            lower, upper = self.adaptive_interval_bounds(
                remaining, num_views, self.min_interval, self.max_interval,
            )
            positions, _ = self.get_seq_from_start_id(
                num_views, start, list(range(len(names))), rng,
                min_interval=lower, max_interval=upper, video_prob=1.0,
                fix_interval_prob=0.9,
            )

        views = []
        for position in positions:
            name = names[position]
            image = image_open(directory / f"{name}_rgb.jpg")
            depth = imread_cv2(
                directory / f"{name}_depth.png",
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            ).astype(np.float32) / 100.0
            if depth.ndim != 2 or depth.shape != (image.height, image.width):
                raise ValueError(f"RGB/depth shape mismatch: {directory}/{name}")
            depth[depth >= 655] = -1.0  # Preserve the development invalid-sky sentinel.
            with np_load(directory / f"{name}_cam.npz") as camera:
                intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float32).copy()
                pose = np.asarray(camera["camera_pose"], dtype=np.float32).copy()
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid camera pose: {directory}/{name}")
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid camera intrinsics: {directory}/{name}")
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"Non-positive focal length: {directory}/{name}")
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth, intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(directory / f"{name}_rgb.jpg"),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_pose=pose,
                camera_intrinsics=intrinsics, dataset=self.dataset_name,
                label=f"{directory.relative_to(self.root).as_posix()}/{name}",
            ))
        return views
