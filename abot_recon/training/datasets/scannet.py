# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""ScanNet after CUT3R's preprocess_scannet + generate_set_scannet passes."""

from __future__ import annotations

import cv2
import numpy as np

from .base import MultiViewDataset
from .io import imread_cv2, np_load
from .sequence import SequenceIndex


class ScanNet(MultiViewDataset):
    """Read the final ScanNet export, preserving the development video sampler.

    ``root`` contains ``scans_train`` and/or ``scans_test``. Each scene requires
    ``new_scene_metadata.npz`` from the second preprocessing pass. Its ``images``
    array provides the frame order; ``video_collection`` is deliberately unused,
    just as in the W/X loaders. Temporal gaps do not create new boundaries.
    """

    dataset_name = "scannet"

    def __init__(self, root, *, min_interval=1, max_interval=30, **kwargs):
        self.min_interval, self.max_interval = int(min_interval), int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("interval bounds must satisfy 1 <= min_interval <= max_interval")
        super().__init__(root=root, **kwargs)
        if self.split not in ("train", "test"):
            raise ValueError("ScanNet split must be 'train' or 'test'")
        self.scene_root = self.root / ("scans_train" if self.split == "train" else "scans_test")
        sequences = []
        for scene in sorted(self.scene_root.iterdir()):
            if not scene.is_dir() or not scene.name.startswith("scene"):
                continue
            metadata_path = scene / "new_scene_metadata.npz"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing {metadata_path}; run CUT3R generate_set_scannet.py "
                    "after preprocess_scannet.py"
                )
            # The official writer stores images as Unicode, but the unused
            # video_collection as an object array. Access only images so the
            # official archive works without ever unpickling the object member.
            with np.load(metadata_path, allow_pickle=False) as metadata:
                basenames = metadata["images"]
                if basenames.ndim != 1 or (basenames.size and basenames.dtype.kind != "U"):
                    raise ValueError(f"Expected a 1-D Unicode images array in {metadata_path}")
                names = basenames.tolist()
            sequences.append((scene, names))
        self.index = SequenceIndex(sequences, self.num_views, self.allow_repeat)

    def __len__(self):
        return len(self.index)

    def get_image_num(self):
        return self.index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        scene, names, start = self.index.resolve(idx)
        if num_views < 1:
            raise ValueError("num_views must be positive")
        remaining = len(names) - 1 - start
        if not self.allow_repeat and remaining < num_views - 1:
            raise ValueError("requested num_views exceeds the non-repeating sequence tail")
        lower, upper = self.adaptive_interval_bounds(
            remaining, num_views, self.min_interval, self.max_interval
        )
        positions, _ = self.get_seq_from_start_id(
            num_views, start, list(range(len(names))), rng,
            min_interval=lower, max_interval=upper,
            video_prob=1.0, fix_interval_prob=0.6, block_shuffle=16,
        )
        views = []
        for position in positions:
            basename = names[position]
            image = imread_cv2(scene / "color" / (basename + ".jpg"))
            depth = imread_cv2(scene / "depth" / (basename + ".png"), cv2.IMREAD_UNCHANGED)
            if depth.ndim != 2 or depth.shape != image.shape[:2]:
                raise ValueError(f"RGB/depth shape mismatch: {scene}/{basename}")
            depth = depth.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0
            with np_load(scene / "cam" / (basename + ".npz")) as camera:
                pose = np.asarray(camera["pose"], dtype=np.float32).copy()
                intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid camera pose: {scene}/{basename}")
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid camera intrinsics: {scene}/{basename}")
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"Non-positive focal length: {scene}/{basename}")
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth, intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=f"{scene}/{basename}",
            )
            views.append(dict(
                img=image, depthmap=depth, camera_pose=pose,
                camera_intrinsics=intrinsics, dataset=self.dataset_name,
                label=f"{scene.name}_{basename}",
            ))
        return views
