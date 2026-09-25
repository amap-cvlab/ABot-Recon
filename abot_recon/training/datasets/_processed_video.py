"""Shared local I/O for CUT3R rgb/depth/cam video exports."""

from pathlib import Path

import numpy as np

from .base import MultiViewDataset
from .io import image_open, np_load
from .sequence import SequenceIndex


class ProcessedVideoDataset(MultiViewDataset):
    """Subclasses supply scene discovery, filename order and depth filtering."""

    image_suffix = ".png"
    fix_interval_prob = 1.0
    depth_limit = None
    depth_limit_inclusive = False

    def __init__(self, root, *args, min_interval=1, max_interval=4, **kwargs):
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("interval bounds must satisfy 1 <= min_interval <= max_interval")
        super().__init__(root=root, *args, **kwargs)
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        self.index = SequenceIndex(self._find_sequences(), self.num_views, self.allow_repeat)

    def _find_sequences(self):
        raise NotImplementedError

    def _basenames(self, directory, *, key=None):
        return sorted(
            (path.stem for path in (directory / "rgb").iterdir()
             if path.is_file() and path.suffix == self.image_suffix),
            key=key,
        )

    def _scene_label(self, directory):
        return directory.name

    def __len__(self):
        return len(self.index)

    def get_image_num(self):
        return self.index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        directory, names, start = self.index.resolve(idx)
        directory = Path(directory)
        if num_views < 1:
            raise ValueError("num_views must be positive")
        remaining = len(names) - 1 - start
        if not self.allow_repeat and remaining < num_views - 1:
            raise ValueError("requested num_views exceeds the non-repeating sequence tail")
        if num_views == 1:
            positions = [start]
        else:
            lower, upper = self.adaptive_interval_bounds(
                remaining, num_views, self.min_interval, self.max_interval
            )
            positions, _ = self.get_seq_from_start_id(
                num_views, start, list(range(len(names))), rng,
                min_interval=lower, max_interval=upper,
                video_prob=1.0, fix_interval_prob=self.fix_interval_prob,
            )

        views = []
        for position in positions:
            basename = names[position]
            image = image_open(directory / "rgb" / (basename + self.image_suffix))
            depth = np.asarray(np_load(directory / "depth" / (basename + ".npy"))).copy()
            if depth.ndim != 2 or depth.shape != (image.height, image.width):
                raise ValueError(f"RGB/depth shape mismatch: {directory}/{basename}")
            depth[~np.isfinite(depth)] = 0
            if self.depth_limit is not None:
                invalid = (
                    depth >= self.depth_limit
                    if self.depth_limit_inclusive else depth > self.depth_limit
                )
                depth[invalid] = 0
            with np_load(directory / "cam" / (basename + ".npz")) as camera:
                pose = np.asarray(camera["pose"], dtype=np.float32).copy()
                intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid camera pose: {directory}/{basename}")
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid camera intrinsics: {directory}/{basename}")
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"Non-positive focal length: {directory}/{basename}")
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth.astype(np.float32), intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=f"{directory}/{basename}",
            )
            views.append(dict(
                img=image,
                depthmap=depth,
                camera_pose=pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_name,
                label=f"{self._scene_label(directory)}_{basename}",
            ))
        return views
