# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""TartanAir v1 in the unmodified CUT3R processed layout."""

from __future__ import annotations

import numpy as np

from .base import MultiViewDataset
from .io import image_open, np_load
from .sequence import SequenceIndex


class TartanAir(MultiViewDataset):
    """Read environment/{Easy,Hard}/sequence/*_{rgb,depth,cam} files.

    The root is already split by the caller: ``split`` does not filter it or
    create a held-out set. The development default excludes environment names
    containing "Ocean" (case-insensitive); pass ``exclude_scene_substrings=()``
    to include every environment. No external index or blacklist is required.
    """

    dataset_name = "tartanair"

    def __init__(
        self, root, *, min_interval=1, max_interval=20, allow_repeat=True,
        exclude_scene_substrings=None, **kwargs,
    ):
        self.min_interval, self.max_interval = int(min_interval), int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("require 1 <= min_interval <= max_interval")
        self.exclude_scene_substrings = frozenset(
            ("Ocean",) if exclude_scene_substrings is None else exclude_scene_substrings
        )
        super().__init__(root=root, allow_repeat=allow_repeat, **kwargs)
        sequences = []
        for environment in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if any(str(token).casefold() in environment.name.casefold()
                   for token in self.exclude_scene_substrings):
                continue
            for mode in ("Easy", "Hard"):
                directory = environment / mode
                if not directory.is_dir():
                    continue
                for sequence in sorted(path for path in directory.iterdir() if path.is_dir()):
                    names = sorted(path.name[:-8] for path in sequence.glob("*_rgb.png") if path.is_file())
                    sequences.append((sequence, names))
        self._index = SequenceIndex(sequences, self.num_views, self.allow_repeat)

    def __len__(self):
        return len(self._index)

    def get_image_num(self):
        return self._index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        sequence, names, start = self._index.resolve(idx)
        if num_views == 1:
            positions = [start]
        else:
            lower, upper = self.adaptive_interval_bounds(
                len(names) - 1 - start, num_views, self.min_interval, self.max_interval,
            )
            positions, _ = self.get_seq_from_start_id(
                num_views, start, list(range(len(names))), rng,
                min_interval=lower, max_interval=upper, video_prob=1.0,
                fix_interval_prob=0.8, block_shuffle=16,
            )
        views = []
        for position in positions:
            name = names[position]
            image = image_open(sequence / f"{name}_rgb.png")
            depth = np.asarray(np_load(sequence / f"{name}_depth.npy"), dtype=np.float32)
            depth[depth >= 1000] = -1.0  # Development convention: invalid sky.
            depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
            positive = depth[depth > 0]
            threshold = np.percentile(positive, 98) if positive.size else 0
            depth[depth > threshold] = 0
            with np_load(sequence / f"{name}_cam.npz") as camera:
                intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float32)
                pose = np.asarray(camera["camera_pose"], dtype=np.float32)
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth, intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(sequence / f"{name}_rgb.png"),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_pose=pose, camera_intrinsics=intrinsics,
                dataset=self.dataset_name, label=f"{sequence.relative_to(self.root).as_posix()}/{name}",
            ))
        return views
