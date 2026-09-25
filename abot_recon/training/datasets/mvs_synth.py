# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""MVS-Synth sequences in the unmodified CUT3R processed layout."""

from __future__ import annotations

import numpy as np

from .base import MultiViewDataset
from .foldback import foldback_from_start_long
from .io import image_open, np_load
from .sequence import SequenceIndex


class MVSSynth(MultiViewDataset):
    """Read sequence/{rgb/*.jpg,depth/*.npy,cam/*.npz} without unit conversion.

    Official RGB exports may contain PNG bytes despite their .jpg suffix;
    decoding uses the actual image header. NPZ keys are ``intrinsics`` and
    camera-to-world ``pose``. ``split`` does not partition the supplied root.
    """

    dataset_name = "mvs_synth"

    def __init__(
        self, root, *, min_interval=1, max_interval=4, allow_repeat=True, **kwargs,
    ):
        self.min_interval, self.max_interval = int(min_interval), int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("require 1 <= min_interval <= max_interval")
        super().__init__(root=root, allow_repeat=allow_repeat, **kwargs)
        sequences = []
        for sequence in sorted(path for path in self.root.iterdir() if path.is_dir()):
            rgb = sequence / "rgb"
            if rgb.is_dir():
                names = sorted(path.stem for path in rgb.glob("*.jpg") if path.is_file())
                sequences.append((sequence, names))
        self._index = SequenceIndex(sequences, self.num_views, self.allow_repeat)

    def __len__(self):
        return len(self._index)

    def get_image_num(self):
        return self._index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        sequence, names, start = self._index.resolve(idx)
        positions, _ = foldback_from_start_long(
            num_views=num_views, id_ref=start, ids_all=list(range(len(names))), rng=rng,
            min_interval=self.min_interval, max_interval=self.max_interval,
            forward_only=False, recent_stride_memory=1,
        )
        views = []
        for position in positions:
            name = names[position]
            image = image_open(sequence / "rgb" / f"{name}.jpg")
            depth = np.asarray(np_load(sequence / "depth" / f"{name}.npy"), dtype=np.float32)
            depth[~np.isfinite(depth)] = 0
            positive = depth[depth > 0]
            threshold = np.percentile(positive, 98) if positive.size else 0
            depth[depth > threshold] = 0
            depth[depth > 1000] = 0
            with np_load(sequence / "cam" / f"{name}.npz") as camera:
                intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
                pose = np.asarray(camera["pose"], dtype=np.float32)
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth, intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(sequence / "rgb" / f"{name}.jpg"),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_pose=pose, camera_intrinsics=intrinsics,
                dataset=self.dataset_name, label=f"{sequence.name}/{name}",
            ))
        return views
