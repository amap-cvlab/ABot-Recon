# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""Independent ScanNet++ device streams from preprocess_scannet_seq.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import MultiViewDataset
from .foldback import foldback_from_start_long
from .io import imread_cv2, np_load
from .sequence import SequenceIndex


_SEQUENCE_BLACKLIST_FILENAME = "scannetpp_geometry_blacklist_20260810.txt"
# Additional geometry exclusions, separate from the legacy built-in below.
_DEFAULT_GEOMETRY_EXCLUDED_SEQUENCES = frozenset({
    "0e350246d3_dslr",
    "9ef704a38d_dslr",
    "eaa6c90310_dslr",
    "fe5fe0a8a4_dslr",
    "46001f434d_iphone",
    "99010a8938_dslr",
    "c8d099ecd8_dslr",
})


class ScanNetPPSeq(MultiViewDataset):
    """Read ordered iPhone/DSLR exports with unit-stride boundary foldback.

    The root metadata, not ``split``, selects sequences. The supplied producer
    selects its train split before exporting. Each metadata key is independent;
    neither device streams nor physical scenes are joined. Depth NPYs are
    already metric, and masks are True for invalid pixels.

    ``allow_repeat`` does not control this sampler: default foldback admits
    sequences of two frames and every frame is a start. ``forward_only`` instead
    requires enough headroom for the requested number of views.

    ``sequence_blacklist_path=None`` or ``"auto"`` uses the dated blacklist
    file at the dataset root when present, otherwise seven inline geometry
    exclusions. An explicit local text-file path replaces those defaults;
    relative paths are resolved against the dataset root. ``""`` disables the
    additional geometry blacklist. The legacy ``cc0aa81452_iphone`` exclusion
    and caller-supplied ``excluded_sequences`` always remain in effect.
    Blank lines and # comments in blacklist files are ignored.
    """

    dataset_name = "scannetpp_seq"

    def __init__(
        self, root, *, stride=1, min_interval=1, max_interval=1,
        metadata_filename="all_metadata.npz", max_starts_per_scene=None,
        excluded_sequences=None, sequence_blacklist_path=None,
        forward_only=False, recent_stride_memory=0, **kwargs,
    ):
        if any(value != 1 for value in (stride, min_interval, max_interval)):
            raise ValueError("ScanNetPPSeq requires stride=min_interval=max_interval=1")
        self.stride = self.min_interval = self.max_interval = 1
        self.forward_only = bool(forward_only)
        self.recent_stride_memory = int(recent_stride_memory)
        self.metadata_filename = str(metadata_filename)
        self.max_starts_per_scene = (
            None if max_starts_per_scene is None else int(max_starts_per_scene)
        )
        if self.max_starts_per_scene is not None and self.max_starts_per_scene < 1:
            raise ValueError("max_starts_per_scene must be positive or None")
        # Keep the legacy exclusion independent of the optional geometry list.
        self.excluded_sequences = {"cc0aa81452_iphone"}
        self.excluded_sequences.update(str(name) for name in (excluded_sequences or ()))
        self.sequence_blacklist_path = sequence_blacklist_path
        root_path = Path(root).expanduser()
        if sequence_blacklist_path is None or sequence_blacklist_path == "auto":
            path = root_path / _SEQUENCE_BLACKLIST_FILENAME
            if not path.is_file():
                self.excluded_sequences.update(_DEFAULT_GEOMETRY_EXCLUDED_SEQUENCES)
                path = None
        elif sequence_blacklist_path == "":
            path = None
        else:
            path = Path(sequence_blacklist_path).expanduser()
            if not path.is_absolute():
                path = root_path / path
        if path is not None:
            text = path.read_text(encoding="utf-8")
            self.excluded_sequences.update(
                line.strip() for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        super().__init__(root=root, **kwargs)
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        # The producer's numeric arrays need no pickle. The development loader
        # also accepts a legacy scalar `metadata` dictionary in trusted exports.
        with np.load(self.root / self.metadata_filename, allow_pickle=True) as archive:
            if "metadata" in archive.files:
                value = archive["metadata"]
                metadata = value.item() if value.shape == () else value.tolist()
                if not isinstance(metadata, dict):
                    raise ValueError("ScanNet++ metadata entry must be a dictionary")
            else:
                metadata = {key: archive[key] for key in archive.files}
        sequences = []
        minimum = self.num_views if self.forward_only else 2
        for name in sorted(set(metadata).difference(self.excluded_sequences)):
            directory = self.root / name
            if not directory.is_dir():
                continue
            info = metadata[name]
            if isinstance(info, dict):
                for key in ("frame_indices", "frame_ids", "frames", "images"):
                    if key in info:
                        info = info[key]
                        break
                else:
                    raise KeyError(f"Missing frame indices for ScanNet++ sequence {name!r}")
            frame_ids = [int(value) for value in np.asarray(info).reshape(-1).tolist()]
            if frame_ids != sorted(frame_ids):
                raise ValueError(f"Frame indices must be sorted in ScanNet++ sequence {name!r}")
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError(f"Duplicate frame indices in ScanNet++ sequence {name!r}")
            if any(value < 0 for value in frame_ids):
                raise ValueError(f"Negative frame indices in ScanNet++ sequence {name!r}")
            if len(frame_ids) >= minimum:
                sequences.append((directory, [f"{value:05d}" for value in frame_ids]))
        # Use cutoff=1 for all foldback starts, after the explicit two-frame
        # filter. The generic allow_repeat tail cutoff would change W/X behavior.
        cutoff = self.num_views if self.forward_only else 1
        self.index = SequenceIndex(sequences, cutoff, False)
        self._starts = None
        if self.max_starts_per_scene is not None:
            self._starts, offset = [], 0
            for _, names in self.index.sequences:
                count = len(names) - cutoff + 1
                positions = np.linspace(0, count - 1, min(self.max_starts_per_scene, count)).astype(int)
                self._starts.extend(offset + int(position) for position in positions)
                offset += count

    def __len__(self):
        return len(self.index) if self._starts is None else len(self._starts)

    def get_image_num(self):
        return self.index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        item = idx if self._starts is None else self._starts[idx]
        directory, names, start = self.index.resolve(item)
        if self.forward_only and len(names) - start < num_views:
            raise ValueError("Requested views exceed forward-only sequence headroom")
        positions, _ = foldback_from_start_long(
            num_views=num_views, id_ref=start, ids_all=list(range(len(names))), rng=rng,
            min_interval=1, max_interval=1, forward_only=self.forward_only,
            recent_stride_memory=self.recent_stride_memory, fix_interval_prob=1.0,
        )
        views = []
        for position in positions:
            basename = names[position]
            image = imread_cv2(directory / "images" / (basename + ".jpg"))
            depth = np.array(np_load(directory / "depths" / (basename + ".npy")), dtype=np.float32, copy=True)
            invalid = np.asarray(np_load(directory / "masks" / (basename + ".npy")), dtype=bool)
            if depth.ndim != 2 or image.shape[:2] != depth.shape:
                raise ValueError(f"RGB/depth shape mismatch: {directory}/{basename}")
            if invalid.shape != depth.shape:
                raise ValueError(f"Mask/depth shape mismatch: {directory}/{basename}")
            depth[~np.isfinite(depth) | invalid | (depth <= 0)] = 0
            with np_load(directory / "cameras" / (basename + ".npz")) as camera:
                intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float32).copy()
                pose = np.asarray(camera["camera_pose"], dtype=np.float32).copy()
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid camera intrinsics: {directory}/{basename}")
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"Non-positive focal length: {directory}/{basename}")
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid camera pose: {directory}/{basename}")
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth, intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=f"{directory}/{basename}",
            )
            views.append(dict(
                img=image, depthmap=depth, camera_intrinsics=intrinsics,
                camera_pose=pose, dataset=self.dataset_name,
                label=f"{directory.name}/{basename}",
            ))
        return views
