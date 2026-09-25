# Adapted from CUT3R and the development loaders; see THIRD_PARTY_NOTICES.md.
# CUT3R-derived portions retain CC BY-NC-SA 4.0.
"""Waymo's unmodified CUT3R export and separately supplied invalid-pair list."""

from __future__ import annotations

import cv2
import numpy as np

from .base import MultiViewDataset
from .io import image_open, imread_cv2, np_load
from .sequence import SequenceIndex


class Waymo(MultiViewDataset):
    """Read ``scene/00000_1.{jpg,exr,npz}`` and ``invalid_files.h5``.

    The producer stores sparse metric camera-z depth, ``intrinsics``, and
    already converted OpenCV-axis camera-to-world ``cam2world``. Its extra
    ``distortion`` coefficients do not imply another undistortion step.
    The HDF5 artifact is a separate official download, not an output of
    ``preprocess_waymo.py``; each ``scene/invalid_pairs`` value is the UTF-8
    byte string ``camera_frame``. No packs or dataset_metadata.npz are used.

    As upstream, this root has no loader-level train/validation partition:
    ``split`` must be None. Camera 5 is excluded, cameras are sampled
    independently, and camera 4 uses half the configured maximum interval.
    """

    dataset_name = "waymo"

    def __init__(self, root, *, split=None, min_interval=1, max_interval=8, **kwargs):
        if split is not None:
            raise ValueError("Waymo requires split=None; the supplied root selects the data")
        self.min_interval, self.max_interval = int(min_interval), int(max_interval)
        if not 1 <= self.min_interval <= self.max_interval:
            raise ValueError("require 1 <= min_interval <= max_interval")
        super().__init__(root=root, split=split, **kwargs)
        self.split = None
        invalid_path = self.root / "invalid_files.h5"
        if not invalid_path.is_file():
            raise FileNotFoundError(
                f"Missing {invalid_path}; CUT3R requires the separately supplied "
                "invalid_files.h5 after preprocessing Waymo"
            )
        invalid = self._load_invalid_pairs(invalid_path)
        sequences = []
        for scene in sorted(path for path in self.root.iterdir() if path.is_dir()):
            cameras = {}
            for path in sorted(scene.glob("*.jpg")):
                if not path.is_file():
                    continue
                parts = path.stem.split("_")
                if (len(parts) != 2 or not parts[0].isdigit()
                        or parts[1] not in {"1", "2", "3", "4", "5"}):
                    raise ValueError(f"Invalid Waymo frame basename: {path.name}")
                frame, camera = parts
                if camera == "5" or (camera, frame) in invalid.get(scene.name, set()):
                    continue
                cameras.setdefault(camera, []).append(frame)
            for camera, frames in sorted(cameras.items()):
                sequences.append(((scene, camera), sorted(frames)))
        self._index = SequenceIndex(sequences, self.num_views, self.allow_repeat)

    @staticmethod
    def _load_invalid_pairs(path):
        import h5py  # Needed only when the Waymo source is enabled.

        invalid = {}
        with h5py.File(path, "r") as source:
            for scene, group in source.items():
                pairs = set()
                for value in group["invalid_pairs"][:]:
                    value = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    pair = tuple(value.split("_"))
                    if len(pair) != 2:
                        raise ValueError(f"Invalid Waymo exclusion in {path}: {value!r}")
                    pairs.add(pair)
                invalid[scene] = pairs
        return invalid

    def __len__(self):
        return len(self._index)

    def get_image_num(self):
        return self._index.image_count

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        (scene, camera), names, start = self._index.resolve(idx)
        if num_views < 1:
            raise ValueError("num_views must be positive")
        remaining = len(names) - 1 - start
        if not self.allow_repeat and remaining < num_views - 1:
            raise ValueError("requested num_views exceeds the non-repeating sequence tail")
        maximum = max(1, self.max_interval // 2) if camera == "4" else self.max_interval
        lower, upper = self.adaptive_interval_bounds(
            remaining, num_views, self.min_interval, maximum
        )
        positions, _ = self.get_seq_from_start_id(
            num_views, start, list(range(len(names))), rng,
            min_interval=lower, max_interval=upper,
            video_prob=1.0, fix_interval_prob=0.9, block_shuffle=16,
        )
        views = []
        for position in positions:
            basename = f"{names[position]}_{camera}"
            image_path = scene / f"{basename}.jpg"
            image = image_open(image_path)
            depth_path = scene / f"{basename}.exr"
            try:
                depth = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            except cv2.error as error:
                raise OSError(
                    f"Cannot read Waymo EXR depth {depth_path}; use an OpenCV build "
                    "with OpenEXR support and OPENCV_IO_ENABLE_OPENEXR=1"
                ) from error
            if depth.ndim != 2 or depth.shape != (image.height, image.width):
                raise ValueError(f"RGB/depth shape mismatch: {scene}/{basename}")
            depth = np.asarray(depth, dtype=np.float32).copy()
            depth[~np.isfinite(depth)] = 0
            with np_load(scene / f"{basename}.npz") as camera_data:
                intrinsics = np.asarray(camera_data["intrinsics"], dtype=np.float32).copy()
                pose = np.asarray(camera_data["cam2world"], dtype=np.float32).copy()
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid camera pose: {scene}/{basename}")
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid camera intrinsics: {scene}/{basename}")
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise ValueError(f"Non-positive focal length: {scene}/{basename}")
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image, depth, intrinsics, resolution, rng,
                preserve_fov=preserve_fov, sequence_aug=sequence_aug,
                info=str(image_path),
            )
            views.append(dict(
                img=image, depthmap=depth, camera_pose=pose,
                camera_intrinsics=intrinsics, dataset=self.dataset_name,
                label=f"{scene.name}/{basename}",
            ))
        return views
