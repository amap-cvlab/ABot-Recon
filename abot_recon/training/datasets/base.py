from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset, get_worker_info

from . import cropping
from .transforms import (
    ImgToTensor,
    JitterJpegLossBlurring,
    apply_transform_with_sequence_params,
    sample_sequence_transform_params,
)


_MEAN_RGB = (0.485, 0.456, 0.406)
_CONTENT_BBOX_INFO_KEY = "_abot_recon_content_bbox"


def _unproject(depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth.astype(np.float32)
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1).astype(np.float32)
    valid = np.isfinite(points).all(axis=-1) & (z > 0)
    return points, valid


def _to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.einsum("ij,hwj->hwi", pose[:3, :3], points) + pose[:3, 3]


class MultiViewDataset(Dataset):
    """Shared data path matching the current Pi3/CUT3R training contract."""

    def __init__(
        self,
        *,
        root: str,
        num_views: int,
        resolution: tuple[int, int] = (504, 280),
        split: str = "train",
        seed: int = 2024,
        allow_repeat: bool = False,
        aug_crop: int = 16,
        aug_focal: float = 0.9,
        preserve_fov_prob: float = 0.9,
        principal_align_skip_prob: float = 0.0,
        sequence_consistent_aug_prob: float = 0.2,
        z_far: float = 0.0,
        train_augmentation: bool = True,
        max_refetch: int = 15,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.ROOT = str(self.root)
        self.num_views = int(num_views)
        self.resolution = tuple(int(value) for value in resolution)
        self.split = str(split)
        self.seed = int(seed)
        self.epoch = 0
        # Persistent workers own independent dataset copies.  Keep an advancing
        # RNG in each copy, as in the production loader, instead of attempting
        # to reseed workers by mutating the parent dataset at epoch boundaries.
        self._rng = None
        self.allow_repeat = bool(allow_repeat)
        self.aug_crop = int(aug_crop)
        self.aug_focal = float(aug_focal)
        self.preserve_fov_prob = float(preserve_fov_prob)
        self.principal_align_skip_prob = float(principal_align_skip_prob)
        if not 0.0 <= self.principal_align_skip_prob <= 1.0:
            raise ValueError("principal_align_skip_prob must be in [0, 1]")
        self.sequence_consistent_aug_prob = float(sequence_consistent_aug_prob)
        if not 0.0 <= self.sequence_consistent_aug_prob <= 1.0:
            raise ValueError("sequence_consistent_aug_prob must be in [0, 1]")
        self.z_far = float(z_far)
        self.max_refetch = int(max_refetch)
        self.transform = JitterJpegLossBlurring if train_augmentation else ImgToTensor

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _next_rng(self) -> np.random.Generator:
        """Derive a fresh per-sample RNG from this worker's persistent stream."""
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = worker.seed if worker is not None else self.seed
            self._rng = np.random.default_rng(worker_seed)
        sample_seed = int(
            self._rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        return np.random.default_rng(sample_seed)

    @staticmethod
    def blockwise_shuffle(values, rng, block_size):
        if block_size is None:
            return rng.permutation(values).tolist()
        blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]
        return [item for block in blocks for item in rng.permutation(block).tolist()]

    @staticmethod
    def adaptive_interval_bounds(remaining, num_views, min_interval=1, max_interval=25):
        if num_views < 2:
            return max(1, int(min_interval)), max(1, int(max_interval))
        cap = int(remaining) // (int(num_views) - 1)
        upper = min(max(1, int(max_interval)), max(1, cap))
        lower = max(1, min(int(min_interval), upper))
        return lower, max(lower, upper)

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        if min_interval <= 0 or min_interval > max_interval:
            raise ValueError("invalid interval bounds")
        pos_ref = ids_all.index(id_ref)
        if num_views == 1:
            return [pos_ref], True
        possible = np.arange(pos_ref, len(ids_all))
        remaining = len(ids_all) - 1 - pos_ref
        if remaining >= num_views - 1:
            if remaining == num_views - 1:
                return [pos_ref + index for index in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]
            is_video = rng.random() < video_prob
            if is_video and rng.random() < fix_interval_prob:
                stride = rng.choice(
                    range(1, min(remaining // (num_views - 1) + 1, max_interval + 1))
                )
                intervals = [stride] * (num_views - 1)
            positions = [p for p in itertools.accumulate([pos_ref] + intervals) if p < len(ids_all)]
            candidates = [p for p in possible if p not in positions]
            positions += rng.choice(candidates, num_views - len(positions), replace=False).tolist()
            positions = (
                sorted(positions)
                if is_video
                else self.blockwise_shuffle(positions, rng, block_shuffle)
            )
        else:
            unique = max(remaining, 2)
            new_start = int(rng.choice(np.arange(pos_ref + 1)))
            new_remaining = len(ids_all) - 1 - new_start
            upper = max(1, min(max_interval, new_remaining // (unique - 1)))
            intervals = [rng.choice(range(1, upper + 1)) for _ in range(unique - 1)]
            revisit_random, video_random = rng.random(), rng.random()
            if rng.random() < fix_interval_prob and video_random < video_prob:
                stride = rng.choice(range(1, upper + 1))
                intervals = [stride] * (unique - 1)
            positions = list(itertools.accumulate([new_start] + intervals))
            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:
                is_video = video_random < video_prob
                if not is_video:
                    positions = self.blockwise_shuffle(positions, rng, block_shuffle)
                repeats, tail = divmod(num_views, len(positions))
                positions = positions * repeats + positions[:tail]
            elif revisit_random < 0.9:
                positions = rng.choice(positions, num_views, replace=True).tolist()
            else:
                positions = sorted(rng.choice(positions, num_views, replace=True).tolist())
        if len(positions) != num_views:
            raise RuntimeError("sequence sampler returned the wrong number of frames")
        return positions, is_video

    def _crop_resize_if_necessary(
        self,
        image,
        depth,
        intrinsics,
        resolution,
        rng,
        *,
        preserve_fov,
        sequence_aug=None,
        info=None,
    ):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        width, height = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        margin_x, margin_y = min(cx, width - cx), min(cy, height - cy)
        if margin_x <= width / 5 or margin_y <= height / 5:
            raise ValueError(f"Bad principal point in view={info}")
        if not (sequence_aug or {}).get("skip_principal_align", False):
            image, depth, intrinsics, _, _ = cropping.crop_image_depthmap(
                image,
                depth,
                intrinsics,
                (cx - margin_x, cy - margin_y, cx + margin_x, cy + margin_y),
            )
        crop_scale = (
            sequence_aug["crop_scale"]
            if sequence_aug is not None and "crop_scale" in sequence_aug
            else self._sample_crop_scale(rng)
        )
        image, depth, intrinsics, _, _ = cropping.center_crop_image_depthmap(
            image, depth, intrinsics, crop_scale
        )
        if preserve_fov:
            image, depth, intrinsics, _, _, content_bbox = (
                cropping.resize_to_width_and_crop_or_pad(
                    image, depth, intrinsics, resolution, pad_rgb=_MEAN_RGB
                )
            )
            full_bbox = (0, 0, int(resolution[0]), int(resolution[1]))
            if content_bbox != full_bbox:
                image.info[_CONTENT_BBOX_INFO_KEY] = content_bbox
            return image, depth, intrinsics
        crop_delta = (
            sequence_aug["crop_delta"]
            if sequence_aug is not None and "crop_delta" in sequence_aug
            else self._sample_crop_delta(rng)
        )
        target = np.asarray(resolution) + crop_delta
        image, depth, intrinsics, _, _ = cropping.rescale_image_depthmap(
            image, depth, intrinsics, target
        )
        output_intrinsics = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, output_intrinsics, resolution
        )
        image, depth, output_intrinsics, _, _ = cropping.crop_image_depthmap(
            image, depth, intrinsics, bbox
        )
        return image, depth, output_intrinsics

    def _transform_padded_image_content(
        self,
        image: PIL.Image.Image,
        content_bbox: tuple[int, int, int, int],
        sequence_params,
    ) -> torch.Tensor:
        """Augment only real pixels and restore exact mean-color padding."""
        left, top, right, bottom = (int(value) for value in content_bbox)
        width, height = image.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(
                f"invalid content bbox {content_bbox} for padded image size {image.size}"
            )
        content = image.crop((left, top, right, bottom))
        transformed = apply_transform_with_sequence_params(
            self.transform, content, sequence_params
        )
        if not torch.is_tensor(transformed) or transformed.ndim != 3:
            raise TypeError(
                "padded-image transform must return a [C,H,W] tensor, got "
                f"{type(transformed).__name__} shape={getattr(transformed, 'shape', None)}"
            )
        expected_shape = (bottom - top, right - left)
        if tuple(transformed.shape[-2:]) != expected_shape or transformed.shape[0] != 3:
            raise ValueError(
                "padded-image transform changed the content shape: "
                f"expected (3, {expected_shape[0]}, {expected_shape[1]}), "
                f"got {tuple(transformed.shape)}"
            )
        fill = transformed.new_tensor(_MEAN_RGB).reshape(3, 1, 1)
        canvas = fill.expand(3, height, width).clone()
        canvas[:, top:bottom, left:right] = transformed
        return canvas

    def _sample_crop_scale(self, rng) -> float:
        return float(
            self.aug_focal + (1.0 - self.aug_focal) * rng.beta(0.5, 0.5)
        )

    def _sample_crop_delta(self, rng) -> int:
        return int(rng.integers(0, self.aug_crop)) if self.aug_crop > 1 else 0

    def _sample_sequence_aug(self, rng) -> dict[str, float | int]:
        return {
            "crop_scale": self._sample_crop_scale(rng),
            "crop_delta": self._sample_crop_delta(rng),
        }

    def _get_views(
        self, idx, resolution, rng, num_views, preserve_fov, sequence_aug
    ):
        raise NotImplementedError

    def __getitem__(self, request):
        if isinstance(request, (tuple, list, np.ndarray)):
            index, _, num_views = (int(value) for value in request)
        else:
            index, num_views = int(request), self.num_views
        original = index
        for attempt in range(self.max_refetch):
            rng = self._next_rng()
            try:
                sequence_consistent = bool(
                    self.sequence_consistent_aug_prob > 0.0
                    and rng.random() < self.sequence_consistent_aug_prob
                )
                sequence_aug = (
                    self._sample_sequence_aug(rng) if sequence_consistent else None
                )
                preserve_fov = bool(
                    self.preserve_fov_prob > 0.0
                    and rng.random() < self.preserve_fov_prob
                )
                if self.principal_align_skip_prob > 0.0:
                    # Independent sequence-level choice; default 0 preserves the RNG stream.
                    if sequence_aug is None:
                        sequence_aug = {}
                    sequence_aug["skip_principal_align"] = bool(
                        rng.random() < self.principal_align_skip_prob
                    )
                views = self._get_views(
                    index,
                    self.resolution,
                    rng,
                    num_views,
                    preserve_fov,
                    sequence_aug,
                )
                sequence_params = (
                    sample_sequence_transform_params(self.transform, rng)
                    if sequence_consistent
                    else None
                )
                output = []
                for view in views:
                    image = view.pop("img")
                    content_bbox = (
                        image.info.pop(_CONTENT_BBOX_INFO_KEY, None)
                        if isinstance(image, PIL.Image.Image)
                        else None
                    )
                    if content_bbox is None:
                        image_tensor = apply_transform_with_sequence_params(
                            self.transform, image, sequence_params
                        )
                    else:
                        image_tensor = self._transform_padded_image_content(
                            image, content_bbox, sequence_params
                        )
                    depth = np.nan_to_num(view.pop("depthmap"), nan=0, posinf=0, neginf=0)
                    intrinsics = np.asarray(view["camera_intrinsics"], dtype=np.float32)
                    pose = np.asarray(view["camera_pose"], dtype=np.float32)
                    local_points, valid = _unproject(depth, intrinsics)
                    if self.z_far > 0:
                        valid &= depth < self.z_far
                    world_points = _to_world(local_points, pose)
                    output.append(
                        {
                            "img": image_tensor,
                            "pts3d": torch.from_numpy(world_points),
                            "valid_mask": torch.from_numpy(valid),
                            "camera_pose": torch.from_numpy(pose),
                            "camera_intrinsics": torch.from_numpy(intrinsics),
                            "dataset": str(view.get("dataset", self.dataset_name)),
                            "label": str(view.get("label", "")),
                            "camera_only": bool(view.get("camera_only", False)),
                        }
                    )
                return output
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                if attempt + 1 == self.max_refetch:
                    raise RuntimeError(
                        f"Failed to load {self.dataset_name} sample {original} after "
                        f"{self.max_refetch} attempts"
                    ) from error
                index = int(self._rng.integers(0, len(self)))
        raise AssertionError("unreachable")
