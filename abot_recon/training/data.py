from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .config import DATASET_NAMES, DataConfig, LoaderConfig
from .datasets import (
    ARKitHR, ARKitScenes, BlendedMVSSeq, DL3DV, DynamicReplica, HyperSimSeq,
    MVSSynth, PointOdyssey, ScanNetPPSeq, Spring,
    ScanNet, TartanAir, TartanGround, UASOL, UnrealStereo4KSeq, VirtualKITTI2,
    Waymo, WildRGBD,
)


_CUT3R_DATASETS = {
    "tartanair": TartanAir,
    "pointodyssey": PointOdyssey,
    "spring": Spring,
    "mvs_synth": MVSSynth,
    "dynamic_replica": DynamicReplica,
    "uasol": UASOL,
    "arkit_hr": ARKitHR,
    "wildrgbd": WildRGBD,
    "unreal4k_seq": UnrealStereo4KSeq,
    "scannet": ScanNet,
    "waymo": Waymo,
    "vkitti2": VirtualKITTI2,
    "hypersim_seq": HyperSimSeq,
    "blendedmvs_seq": BlendedMVSSeq,
    "arkit": ARKitScenes,
    "scannetpp_seq": ScanNetPPSeq,
}


def collate_views(samples: list[list[dict]]) -> list[dict]:
    """Transpose batch-of-sequences into the original list-of-batched-views contract."""
    frames = len(samples[0])
    output = []
    for frame in range(frames):
        view = {}
        for key in samples[0][frame]:
            values = [sample[frame][key] for sample in samples]
            view[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        output.append(view)
    return output


def _loader(dataset, config: LoaderConfig, *, seed: int, shuffle: bool) -> DataLoader:
    if len(dataset) == 0:
        raise ValueError(
            f"{type(dataset).__name__} has no eligible sequences; "
            "check root, split, data.num_frames and allow_repeat"
        )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        drop_last=shuffle,
        collate_fn=collate_views,
        generator=generator,
    )


def build_dl3dv_dataloader(
    config: DataConfig, *, seed: int, training: bool = True
) -> DataLoader:
    item = config.dl3dv
    dataset = DL3DV(
        root=item.root,
        split="train" if training else "val",
        num_views=config.num_frames,
        resolution=(config.width, config.height),
        seed=seed,
        allow_repeat=item.allow_repeat,
        aug_crop=config.aug_crop if training else 0,
        aug_focal=config.aug_focal if training else 1.0,
        preserve_fov_prob=config.preserve_fov_prob,
        principal_align_skip_prob=config.principal_align_skip_prob if training else 0.0,
        sequence_consistent_aug_prob=(
            config.sequence_consistent_aug_prob if training else 0.0
        ),
        z_far=item.z_far,
        train_augmentation=training,
        sequence_blacklist_path="auto" if item.blacklist is None else item.blacklist,
        min_interval=item.min_interval,
        max_interval=item.max_interval,
    )
    return _loader(dataset, item, seed=seed, shuffle=training)


def build_tartanground_dataloader(
    config: DataConfig, *, seed: int, training: bool = True, eval_stride: int = 1
) -> DataLoader:
    if not training and eval_stride < 1:
        raise ValueError("eval_stride must be positive")
    item = config.tartanground
    dataset = TartanGround(
        root=item.root,
        split="train" if training else "val",
        num_views=config.num_frames if training else 128,
        resolution=(config.width, config.height),
        seed=seed,
        aug_crop=config.aug_crop if training else 0,
        aug_focal=config.aug_focal if training else 1.0,
        preserve_fov_prob=config.preserve_fov_prob if training else 0.0,
        principal_align_skip_prob=config.principal_align_skip_prob if training else 0.0,
        sequence_consistent_aug_prob=(
            config.sequence_consistent_aug_prob if training else 0.0
        ),
        z_far=item.z_far,
        train_augmentation=training,
        min_interval=item.min_interval if training else eval_stride,
        max_interval=item.max_interval if training else eval_stride,
        interval_by_env=None if training else {},
        forward_only=False,
        exclude_scenes=item.exclude_scenes,
        eval_only=not training,
        include_scenes=item.eval_scenes,
        max_starts_per_scene=item.eval_starts_per_scene if not training else None,
    )
    return _loader(dataset, item, seed=seed, shuffle=training)


def build_validation_loaders(config: DataConfig, *, seed: int) -> dict[str, DataLoader]:
    """Fixed 128-frame held-out streams, matching development stride-1/6 eval.

    Equal interval bounds make foldback sampling deterministic, including
    starts near scene boundaries; training clip length and augmentation do
    not affect either validation stream.
    """
    return {
        f"s{stride}": build_tartanground_dataloader(
            config, seed=seed, training=False, eval_stride=stride
        )
        for stride in (1, 6)
    }


def build_cut3r_dataloader(config: DataConfig, name: str, *, seed: int) -> DataLoader:
    """Build an opt-in source using the same augmentation and batching contract."""
    item = getattr(config, name)
    options = asdict(item)
    for key in ("weight", "batch_size", "num_workers", "pin_memory"):
        options.pop(key)
    if name in {"unreal4k_seq", "hypersim_seq", "blendedmvs_seq"}:
        # Spatial pose-graph sampling has no temporal interval bounds.
        options.pop("min_interval")
        options.pop("max_interval")
    dataset = _CUT3R_DATASETS[name](
        **options,
        num_views=config.num_frames,
        resolution=(config.width, config.height),
        seed=seed,
        aug_crop=config.aug_crop,
        aug_focal=config.aug_focal,
        preserve_fov_prob=config.preserve_fov_prob,
        principal_align_skip_prob=config.principal_align_skip_prob,
        sequence_consistent_aug_prob=config.sequence_consistent_aug_prob,
        train_augmentation=True,
    )
    return _loader(dataset, item, seed=seed, shuffle=True)


class MixedDataLoader:
    """Draw from independent dataset loaders with deterministic configured weights."""

    def __init__(self, loaders: dict[str, DataLoader], weights: dict[str, float], seed: int):
        self.loaders = loaders
        self.names = list(loaders)
        self.weights = [float(weights[name]) for name in self.names]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for loader in self.loaders.values():
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[dict]]:
        rng = random.Random(self.seed + self.epoch)
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            name = rng.choices(self.names, weights=self.weights, k=1)[0]
            try:
                yield next(iterators[name])
            except StopIteration:
                iterators[name] = iter(self.loaders[name])
                yield next(iterators[name])


def build_train_loaders(config: DataConfig, *, seed: int):
    builders = {
        "dl3dv": build_dl3dv_dataloader,
        "tartanground": build_tartanground_dataloader,
    }
    independent = {}
    for offset, name in enumerate(DATASET_NAMES):
        if getattr(config, name).weight <= 0:
            continue
        independent[name] = (
            builders[name](config, seed=seed + offset, training=True)
            if name in builders
            else build_cut3r_dataloader(config, name, seed=seed + offset)
        )
    if not independent:
        raise ValueError("At least one training dataset must have weight > 0")
    empty = [name for name, loader in independent.items() if len(loader) == 0]
    if empty:
        raise ValueError(
            "Training DataLoader(s) are empty: "
            + ", ".join(empty)
            + "; check roots, data.num_frames and per-rank batch sizes"
        )
    mixed = MixedDataLoader(
        independent,
        {name: getattr(config, name).weight for name in independent},
        seed,
    )
    return independent, mixed
