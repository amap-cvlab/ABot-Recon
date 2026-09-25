import math
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from abot_recon.training.config import (
    DL3DVConfig,
    DataConfig,
    TartanGroundConfig,
    TrainConfig,
    load_config,
)
from abot_recon.training.data import (
    build_dl3dv_dataloader,
    build_tartanground_dataloader,
    build_train_loaders,
    collate_views,
)
from abot_recon.training.datasets.base import MultiViewDataset
from abot_recon.training.datasets.transforms import ImgToTensor
from abot_recon.training.loss import CameraLossConfig, Pi3Loss, PointLoss
from abot_recon.training.ema import ModelEMA
from abot_recon.training.trainer import (
    _restore_scheduler_progress,
    build_optimizer,
    configure_trainable_scope,
    run_training,
)


def test_release_recipe_defaults():
    cfg = load_config("configs/finetune.yaml")
    assert (cfg.epochs, cfg.steps_per_epoch, cfg.gradient_accumulation_steps) == (20, 800, 1)
    assert (cfg.seed, cfg.data_seed) == (1111, 666)
    assert cfg.trainable_scope == "all"
    assert not cfg.enable_confidence
    assert (
        cfg.learning_rate,
        cfg.gate_learning_rate,
        cfg.corr_learning_rate,
        cfg.confidence_learning_rate,
    ) == (1e-6, 1e-6, 1e-6, 1e-6)
    assert (cfg.data.num_frames, cfg.data.width, cfg.data.height) == (32, 504, 280)
    assert cfg.adam_beta2 == 0.999
    assert cfg.validation_steps == -1
    assert cfg.data.tartanground.eval_starts_per_scene == 4
    assert cfg.data.sequence_consistent_aug_prob == 0.2
    assert cfg.data.dl3dv.weight / cfg.data.tartanground.weight == 100_000 / 75_000
    assert (cfg.data.dl3dv.z_far, cfg.data.tartanground.z_far) == (0.0, 80.0)
    assert cfg.loss.camera_pair_mode == "causal_upper"
    assert cfg.loss.camera_max_pair_distance == 11
    assert cfg.loss.camera_rotation_gap_weight == "pow0.75"
    assert (cfg.loss.camera_alpha_corr_magnitude, cfg.loss.camera_alpha_corr_smooth) == (
        1e-3,
        1e-3,
    )


@pytest.mark.parametrize("steps", [-2, 0])
def test_validation_steps_reject_invalid_limits(tmp_path, steps):
    cfg = TrainConfig(data=_data_config(tmp_path), validation_steps=steps)
    path = tmp_path / "validation_steps.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    with pytest.raises(ValueError, match="validation_steps must be -1"):
        load_config(path)


def test_validation_capacity_is_independent_of_training_length(tmp_path):
    cfg = TrainConfig(data=_data_config(tmp_path), max_frames=64)
    path = tmp_path / "validation_capacity.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    with pytest.raises(ValueError, match="at least 128"):
        load_config(path)


def test_release_loss_excludes_inactive_experimental_objectives():
    fields = set(CameraLossConfig.__dataclass_fields__)
    removed = {
        "alpha_direction",
        "alpha_consistency",
        "alpha_rotation_step",
        "alpha_rotation_zero_mean",
        "alpha_rotation_cumsum",
        "alpha_hard",
        "alpha_easy_identity",
        "alpha_noharm",
        "alpha_hard_improve",
    }
    assert fields.isdisjoint(removed)


def test_normal_loss_matches_source_four_triangle_objective():
    target = torch.tensor(
        [[[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]]
    )
    predicted = torch.tensor(
        [[[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]]]],
        requires_grad=True,
    )
    loss = PointLoss.normal_loss(predicted, target, torch.ones(target.shape[:-1], dtype=torch.bool))
    expected = (math.pi / 2 - math.radians(3.0) / 2) / 2
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()


def _camera():
    return np.array([[20, 0, 14], [0, 20, 7], [0, 0, 1]], np.float32), np.eye(4, dtype=np.float32)


def _write_dl3dv(root):
    dense = root / "bucket" / "scene" / "dense"
    for folder in ("rgb", "depth", "cam", "sky_mask", "outlier_mask"):
        (dense / folder).mkdir(parents=True, exist_ok=True)
    (root / "dl3dv_geometry_blacklist_20260810.txt").write_text("")
    intrinsic, pose = _camera()
    for index in range(4):
        name = f"{index:05d}"
        image = np.full((14, 28, 3), 96 + index, np.uint8)
        cv2.imwrite(str(dense / "rgb" / f"{name}.png"), image)
        cv2.imwrite(str(dense / "sky_mask" / f"{name}.png"), np.zeros((14, 28), np.uint8))
        cv2.imwrite(str(dense / "outlier_mask" / f"{name}.png"), np.zeros((14, 28), np.uint8))
        np.save(dense / "depth" / f"{name}.npy", np.ones((14, 28), np.float32))
        np.savez(dense / "cam" / f"{name}.npz", intrinsic=intrinsic, pose=pose)


def _write_tartanground(root):
    sequence = root / "FakeEnv__omni__P0001__rcam_front"
    for folder in ("images", "depths", "cameras", "masks"):
        (sequence / folder).mkdir(parents=True, exist_ok=True)
    intrinsic, pose = _camera()
    for index in range(4):
        name = f"{index:05d}"
        cv2.imwrite(str(sequence / "images" / f"{name}.jpg"), np.full((14, 28, 3), 120, np.uint8))
        np.save(sequence / "depths" / f"{name}.npy", np.ones((14, 28), np.float32))
        np.save(sequence / "masks" / f"{name}.npy", np.zeros((14, 28), bool))
        np.savez(sequence / "cameras" / f"{name}.npz", camera_intrinsics=intrinsic, camera_pose=pose)


def _data_config(tmp_path):
    return DataConfig(
        num_frames=3,
        height=14,
        width=28,
        aug_crop=0,
        aug_focal=1.0,
        dl3dv=DL3DVConfig(root=str(tmp_path / "dl3dv"), batch_size=1, num_workers=0, max_interval=1),
        tartanground=TartanGroundConfig(root=str(tmp_path / "tg"), batch_size=1, num_workers=0, max_interval=1, exclude_scenes=[]),
    )


def test_each_dataset_has_an_independent_real_loader(tmp_path):
    _write_dl3dv(tmp_path / "dl3dv")
    _write_tartanground(tmp_path / "tg")
    cfg = _data_config(tmp_path)
    dl_batch = next(iter(build_dl3dv_dataloader(cfg, seed=1)))
    tg_batch = next(iter(build_tartanground_dataloader(cfg, seed=2)))
    assert len(dl_batch) == len(tg_batch) == 3
    assert dl_batch[0]["dataset"] == ["dl3dv"]
    assert tg_batch[0]["dataset"] == ["tartanground"]
    assert dl_batch[0]["img"].shape == tg_batch[0]["img"].shape == (1, 3, 14, 28)


@pytest.mark.parametrize("active", ["dl3dv", "tartanground"])
def test_zero_weight_skips_missing_dataset(tmp_path, active):
    data = _data_config(tmp_path)
    if active == "dl3dv":
        _write_dl3dv(tmp_path / "dl3dv")
        data.tartanground.root = ""
        data.tartanground.weight = 0
    else:
        _write_tartanground(tmp_path / "tg")
        data.dl3dv.root = ""
        data.dl3dv.weight = 0
    path = tmp_path / "single_dataset.yaml"
    OmegaConf.save(OmegaConf.structured(TrainConfig(data=data, validation_enabled=False)), path)
    cfg = load_config(path)
    loaders, mixed = build_train_loaders(cfg.data, seed=4)
    assert list(loaders) == mixed.names == [active]
    batches = iter(mixed)
    for _ in range(5):  # includes an iterator restart for the small fixture
        batch = next(batches)
        assert len(batch) == 3
        assert batch[0]["dataset"] == [active]


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_dataset_weights_are_rejected(tmp_path, weight):
    cfg = TrainConfig(data=_data_config(tmp_path))
    cfg.data.dl3dv.weight = weight
    path = tmp_path / "invalid_weight.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    with pytest.raises(ValueError, match="invalid weight"):
        load_config(path)


def test_all_datasets_disabled_is_rejected(tmp_path):
    cfg = TrainConfig(data=_data_config(tmp_path), validation_enabled=False)
    cfg.data.dl3dv.weight = cfg.data.tartanground.weight = 0
    path = tmp_path / "empty_mixture.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    with pytest.raises(ValueError, match="At least one"):
        load_config(path)
    with pytest.raises(ValueError, match="At least one"):
        build_train_loaders(cfg.data, seed=4)


def test_validation_requires_tartanground_root_when_training_disabled(tmp_path):
    cfg = TrainConfig(data=_data_config(tmp_path))
    cfg.data.tartanground.root = ""
    cfg.data.tartanground.weight = 0
    path = tmp_path / "missing_validation.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    with pytest.raises(ValueError, match="required for validation"):
        load_config(path)


class _PersistentRngProbeDataset(MultiViewDataset):
    dataset_name = "probe"

    def __len__(self):
        return 1

    def _get_views(
        self,
        _idx,
        _resolution,
        rng,
        num_views,
        _preserve_fov,
        _sequence_aug,
    ):
        intrinsic, pose = _camera()
        label = str(int(rng.integers(0, np.iinfo(np.int64).max)))
        return [
            {
                "img": PIL.Image.fromarray(np.full((14, 28, 3), 127, np.uint8)),
                "depthmap": np.ones((14, 28), np.float32),
                "camera_intrinsics": intrinsic.copy(),
                "camera_pose": pose.copy(),
                "label": label,
            }
            for _ in range(num_views)
        ]


def test_persistent_worker_rng_advances_between_epochs(tmp_path):
    dataset = _PersistentRngProbeDataset(
        root=str(tmp_path),
        num_views=1,
        resolution=(28, 14),
        aug_crop=0,
        aug_focal=1.0,
        preserve_fov_prob=0.0,
        train_augmentation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        collate_fn=collate_views,
        shuffle=False,
    )
    first = next(iter(loader))[0]["label"][0]
    dataset.set_epoch(1)
    second = next(iter(loader))[0]["label"][0]
    assert first != second


class _TransformPipeline:
    def __init__(self, *steps):
        self.transforms = list(steps)

    def __call__(self, image):
        for step in self.transforms:
            image = step(image)
        return image


class _StampPhotometricStep:
    def __init__(self):
        self.samples = 0

    def sample_params(self, _rng=None):
        self.samples += 1
        return {"value": self.samples * 32}

    @staticmethod
    def apply_with_params(image, params):
        value = int(params["value"])
        return PIL.Image.fromarray(
            np.full((image.height, image.width, 3), value, dtype=np.uint8)
        )

    def __call__(self, image):
        return self.apply_with_params(image, self.sample_params())


@pytest.mark.parametrize(
    ("probability", "expected_unique_frames"),
    [(1.0, 1), (0.0, 3)],
)
def test_sequence_photometric_parameters_are_mixed_by_probability(
    tmp_path, probability, expected_unique_frames
):
    dataset = _PersistentRngProbeDataset(
        root=str(tmp_path),
        num_views=3,
        resolution=(28, 14),
        aug_crop=0,
        aug_focal=1.0,
        preserve_fov_prob=0.0,
        sequence_consistent_aug_prob=probability,
        train_augmentation=False,
    )
    dataset.transform = _TransformPipeline(_StampPhotometricStep(), ImgToTensor)
    sample = dataset[0]
    frame_values = {float(view["img"][0, 0, 0]) for view in sample}
    assert len(frame_values) == expected_unique_frames


class _FillWhite:
    def __call__(self, image):
        return PIL.Image.fromarray(
            np.full((image.height, image.width, 3), 255, dtype=np.uint8)
        )


class _PaddingProbeDataset(MultiViewDataset):
    dataset_name = "padding_probe"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preserve_fov_choices = []

    def __len__(self):
        return 1

    def _get_views(
        self, _idx, resolution, rng, num_views, preserve_fov, sequence_aug
    ):
        self.preserve_fov_choices.append(preserve_fov)
        intrinsics = np.array(
            [[20.0, 0.0, 14.0], [0.0, 20.0, 3.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pose = np.eye(4, dtype=np.float32)
        views = []
        for _ in range(num_views):
            image, depth, transformed_intrinsics = self._crop_resize_if_necessary(
                PIL.Image.fromarray(np.zeros((6, 28, 3), dtype=np.uint8)),
                np.ones((6, 28), dtype=np.float32),
                intrinsics.copy(),
                resolution,
                rng,
                preserve_fov=preserve_fov,
                sequence_aug=sequence_aug,
            )
            views.append(
                {
                    "img": image,
                    "depthmap": depth,
                    "camera_intrinsics": transformed_intrinsics,
                    "camera_pose": pose.copy(),
                }
            )
        return views


class _JointAugProbeDataset(_PaddingProbeDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crop_scale_samples = 0
        self.crop_delta_samples = 0

    def _sample_crop_scale(self, _rng):
        self.crop_scale_samples += 1
        return 1.0

    def _sample_crop_delta(self, _rng):
        self.crop_delta_samples += 1
        return self.crop_delta_samples


@pytest.mark.parametrize(
    ("probability", "expected_samples"),
    [(1.0, 1), (0.0, 3)],
)
def test_geometry_and_photometric_sequence_consistency_share_one_decision(
    tmp_path, probability, expected_samples
):
    photometric = _StampPhotometricStep()
    dataset = _JointAugProbeDataset(
        root=str(tmp_path),
        num_views=3,
        resolution=(28, 14),
        aug_crop=16,
        aug_focal=0.9,
        preserve_fov_prob=0.0,
        sequence_consistent_aug_prob=probability,
        train_augmentation=False,
    )
    dataset.transform = _TransformPipeline(photometric, ImgToTensor)

    dataset[0]

    assert dataset.crop_scale_samples == expected_samples
    assert dataset.crop_delta_samples == expected_samples
    assert photometric.samples == expected_samples


class _ScriptedRng:
    def __init__(self, random_values):
        self._generator = np.random.default_rng(0)
        self._random_values = iter(random_values)
        self.random_calls = 0

    def random(self):
        self.random_calls += 1
        return next(self._random_values)

    def __getattr__(self, name):
        return getattr(self._generator, name)


def test_preserve_fov_mode_is_sampled_once_per_sequence(tmp_path):
    dataset = _PaddingProbeDataset(
        root=str(tmp_path),
        num_views=3,
        resolution=(28, 14),
        aug_crop=0,
        aug_focal=1.0,
        preserve_fov_prob=0.5,
        sequence_consistent_aug_prob=0.0,
        train_augmentation=False,
    )
    rngs = [
        _ScriptedRng([0.25, 0.75]),
        _ScriptedRng([0.75, 0.25]),
    ]
    rng_iterator = iter(rngs)

    def next_rng():
        return next(rng_iterator)

    dataset._next_rng = next_rng

    views = dataset[0]
    dataset[0]

    assert dataset.preserve_fov_choices == [True, False]
    assert [rng.random_calls for rng in rngs] == [1, 1]
    mean = views[0]["img"].new_tensor((0.485, 0.456, 0.406)).reshape(3, 1, 1)
    for view in views:
        assert torch.allclose(view["img"][:, :4], mean.expand(3, 4, 28))


def test_photometric_augmentation_does_not_modify_synthetic_padding(tmp_path):
    dataset = _PaddingProbeDataset(
        root=str(tmp_path),
        num_views=1,
        resolution=(28, 14),
        aug_crop=0,
        aug_focal=1.0,
        preserve_fov_prob=1.0,
        sequence_consistent_aug_prob=0.0,
        train_augmentation=False,
    )
    dataset.transform = _TransformPipeline(_FillWhite(), ImgToTensor)
    image = dataset[0][0]["img"]
    mean = image.new_tensor((0.485, 0.456, 0.406)).reshape(3, 1, 1)
    assert torch.allclose(image[:, :4], mean.expand(3, 4, 28))
    assert torch.equal(image[:, 4:10], torch.ones(3, 6, 28))
    assert torch.allclose(image[:, 10:], mean.expand(3, 4, 28))


class TinyCorrModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_backbone = torch.nn.Parameter(torch.tensor(1.0))
        self.rot_correction = torch.nn.Parameter(torch.tensor(0.05))

    def inference_stream(self, images, **kwargs):
        # This synthetic model has no attention/cache; exercise the trainer API.
        return self.forward(images, **kwargs)

    def forward(self, images, **_kwargs):
        batch, frames, _, height, width = images.shape
        points = torch.zeros(batch, frames, height, width, 3, device=images.device)
        points[..., 2] = self.frozen_backbone
        angle = self.rot_correction
        one, zero = torch.ones_like(angle), torch.zeros_like(angle)
        rotation = torch.stack((one, zero, zero, zero, angle.cos(), -angle.sin(), zero, angle.sin(), angle.cos())).reshape(3, 3)
        step = torch.eye(4, device=images.device).clone()
        step = torch.cat((torch.cat((rotation, step[:3, 3:]), dim=1), step[3:]), dim=0)
        poses = [torch.eye(4, device=images.device).expand(batch, 4, 4)]
        for _ in range(1, frames):
            poses.append(poses[-1] @ step)
        poses = torch.stack(poses, dim=1)
        residual = torch.stack((angle.expand(batch, frames - 1), torch.zeros(batch, frames - 1, device=images.device), torch.zeros(batch, frames - 1, device=images.device)), dim=-1)
        return {"local_points": points, "camera_poses": poses, "camera_state": {"rotation_residual": residual}}


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, value):
        return value.clone()

    @staticmethod
    def backward(_ctx, grad_output):
        return torch.full_like(grad_output, float("inf"))


class _GuardCriterion(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, prediction, _batch):
        anchor = prediction["camera_state"]["rotation_residual"].sum()
        if self.mode == "large":
            loss = anchor * 0 + 2.0
        elif self.mode == "nonfinite_loss":
            loss = anchor * float("nan")
        elif self.mode == "nonfinite_gradient":
            loss = _FiniteForwardInfiniteBackward.apply(anchor)
        else:
            raise AssertionError(self.mode)
        return loss, {"loss": loss}


@pytest.mark.parametrize(
    ("mode", "metric"),
    [
        ("large", "loss_exceeds_max"),
        ("nonfinite_loss", "finite"),
        ("nonfinite_gradient", "nonfinite_gradient"),
    ],
)
def test_training_guard_skips_bad_updates(tmp_path, mode, metric):
    _write_dl3dv(tmp_path / "dl3dv")
    _write_tartanground(tmp_path / "tg")
    data = _data_config(tmp_path)
    cfg = TrainConfig(
        output_dir=str(tmp_path / mode),
        epochs=1,
        steps_per_epoch=1,
        trainable_scope="rot_correction_only",
        mixed_precision="no",
        max_loss=0.5,
        corr_learning_rate=1e-2,
        checkpoint_every_epochs=2,
        validation_enabled=False,
        ema_enabled=False,
        data=data,
    )
    loaders, mixed = build_train_loaders(data, seed=4)
    model = TinyCorrModel()
    before = model.rot_correction.detach().clone()
    history = run_training(
        cfg,
        model,
        loaders,
        mixed,
        accelerator=Accelerator(cpu=True, mixed_precision="no"),
        criterion=_GuardCriterion(mode),
    )
    assert torch.equal(model.rot_correction, before)
    assert history[0]["skipped"] == 1
    if mode == "nonfinite_loss":
        assert history[0][metric] == 0
    else:
        assert history[0][metric] == 1


class TinyScopeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)
        self.camera_head = torch.nn.Module()
        self.camera_head.rot_correction = torch.nn.Linear(2, 2)
        self.gate = torch.nn.Linear(2, 2)
        self.conf_decoder = torch.nn.Linear(2, 2)
        self.conf_head = torch.nn.Linear(2, 1)


def test_encoder_is_frozen_for_every_trainable_scope():
    for scope in ("all", "heads", "rot_correction_only"):
        model = TinyScopeModel()
        configure_trainable_scope(model, scope)
        states = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        assert not any(enabled for name, enabled in states.items() if name.startswith("encoder."))
        assert not any(enabled for name, enabled in states.items() if name.startswith("conf_"))
        if scope == "all":
            assert all(
                enabled
                for name, enabled in states.items()
                if not name.startswith(("encoder.", "conf_"))
            )
        elif scope == "rot_correction_only":
            assert all(enabled == ("rot_correction" in name) for name, enabled in states.items())


def test_confidence_is_independently_enabled_for_every_scope():
    for scope in ("all", "heads", "rot_correction_only"):
        model = TinyScopeModel()
        configure_trainable_scope(model, scope, train_confidence=True)
        states = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        assert all(enabled for name, enabled in states.items() if name.startswith("conf_"))
        assert not any(enabled for name, enabled in states.items() if name.startswith("encoder."))


def test_learning_rates_are_user_configurable(tmp_path):
    raw = OmegaConf.load("configs/finetune.yaml")
    raw.learning_rate = 2e-6
    raw.gate_learning_rate = 3e-6
    raw.corr_learning_rate = 4e-6
    raw.confidence_learning_rate = 5e-6
    path = tmp_path / "custom.yaml"
    OmegaConf.save(raw, path)
    cfg = load_config(path)
    model = TinyScopeModel()
    configure_trainable_scope(model, "all", train_confidence=True)
    optimizer = build_optimizer(model, cfg)
    by_name = {}
    for group in optimizer.param_groups:
        by_name.setdefault(group["group_name"], set()).add(group["lr"])
    assert by_name == {
        "other": {2e-6},
        "gate": {3e-6},
        "corr": {4e-6},
        "confidence": {5e-6},
    }


class TinyConfidenceModel(TinyCorrModel):
    def __init__(self):
        super().__init__()
        self.conf_decoder = torch.nn.Linear(1, 1)
        self.conf_head = torch.nn.Linear(1, 1)
        self.train_conf = True

    def forward(self, images, **kwargs):
        output = super().forward(images, **kwargs)
        if self.train_conf:
            batch, frames, _, height, width = images.shape
            features = torch.ones(batch, frames, height, width, 1, device=images.device)
            output["conf"] = self.conf_head(self.conf_decoder(features))
        return output


def test_optional_confidence_loss_has_gradients_and_no_unused_trainable_parameters(tmp_path):
    _write_tartanground(tmp_path / "tg")
    cfg = _data_config(tmp_path)
    batch = next(iter(build_tartanground_dataloader(cfg, seed=2)))
    model = TinyConfidenceModel()
    configure_trainable_scope(model, "all", train_confidence=True)
    prediction = model(torch.stack([view["img"] for view in batch], dim=1))
    criterion = Pi3Loss(train_confidence=True, confidence_invalid_as_zero=True)
    loss, metrics = criterion(prediction, batch)
    loss.backward()
    assert "confidence_loss" in metrics
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("conf_") and parameter.requires_grad
    )


def test_device_resident_ema_updates_swaps_and_restores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyScopeModel().to(device)
    configure_trainable_scope(model, "rot_correction_only")
    ema = ModelEMA(model, decay=0.5, trainable_only=True)
    name = ema.param_names[0]
    parameter = dict(model.named_parameters())[name]
    assert ema.shadow[name].device == parameter.device
    before = ema.shadow[name].clone()
    parameter.data.add_(2)
    online = parameter.detach().clone()
    ema.update(model)
    torch.testing.assert_close(ema.shadow[name], before + 1)
    assert ema.num_updates == 1
    with ema.apply(model):
        torch.testing.assert_close(parameter, before + 1)
    torch.testing.assert_close(parameter, online)


def test_confidence_enabled_trainer_step_and_ema_checkpoint(tmp_path):
    _write_dl3dv(tmp_path / "dl3dv")
    _write_tartanground(tmp_path / "tg")
    data = _data_config(tmp_path)
    cfg = TrainConfig(
        output_dir=str(tmp_path / "confidence_out"),
        epochs=1,
        steps_per_epoch=1,
        trainable_scope="all",
        enable_confidence=True,
        mixed_precision="no",
        checkpoint_every_epochs=1,
        validation_enabled=False,
        data=data,
    )
    loaders, mixed = build_train_loaders(data, seed=4)
    model = TinyConfidenceModel()
    history = run_training(
        cfg,
        model,
        loaders,
        mixed,
        accelerator=Accelerator(cpu=True, mixed_precision="no"),
    )
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["confidence_loss"]))
    checkpoint = tmp_path / "confidence_out" / "checkpoint-0001"
    merged = load_file(str(checkpoint / "abot_recon_ema.safetensors"))
    assert any(name.startswith("conf_decoder.") for name in merged)
    assert any(name.startswith("conf_head.") for name in merged)


def test_manual_ema_merge_script_produces_complete_checkpoint(tmp_path):
    online = {
        "frozen": torch.tensor([1.0]),
        "trained": torch.tensor([2.0]),
    }
    save_file(online, str(tmp_path / "online.safetensors"))
    torch.save(
        {"model": {"trained": torch.tensor([3.0])}, "trainable_only": True},
        tmp_path / "ema.pt",
    )
    output = tmp_path / "merged.safetensors"
    subprocess.run(
        [
            sys.executable,
            "scripts/merge_ema_checkpoint.py",
            "--model",
            str(tmp_path / "online.safetensors"),
            "--ema",
            str(tmp_path / "ema.pt"),
            "--output",
            str(output),
        ],
        check=True,
    )
    merged = load_file(str(output))
    torch.testing.assert_close(merged["frozen"], online["frozen"])
    torch.testing.assert_close(merged["trained"], torch.tensor([3.0]))


def test_active_corr_losses_are_connected_to_public_camera_state(tmp_path):
    _write_tartanground(tmp_path / "tg")
    cfg = _data_config(tmp_path)
    batch = next(iter(build_tartanground_dataloader(cfg, seed=2)))
    model = TinyCorrModel()
    prediction = model(torch.stack([view["img"] for view in batch], dim=1))
    criterion = Pi3Loss(CameraLossConfig(alpha_corr_magnitude=1e-3, alpha_corr_smooth=1e-3))
    loss, metrics = criterion(prediction, batch)
    loss.backward()
    assert model.rot_correction.grad is not None and model.rot_correction.grad.abs() > 0
    for key in ("rot_corr_mag_loss", "rot_corr_smooth_loss"):
        assert key in metrics and torch.isfinite(metrics[key])


def test_fake_end_to_end_train_checkpoint_ema_and_resume(tmp_path):
    _write_dl3dv(tmp_path / "dl3dv")
    _write_tartanground(tmp_path / "tg")
    data = _data_config(tmp_path)
    cfg = TrainConfig(output_dir=str(tmp_path / "out"), epochs=2, steps_per_epoch=2, gradient_accumulation_steps=1, trainable_scope="rot_correction_only", mixed_precision="no", corr_learning_rate=1e-2, checkpoint_every_epochs=1, data=data)
    loaders, mixed = build_train_loaders(data, seed=3)
    model = TinyCorrModel()
    frozen_before = model.frozen_backbone.detach().clone()
    corr_before = model.rot_correction.detach().clone()
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    validation = build_tartanground_dataloader(data, seed=9, training=True)
    history = run_training(
        cfg,
        model,
        loaders,
        mixed,
        validation_loader=validation,
        accelerator=accelerator,
    )
    assert len(history) == 6 and all(item.get("finite", 1) == 1 for item in history)
    assert torch.equal(model.frozen_backbone, frozen_before)
    assert not torch.equal(model.rot_correction, corr_before)
    checkpoint = tmp_path / "out" / "checkpoint-0001"
    assert (checkpoint / "ema.pt").is_file()
    assert (checkpoint / "abot_recon_ema.safetensors").is_file()
    assert load_file(str(checkpoint / "abot_recon.safetensors"))
    assert load_file(str(checkpoint / "abot_recon_ema.safetensors"))
    metrics_lines = (Path(cfg.output_dir) / "metrics.jsonl").read_text().splitlines()
    metrics_records = [json.loads(line) for line in metrics_lines]
    assert len([row for row in metrics_records if row["phase"] == "validation"]) == 2
    assert len([row for row in metrics_records if row["phase"] == "train"]) == 4
    final_weights = load_file(str(tmp_path / "out/checkpoint-0002/abot_recon.safetensors"))

    resumed = TrainConfig(output_dir=str(tmp_path / "resumed"), resume=str(checkpoint), epochs=2, steps_per_epoch=2, trainable_scope="rot_correction_only", mixed_precision="no", corr_learning_rate=1e-2, data=data)
    loaders2, mixed2 = build_train_loaders(data, seed=3)
    result = run_training(resumed, TinyCorrModel(), loaders2, mixed2, accelerator=Accelerator(cpu=True, mixed_precision="no"))
    assert len(result) == 2
    assert (tmp_path / "resumed" / "checkpoint-0002" / "abot_recon.safetensors").is_file()
    metadata = json.loads((tmp_path / "resumed/checkpoint-0002/training_metadata.json").read_text())
    assert metadata["optimizer_step"] == metadata["scheduler_step"] == 4
    torch.testing.assert_close(
        load_file(str(tmp_path / "resumed/checkpoint-0002/abot_recon.safetensors")), final_weights
    )


def test_legacy_scheduler_resume_repairs_lr_and_momentum_without_changing_weights():
    cfg = TrainConfig(epochs=20, steps_per_epoch=800)
    accelerator = Accelerator(cpu=True, mixed_precision="no", step_scheduler_with_optimizer=False)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-6, total_steps=16001, pct_start=0.05,
        div_factor=25, final_div_factor=2,
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    # Record the correct state at step 4800, then reproduce the 2400-step drift.
    expected = None
    for step in range(1, 7201):
        optimizer.step()
        scheduler.step()
        if step == 4800:
            expected = (scheduler.get_last_lr(), optimizer.param_groups[0]["betas"])
    before = parameter.detach().clone()
    _restore_scheduler_progress(
        accelerator, scheduler, {"completed_epoch": 6, "optimizer_step": 4800}, cfg
    )
    assert scheduler.state_dict()["last_epoch"] == 4800
    assert scheduler.state_dict()["_step_count"] == 4801
    assert scheduler.get_last_lr() == expected[0]
    assert optimizer.param_groups[0]["betas"] == expected[1]
    assert torch.equal(parameter, before)
    # Ambiguous legacy metadata must never be silently guessed.
    with pytest.warns(UserWarning, match="cannot be inferred safely"):
        _restore_scheduler_progress(
            accelerator, scheduler, {"completed_epoch": 6, "optimizer_step": 4799}, cfg
        )
    assert scheduler.state_dict()["last_epoch"] == 4800
    # Explicit progress includes intentional skipped updates, not just successes.
    _restore_scheduler_progress(
        accelerator, scheduler,
        {"completed_epoch": 6, "optimizer_step": 4799, "scheduler_step": 4800}, cfg,
    )
    assert scheduler.state_dict()["last_epoch"] == 4800


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("bad", [False, True])
def test_scheduler_counts_update_boundaries_including_intentional_skips(tmp_path, accumulation, bad):
    _write_tartanground(tmp_path / "tg")
    data = _data_config(tmp_path)
    data.dl3dv.weight = 0
    cfg = TrainConfig(
        output_dir=str(tmp_path / "out"), epochs=1, steps_per_epoch=4,
        gradient_accumulation_steps=accumulation,
        trainable_scope="rot_correction_only", mixed_precision="no",
        max_loss=0.5 if bad else 10.0, validation_enabled=False, data=data,
    )
    loaders, mixed = build_train_loaders(data, seed=4)
    # Deliberately inject the default Accelerator: run_training must set the flag.
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=accumulation)
    run_training(
        cfg, TinyCorrModel(), loaders, mixed, accelerator=accelerator,
        criterion=_GuardCriterion("large") if bad else None,
    )
    folder = tmp_path / "out/checkpoint-0001"
    metadata = json.loads((folder / "training_metadata.json").read_text())
    state = torch.load(folder / "trainer_state/scheduler.bin", weights_only=True)
    assert metadata["scheduler_step"] == state["last_epoch"] == 4 // accumulation
    assert state["total_steps"] == 4 // accumulation + 1
    assert metadata["optimizer_step"] == (0 if bad else 4 // accumulation)
