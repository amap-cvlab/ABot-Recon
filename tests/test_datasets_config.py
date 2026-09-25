"""The new sources are opt-in and use the existing public training contract."""
import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from abot_recon.training import data as data_module
from abot_recon.training.config import DATASET_NAMES, DataConfig, TrainConfig, load_config
from abot_recon.training.loss import PointLoss


NEW_DATASETS = DATASET_NAMES[2:]


def test_new_datasets_do_not_change_default_mixture():
    cfg = DataConfig()
    assert (cfg.dl3dv.weight, cfg.tartanground.weight) == (1, 1)
    assert all(getattr(cfg, name).weight == 0 for name in NEW_DATASETS)
    assert set(data_module._CUT3R_DATASETS) == set(NEW_DATASETS)


@pytest.mark.parametrize("name", NEW_DATASETS)
def test_new_dataset_can_be_the_only_configured_source(tmp_path, name):
    cfg = TrainConfig(validation_enabled=False)
    cfg.data.dl3dv.weight = cfg.data.tartanground.weight = 0
    item = getattr(cfg.data, name)
    item.root, item.weight, item.num_workers = str(tmp_path), 1, 0
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    loaded = load_config(path)
    assert type(getattr(loaded.data, name)) is type(item)
    assert getattr(loaded.data, name).weight == 1
    item.root = ""
    OmegaConf.save(OmegaConf.structured(cfg), path)
    with pytest.raises(ValueError, match=f"data.{name}.root"):
        load_config(path)


@pytest.mark.parametrize("mask_bg", [True, False, "rand"])
def test_wildrgbd_mask_config_round_trip(tmp_path, mask_bg):
    cfg = TrainConfig(validation_enabled=False)
    cfg.data.dl3dv.weight = cfg.data.tartanground.weight = 0
    cfg.data.wildrgbd.root = str(tmp_path)
    cfg.data.wildrgbd.weight = 1
    cfg.data.wildrgbd.mask_bg = mask_bg
    path = tmp_path / "wild.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), path)
    assert load_config(path).data.wildrgbd.mask_bg == mask_bg


@pytest.mark.parametrize("name", NEW_DATASETS)
def test_builder_forwards_public_geometry_and_only_enabled_sources(monkeypatch, tmp_path, name):
    cfg = DataConfig(num_frames=32, sequence_consistent_aug_prob=0.2, principal_align_skip_prob=0.7)
    cfg.dl3dv.weight = cfg.tartanground.weight = 0
    item = getattr(cfg, name)
    item.root, item.weight, item.num_workers = str(tmp_path), 1, 0
    received = {}

    class FakeDataset(Dataset):
        def __init__(self, **kwargs):
            received.update(kwargs)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return [{"img": torch.zeros(3, 14, 28), "dataset": name}] * 32

    monkeypatch.setitem(data_module._CUT3R_DATASETS, name, FakeDataset)
    loaders, mixed = data_module.build_train_loaders(cfg, seed=123)
    assert list(loaders) == [name] and mixed.names == [name]
    assert len(next(iter(mixed))) == 32
    assert received["root"] == str(tmp_path)
    assert received["num_views"] == 32
    assert received["resolution"] == (504, 280)
    assert received["principal_align_skip_prob"] == 0.7
    assert received["sequence_consistent_aug_prob"] == 0.2
    assert received["train_augmentation"] is True
    assert received["seed"] == 123 + DATASET_NAMES.index(name)
    assert not {"batch_size", "weight", "num_workers", "pin_memory"} & received.keys()
    if name in {"unreal4k_seq", "hypersim_seq", "blendedmvs_seq"}:
        assert not {"min_interval", "max_interval"} & received.keys()
        assert received["pose_load_workers"] == 1


def test_new_sequence_defaults_match_development_recipe():
    cfg = DataConfig()
    assert cfg.arkit.camera_only and cfg.arkit.allow_repeat
    assert cfg.arkit.max_interval == 1 and cfg.arkit.z_far == 80
    assert cfg.hypersim_seq.allow_repeat and cfg.hypersim_seq.z_far == 80
    assert cfg.blendedmvs_seq.allow_repeat and cfg.blendedmvs_seq.z_far == 0
    assert cfg.scannetpp_seq.max_interval == 1 and cfg.scannetpp_seq.z_far == 0
    assert cfg.hypersim_seq.sequence_blacklist_path is None
    assert cfg.scannetpp_seq.sequence_blacklist_path is None


@pytest.mark.parametrize("name", NEW_DATASETS)
def test_normal_loss_routing_matches_development_cut3r_labels(name):
    target = torch.ones(1, 3, 4, 5, 3)
    predicted = target.clone().requires_grad_()
    poses = torch.eye(4).repeat(1, 3, 1, 1)
    prediction = {"local_points": predicted, "camera_poses": poses}
    ground_truth = {
        "local_points": target,
        "valid_masks": torch.ones(target.shape[:-1], dtype=torch.bool),
        "camera_poses": poses,
        "dataset_names": [name],
    }
    criterion = PointLoss(align_resolution=32)
    calls = []

    def normal(left, right, mask):
        calls.append(name)
        return left.sum() * 0 + 1

    criterion.normal_loss = normal
    total, details, _ = criterion(copy.deepcopy(prediction), ground_truth)
    expected = name in {"tartanair", "pointodyssey", "scannet", "vkitti2"}
    assert bool(calls) == expected
    assert details["normal_loss"].item() == int(expected)
    assert torch.isfinite(total)


def test_example_config_is_loadable():
    cfg = load_config("configs/finetune_cut3r.yaml")
    assert cfg.data.num_frames == 32
    assert not cfg.validation_enabled
    assert cfg.data.tartanair.weight == 1
    assert all(getattr(cfg.data, name).weight == 0 for name in DATASET_NAMES if name != "tartanair")


def test_empty_source_has_actionable_error(tmp_path):
    cfg = DataConfig()
    cfg.dl3dv.weight = cfg.tartanground.weight = 0
    cfg.tartanair.root = str(tmp_path)
    cfg.tartanair.weight = 1
    cfg.tartanair.num_workers = 0
    with pytest.raises(ValueError, match="TartanAir has no eligible sequences"):
        data_module.build_train_loaders(cfg, seed=1)
