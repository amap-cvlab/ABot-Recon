"""Disk-backed CUT3R-format fixtures; no NAS, credentials or private indices.

TartanAir and MVS-Synth have identical W/X development implementations;
Spring differs only in constructor line wrapping. These tests lock their
shared consumed-data behavior, not removed mask fields or their RNG draws.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from abot_recon.training.data import collate_views
from abot_recon.training.datasets.mvs_synth import MVSSynth
from abot_recon.training.datasets.sequence import SequenceIndex
from abot_recon.training.datasets.spring import Spring
from abot_recon.training.datasets.tartanair import TartanAir


WIDTH, HEIGHT = 32, 24
K = np.array([[16, 0, 16], [0, 16, 12], [0, 0, 1]], dtype=np.float32)
CASES = [TartanAir, Spring, MVSSynth]


def _pose(index):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [index * 0.25, 0.5, -0.25]
    return pose


def _write_sequence(root, cls, count, *, scene="forest", depth=None):
    root = Path(root)
    sequence = root / scene
    if cls is TartanAir:
        sequence = sequence / "Easy" / "P000"
        sequence.mkdir(parents=True, exist_ok=True)
    else:
        for part in ("rgb", "depth", "cam"):
            (sequence / part).mkdir(parents=True, exist_ok=True)
    if depth is None:
        depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    for frame in range(count):
        name = f"{frame:06d}"
        pixels = np.full((HEIGHT, WIDTH, 3), 40 + frame, dtype=np.uint8)
        if cls is TartanAir:
            image_path = sequence / f"{name}_rgb.png"
            depth_path = sequence / f"{name}_depth.npy"
            camera_path = sequence / f"{name}_cam.npz"
            camera = dict(camera_intrinsics=K, camera_pose=_pose(frame))
        else:
            suffix = ".jpg" if cls is MVSSynth else ".png"
            image_path = sequence / "rgb" / (name + suffix)
            depth_path = sequence / "depth" / f"{name}.npy"
            camera_path = sequence / "cam" / f"{name}.npz"
            camera = dict(intrinsics=K, pose=_pose(frame))
        # CUT3R MVS-Synth writes PNG content with a .jpg suffix.
        Image.fromarray(pixels).save(image_path, format="PNG")
        np.save(depth_path, depth)
        np.savez(camera_path, **camera)
    return sequence


def _dataset(cls, root, **kwargs):
    options = dict(
        num_views=32, resolution=(WIDTH, HEIGHT), seed=19, split="train",
        train_augmentation=False, aug_crop=0, aug_focal=1.0,
        preserve_fov_prob=0.0, sequence_consistent_aug_prob=1.0,
        max_refetch=1,
    )
    options.update(kwargs)
    return cls(root=str(root), **options)


@pytest.mark.parametrize("cls", CASES)
def test_official_files_to_32_views_and_real_collate(tmp_path, cls):
    sequence = _write_sequence(tmp_path, cls, 11)
    before = set(tmp_path.rglob("*"))
    dataset = _dataset(cls, tmp_path)
    assert dataset.allow_repeat is True
    assert len(dataset) == 2  # W/X repeat cutoff is max(32 // 3, 3) = 10.
    assert dataset.get_image_num() == 11
    dataset._crop_resize_if_necessary = Mock(wraps=dataset._crop_resize_if_necessary)
    samples = [dataset[0], dataset[-1]]
    batch = collate_views(samples)
    assert len(batch) == 32
    for views in samples:
        assert len(views) == 32
        for view in views:
            frame = int(view["label"].rsplit("/", 1)[1])
            assert set(view) == {
                "img", "pts3d", "valid_mask", "camera_pose", "camera_intrinsics",
                "dataset", "label", "camera_only",
            }
            assert view["dataset"] == dataset.dataset_name
            assert view["camera_only"] is False
            assert view["img"].shape == (3, HEIGHT, WIDTH)
            assert view["img"].dtype == torch.float32
            torch.testing.assert_close(view["img"], torch.full_like(view["img"], (40 + frame) / 255))
            assert view["valid_mask"].dtype == torch.bool
            assert view["valid_mask"].all()
            torch.testing.assert_close(view["camera_intrinsics"], torch.from_numpy(K))
            torch.testing.assert_close(view["camera_pose"], torch.from_numpy(_pose(frame)))
            # Metric depth and c2w pose are used directly, not inverted or /1000.
            local_point = np.array([0.5, 0.0, 2.0], dtype=np.float32)
            expected = _pose(frame)[:3, :3] @ local_point + _pose(frame)[:3, 3]
            torch.testing.assert_close(view["pts3d"][12, 20], torch.from_numpy(expected))
    for frame in batch:
        assert frame["img"].shape == (2, 3, HEIGHT, WIDTH)
        assert frame["pts3d"].shape == (2, HEIGHT, WIDTH, 3)
        assert frame["camera_intrinsics"].shape == (2, 3, 3)
        assert frame["camera_pose"].shape == (2, 4, 4)
        assert frame["camera_only"] == [False, False]
    calls = dataset._crop_resize_if_necessary.call_args_list
    assert len(calls) == 64
    for offset in (0, 32):
        sequence_aug = calls[offset].kwargs["sequence_aug"]
        assert sequence_aug is not None
        assert all(call.kwargs["sequence_aug"] is sequence_aug for call in calls[offset:offset + 32])
        assert all(call.kwargs["preserve_fov"] is False for call in calls[offset:offset + 32])
    assert set(tmp_path.rglob("*")) == before  # No generated indices/caches.
    if cls is MVSSynth:
        with Image.open(sequence / "rgb" / "000000.jpg") as image:
            assert image.format == "PNG"


@pytest.mark.parametrize("cls", CASES)
def test_development_depth_filters_and_sequence_kwargs(tmp_path, cls):
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    depth[0, :7] = [np.nan, np.inf, -np.inf, 0, -5, 1000, 100]
    _write_sequence(tmp_path, cls, 10, depth=depth)
    dataset = _dataset(cls, tmp_path)
    sequence_aug = {"crop_scale": 1.0, "crop_delta": 0, "skip_principal_align": True}
    dataset._crop_resize_if_necessary = Mock(wraps=dataset._crop_resize_if_necessary)
    views = dataset._get_views(0, (WIDTH, HEIGHT), np.random.default_rng(3), 32, True, sequence_aug)
    expected = {
        TartanAir: [0, -1, 0, 0, -5, -1, 0],
        Spring: [0, 0, 0, 0, -5, 1000, 100],
        MVSSynth: [0, 0, 0, 0, -5, 0, 0],
    }[cls]
    for view in views:
        np.testing.assert_array_equal(view["depthmap"][0, :7], expected)
        assert view["depthmap"][12, 16] == 2
        assert view["depthmap"].dtype == np.float32
    for call in dataset._crop_resize_if_necessary.call_args_list:
        assert call.kwargs["preserve_fov"] is True
        assert call.kwargs["sequence_aug"] is sequence_aug
    # Exercise the base's geometry/validity conversion as well as raw depth.
    sample = dataset[0]
    for view in sample:
        np.testing.assert_array_equal(view["valid_mask"][0, :7].numpy(), np.array(expected) > 0)
        assert torch.isfinite(view["pts3d"]).all()


@pytest.mark.parametrize("cls", CASES)
def test_minimum_frame_counts_and_single_frame(tmp_path, cls):
    _write_sequence(tmp_path, cls, 9)
    assert len(_dataset(cls, tmp_path)) == 0
    _write_sequence(tmp_path, cls, 31)
    assert len(_dataset(cls, tmp_path, allow_repeat=False)) == 0
    _write_sequence(tmp_path, cls, 32)
    dataset = _dataset(cls, tmp_path, allow_repeat=False)
    assert len(dataset) == 1
    assert len(dataset[0]) == 32
    single = _dataset(cls, tmp_path, num_views=1, allow_repeat=False)
    assert len(single) == 32
    assert len(single[0]) == 1
    assert single[-1][0]["camera_pose"][0, 3] == 31 * 0.25


@pytest.mark.parametrize("cls,maximum", [(TartanAir, 20), (Spring, 4), (MVSSynth, 4)])
def test_interval_defaults_and_validation(tmp_path, cls, maximum):
    _write_sequence(tmp_path, cls, 10)
    dataset = _dataset(cls, tmp_path)
    assert (dataset.min_interval, dataset.max_interval) == (1, maximum)
    for minimum, upper in [(0, 4), (3, 2)]:
        with pytest.raises(ValueError, match="min_interval"):
            _dataset(cls, tmp_path, min_interval=minimum, max_interval=upper)


@pytest.mark.parametrize("cls", [Spring, MVSSynth])
def test_foldback_preserves_sampling_order(tmp_path, cls):
    _write_sequence(tmp_path, cls, 10)
    dataset = _dataset(cls, tmp_path, min_interval=3, max_interval=3)
    frames = [int(view["label"].rsplit("/", 1)[1]) for view in dataset[0]]
    # Fixed stride goes forward, then folds back instead of sorting frames.
    assert frames[:7] == [0, 3, 6, 9, 6, 3, 0]


def test_tartanair_builtin_ocean_filter_without_external_files(tmp_path):
    _write_sequence(tmp_path, TartanAir, 10, scene="forest")
    _write_sequence(tmp_path, TartanAir, 10, scene="OPEN_oCeAn_scene")
    default = _dataset(TartanAir, tmp_path)
    assert len(default) == 1
    assert default[0][0]["label"].startswith("forest/")
    unfiltered = _dataset(TartanAir, tmp_path, exclude_scene_substrings=())
    assert len(unfiltered) == 2
    # The caller controls the root; the split parameter does not invent a holdout.
    assert len(_dataset(TartanAir, tmp_path, split=None)) == len(default)


def test_sequence_index_boundaries():
    short, first, second = list("123456789"), list(range(11)), list(range(12))
    index = SequenceIndex([("short", short), (Path("first"), first), ("second", second)], 32, True)
    assert len(index) == 5
    assert index.image_count == 23
    assert len(index.sequences) == 2
    assert index.resolve(0) == (Path("first"), first, 0)
    assert index.resolve(1) == (Path("first"), first, 1)
    assert index.resolve(2) == ("second", second, 0)
    assert index.resolve(-1) == ("second", second, 2)
    assert index.resolve(-5) == index.resolve(0)
    for invalid in (5, -6):
        with pytest.raises(IndexError):
            index.resolve(invalid)
    empty = SequenceIndex([], 32, True)
    assert len(empty) == empty.image_count == 0
    with pytest.raises(IndexError):
        empty.resolve(0)
    with pytest.raises(ValueError, match="num_views"):
        SequenceIndex([], 0, False)
    assert len(SequenceIndex([("one", ["frame"])], 1, False)) == 1
    assert len(SequenceIndex([("one", ["frame"])], 1, True)) == 0
