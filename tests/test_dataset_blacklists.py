"""Portable training exclusions, tested with index-only filesystem fixtures."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from abot_recon.training.datasets import dl3dv as dl3dv_module
from abot_recon.training.datasets.blendedmvs_seq import BlendedMVSSeq
from abot_recon.training.datasets.dl3dv import DL3DV
from abot_recon.training.datasets.hypersim_seq import HyperSimSeq
from abot_recon.training.datasets.scannetpp_seq import ScanNetPPSeq


METADATA = Path(dl3dv_module.__file__).parent / "metadata"
HYPER_EXCLUDED = {
    "ai_003_001/cam_00", "ai_004_009/cam_01",
    "ai_031_004/cam_00", "ai_052_002/cam_01",
}
SCAN_EXTRA = {
    "0e350246d3_dslr", "9ef704a38d_dslr", "eaa6c90310_dslr",
    "fe5fe0a8a4_dslr", "46001f434d_iphone", "99010a8938_dslr",
    "c8d099ecd8_dslr",
}
SCAN_LEGACY = "cc0aa81452_iphone"
BLENDED_LEGACY = "000000000000000000000012"


def _lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _dataset(cls, root, **kwargs):
    return cls(root=str(root), num_views=2, resolution=(28, 28),
               train_augmentation=False, **kwargs)


def _small_root(root, cls):
    excluded = HYPER_EXCLUDED if cls is HyperSimSeq else SCAN_EXTRA | {SCAN_LEGACY}
    kept = {"keep/cam_00", "ai_003_001/cam_01"} if cls is HyperSimSeq else {
        "keep_iphone", "0e350246d3_iphone",
    }
    for name in excluded | kept:
        directory = root / name
        directory.mkdir(parents=True)
        if cls is HyperSimSeq:
            for i in range(4):
                (directory / f"{i:06d}_rgb.png").touch()
    if cls is ScanNetPPSeq:
        np.savez(root / "all_metadata.npz",
                 **{name: np.arange(4) for name in excluded | kept})
    return excluded, kept


def _selected(dataset):
    if isinstance(dataset, (HyperSimSeq, DL3DV)):
        return set(dataset.scenes)
    if isinstance(dataset, BlendedMVSSeq):
        return set(dataset.data_dict)
    return {directory.name for directory, _ in dataset.index.sequences}


@pytest.mark.parametrize("filename,count", [
    ("dl3dv_geometry_blacklist_20260810.txt", 542),
    ("blendedmvs_blacklist.txt", 38),
])
def test_packaged_lists_have_exact_unique_counts(filename, count):
    values = _lines(METADATA / filename)
    assert len(values) == len(set(values)) == count
    assert all(not value.startswith(("/", "oss:")) for value in values)


@pytest.mark.parametrize("cls", [HyperSimSeq, ScanNetPPSeq])
@pytest.mark.parametrize("setting", [None, "auto"])
def test_small_defaults_filter_exact_keys_before_indexing(tmp_path, monkeypatch, cls, setting):
    root = tmp_path / "dataset"
    _, kept = _small_root(root, cls)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    dataset = _dataset(cls, root, sequence_blacklist_path=setting)
    assert _selected(dataset) == kept
    expected = HYPER_EXCLUDED if cls is HyperSimSeq else SCAN_EXTRA | {SCAN_LEGACY}
    assert dataset.excluded_sequences == expected


@pytest.mark.parametrize("cls,filename", [
    (HyperSimSeq, "hypersim_geometry_blacklist_20260810.txt"),
    (ScanNetPPSeq, "scannetpp_geometry_blacklist_20260810.txt"),
])
def test_small_root_override_custom_relative_path_and_disable(tmp_path, monkeypatch, cls, filename):
    root = tmp_path / "dataset"
    excluded, kept = _small_root(root, cls)
    target = sorted(kept)[0]
    text = f"# exact keys\n\n{target}\n{target}\n"
    (root / filename).write_text(text, encoding="utf-8")
    (root / "custom.txt").write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    permanent = set() if cls is HyperSimSeq else {SCAN_LEGACY}
    expected = (excluded | kept) - permanent
    for path in (None, "auto", "custom.txt", str(root / "custom.txt")):
        assert _selected(_dataset(cls, root, sequence_blacklist_path=path)) == expected - {target}
    assert _selected(_dataset(cls, root, sequence_blacklist_path="")) == expected
    with pytest.raises(FileNotFoundError):
        _dataset(cls, root, sequence_blacklist_path="missing.txt")


def _dl3dv_root(root):
    bad = _lines(METADATA / "dl3dv_geometry_blacklist_20260810.txt")[0]
    kept = {"custom/keep", "another_bucket/" + bad.split("/", 1)[1]}
    for name in kept | {bad}:
        rgb = root / name / "dense" / "rgb"
        rgb.mkdir(parents=True)
        for i in range(2):
            (rgb / f"{i:05d}.png").touch()
    return bad, kept


def test_dl3dv_packaged_fallback_root_priority_disable_and_exact_matching(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    bad, kept = _dl3dv_root(root)
    monkeypatch.chdir(tmp_path)
    assert set(_dataset(DL3DV, root).scenes) == kept
    disabled = _dataset(DL3DV, root, sequence_blacklist_path="")
    assert set(disabled.scenes) == kept | {bad}
    override = root / "dl3dv_geometry_blacklist_20260810.txt"
    override.write_text("# root takes precedence\ncustom/keep\ncustom/keep\n\n")
    assert set(_dataset(DL3DV, root).scenes) == (kept | {bad}) - {"custom/keep"}
    override.unlink()
    custom = tmp_path / "custom.txt"
    custom.write_text("custom/keep\n")
    assert set(_dataset(DL3DV, root, sequence_blacklist_path=str(custom)).scenes) == (
        kept | {bad}
    ) - {"custom/keep"}
    with pytest.raises(FileNotFoundError):
        _dataset(DL3DV, root, sequence_blacklist_path=str(tmp_path / "missing.txt"))
    monkeypatch.setattr(dl3dv_module, "_DEFAULT_SEQUENCE_BLACKLIST_PATH", str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError, match="installed package"):
        _dataset(DL3DV, root)


def _blended_root(root):
    root.mkdir()
    excluded = set(_lines(METADATA / "blendedmvs_blacklist.txt"))
    kept = {"keep", "keep_other"}
    with h5py.File(root / "new_overlap.h5", "w") as handle:
        for name in excluded | kept:
            handle.create_group(name).create_dataset("basenames", data=[b"00000000"])
    return excluded, kept


@pytest.mark.parametrize("setting", [None, "auto"])
def test_blended_packaged_list_filters_before_payload_reads(tmp_path, monkeypatch, setting):
    root = tmp_path / "dataset"
    excluded, kept = _blended_root(root)
    monkeypatch.chdir(tmp_path)
    dataset = _dataset(BlendedMVSSeq, root, scene_blacklist_path=setting)
    assert set(dataset.data_dict) == kept
    assert dataset.sideways_scene_ids == excluded
    assert not dataset._invalid_orientation  # No implicit orientation audit.


def test_blended_root_json_custom_json_and_txt_disable(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    excluded, kept = _blended_root(root)
    monkeypatch.chdir(tmp_path)
    expected = (excluded | kept) - {BLENDED_LEGACY}
    for name, content in (
        ("blendedmvs_scene_blacklist.json", {"drop_scenes": ["keep"]}),
        ("custom.json", ["keep"]),
    ):
        (root / name).write_text(json.dumps(content))
    (root / "custom.txt").write_text("# exact scene IDs\n\nkeep\nkeep\n")
    for path in (None, "auto", "custom.json", "custom.txt", str(root / "custom.txt")):
        ds = _dataset(BlendedMVSSeq, root, scene_blacklist_path=path)
        assert set(ds.data_dict) == expected - {"keep"}
    assert set(_dataset(BlendedMVSSeq, root, scene_blacklist_path="").data_dict) == expected
    with pytest.raises(FileNotFoundError):
        _dataset(BlendedMVSSeq, root, scene_blacklist_path="missing.txt")


def test_scannet_explicit_exclusions_are_additive(tmp_path):
    _, kept = _small_root(tmp_path, ScanNetPPSeq)
    target = sorted(kept)[0]
    ds = _dataset(ScanNetPPSeq, tmp_path, excluded_sequences=[target])
    assert _selected(ds) == kept - {target}


@pytest.mark.parametrize("cls", [DL3DV, HyperSimSeq, ScanNetPPSeq, BlendedMVSSeq])
def test_relative_dataset_root_with_root_sidecar_and_custom_path(tmp_path, monkeypatch, cls):
    root = tmp_path / "dataset"
    monkeypatch.chdir(tmp_path)
    field = "sequence_blacklist_path"
    if cls is DL3DV:
        bad, kept = _dl3dv_root(root)
        target = "custom/keep"
        expected = (kept | {bad}) - {target}
        filename = "dl3dv_geometry_blacklist_20260810.txt"
    elif cls is BlendedMVSSeq:
        excluded, kept = _blended_root(root)
        target = "keep"
        expected = (excluded | kept) - {BLENDED_LEGACY, target}
        filename = "blendedmvs_scene_blacklist.json"
        field = "scene_blacklist_path"
    else:
        excluded, kept = _small_root(root, cls)
        target = sorted(kept)[0]
        permanent = set() if cls is HyperSimSeq else {SCAN_LEGACY}
        expected = (excluded | kept) - permanent - {target}
        filename = ("hypersim" if cls is HyperSimSeq else "scannetpp") + "_geometry_blacklist_20260810.txt"
    content = json.dumps({"drop_scenes": [target]}) if cls is BlendedMVSSeq else target + "\n"
    (root / filename).write_text(content)
    assert _selected(_dataset(cls, Path("dataset"))) == expected
    (root / "custom.txt").write_text(target + "\n")
    custom = "dataset/custom.txt" if cls is DL3DV else "custom.txt"
    assert _selected(_dataset(cls, Path("dataset"), **{field: custom})) == expected
