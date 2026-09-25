
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from abot_recon.checkpoint import (
    RELEASE_CHECKPOINT,
    load_model_checkpoint,
    resolve_checkpoint,
    resolve_pretrained_checkpoint,
)


def test_directory_resolves_only_the_released_checkpoint(tmp_path):
    save_file({"weight": torch.ones(1)}, str(tmp_path / "abot_recon.safetensors"))
    assert resolve_checkpoint(tmp_path).name == "abot_recon.safetensors"


def test_directory_rejects_training_checkpoint_layout(tmp_path):
    torch.save({"weight": torch.ones(1)}, tmp_path / "pytorch_model.bin")
    with pytest.raises(FileNotFoundError, match="abot_recon.safetensors"):
        resolve_checkpoint(tmp_path)


def test_direct_complete_bin_checkpoint_remains_supported(tmp_path):
    model = nn.Linear(1, 1)
    torch.save(model.state_dict(), tmp_path / "complete.bin")
    loaded = nn.Linear(1, 1)
    load_model_checkpoint(loaded, tmp_path / "complete.bin")
    torch.testing.assert_close(loaded.weight, model.weight)
    torch.testing.assert_close(loaded.bias, model.bias)


def test_strict_model_loading_rejects_architecture_drift(tmp_path):
    model = nn.Linear(1, 1)
    torch.save({"weight": torch.ones(1, 1)}, tmp_path / "model.bin")
    with pytest.raises(RuntimeError):
        load_model_checkpoint(model, tmp_path / "model.bin")


def test_unsupported_direct_checkpoint_suffix_is_rejected(tmp_path):
    path = tmp_path / "model.ckpt"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported checkpoint suffix"):
        resolve_checkpoint(path)


def test_pretrained_resolver_keeps_local_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / RELEASE_CHECKPOINT
    save_file({"weight": torch.ones(1)}, str(checkpoint))

    def should_not_download(**kwargs):
        raise AssertionError("local checkpoints must not use the Hugging Face Hub")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", should_not_download)
    assert resolve_pretrained_checkpoint(checkpoint) == checkpoint


def test_pretrained_resolver_downloads_hugging_face_repo(tmp_path, monkeypatch):
    checkpoint = tmp_path / RELEASE_CHECKPOINT
    save_file({"weight": torch.ones(1)}, str(checkpoint))
    received = {}

    def fake_download(**kwargs):
        received.update(kwargs)
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    resolved = resolve_pretrained_checkpoint(
        "acvlab/ABot-Recon",
        cache_dir=tmp_path / "cache",
        revision="release",
        token="test-token",
        local_files_only=True,
    )
    assert resolved == checkpoint
    assert received == {
        "repo_id": "acvlab/ABot-Recon",
        "filename": RELEASE_CHECKPOINT,
        "cache_dir": str(tmp_path / "cache"),
        "revision": "release",
        "token": "test-token",
        "local_files_only": True,
    }


def test_missing_local_checkpoint_is_not_treated_as_repo():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_pretrained_checkpoint("checkpoints/missing.safetensors")
