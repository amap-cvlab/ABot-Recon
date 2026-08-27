from pathlib import Path

from omegaconf import OmegaConf

from scripts.formal_run_metadata import checkpoint_sha256, sha256_file


def test_formal_metadata_hashes_released_checkpoint(tmp_path):
    checkpoint = tmp_path / "abot_recon.safetensors"
    checkpoint.write_bytes(b"released")
    assert checkpoint_sha256(checkpoint) == sha256_file(checkpoint)
    assert checkpoint_sha256(tmp_path) == (
        f"abot_recon.safetensors sha256={sha256_file(checkpoint)}"
    )


def test_eval_config_uses_release_runtime_defaults():
    config_path = Path(__file__).parents[2] / "configs" / "model" / "default.yaml"
    config = OmegaConf.load(config_path).abot_recon.cfg
    assert config._target_ == "models.abot_recon.ABotReconEval"
    assert config.ckpt == "checkpoints/abot_recon.safetensors"
    assert config.attention_backend == "auto"
    assert "source_root" not in config
