from pathlib import Path

import pytest

from abot_recon.config import InferenceConfig


def test_defaults_are_relative_and_release_safe():
    config = InferenceConfig()
    assert config.checkpoint == Path("checkpoints/abot_recon.safetensors")
    assert not config.checkpoint.is_absolute()
    assert config.local_window_frames == 12
    assert config.output_local_points is True
    assert config.output_world_points is False
    assert config.output_confidence is True
    assert config.confidence_threshold == 0.0
    assert config.loop_closure is True
    assert config.attention_backend == "auto"
    assert not config.loop_salad_checkpoint.is_absolute()
    assert not config.loop_dino_checkpoint.is_absolute()


def test_every_runtime_setting_can_be_overridden():
    config = InferenceConfig().override(
        checkpoint=Path("/tmp/model"), device="cpu", amp_dtype="fp32", max_frames=17
    )
    assert config.checkpoint == Path("/tmp/model")
    assert config.device == "cpu"
    assert config.max_frames == 17


def test_dense_outputs_are_independently_configurable():
    config = InferenceConfig(
        output_local_points=False, output_world_points=True, output_confidence=True
    )
    assert config.output_local_points is False
    assert config.output_world_points is True
    assert config.output_confidence is True


def test_attention_backend_is_validated():
    with pytest.raises(ValueError, match="attention_backend"):
        InferenceConfig(attention_backend="legacy")


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_confidence_threshold_is_validated(threshold):
    with pytest.raises(ValueError, match="confidence_threshold"):
        InferenceConfig(confidence_threshold=threshold)
