import os
from pathlib import Path

import pytest
import torch

from abot_recon.checkpoint import load_model_checkpoint
from abot_recon.model import build_released_network
from abot_recon.training.loss import Pi3Loss
from abot_recon.training.trainer import configure_trainable_scope


CHECKPOINT = os.environ.get("ABOT_RECON_CHECKPOINT")


@pytest.mark.skipif(not CHECKPOINT, reason="set ABOT_RECON_CHECKPOINT for GPU integration")
@pytest.mark.parametrize("enable_confidence", [False, True])
def test_real_checkpoint_default_all_forward_loss_and_backward(enable_confidence):
    device = torch.device(os.environ.get("ABOT_RECON_DEVICE", "cuda"))
    frames = int(os.environ.get("ABOT_RECON_TRAIN_FRAMES", "2"))
    model = build_released_network(
        local_window_frames=12,
        max_frames=max(128, frames),
        enable_confidence=True,
        use_paged_kv=False,
        infer_mode="full",
        use_chunk_flex_attention=True,
        use_chunk_flex_compile=True,
    )
    load_model_checkpoint(model, Path(CHECKPOINT))
    model.train_conf = enable_confidence
    configure_trainable_scope(model, "all", train_confidence=enable_confidence)
    model.to(device).train()

    batch, height, width = 1, 280, 504
    images = torch.rand(batch, frames, 3, height, width, device=device)
    points = torch.zeros(batch, height, width, 3, device=device)
    points[..., 2] = 1
    pose = torch.eye(4, device=device).expand(batch, 4, 4).clone()
    views = [
        {
            "img": images[:, frame],
            "pts3d": points,
            "valid_mask": torch.ones(batch, height, width, dtype=torch.bool, device=device),
            "camera_pose": pose,
            "dataset": ["dl3dv"],
        }
        for frame in range(frames)
    ]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        prediction = model(
            images,
            causal_global_attn=True,
            long_sequence_parallel=True,
            streaming_inference=False,
        )
        loss, metrics = Pi3Loss(
            train_confidence=enable_confidence,
            confidence_invalid_as_zero=True,
        )(prediction, views)
    assert torch.isfinite(loss)
    loss.backward()
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert not any(name.startswith("encoder.") for name in trainable)
    confidence_names = [
        name
        for name in trainable
        if name.startswith(("conf_decoder.", "conf_head."))
    ]
    assert bool(confidence_names) is enable_confidence
    missing = [name for name, parameter in trainable.items() if parameter.grad is None]
    assert not missing, f"trainable parameters without gradients: {missing[:20]}"
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable.values())
    assert "rot_corr_mag_loss" in metrics
    assert ("confidence_loss" in metrics) is enable_confidence
