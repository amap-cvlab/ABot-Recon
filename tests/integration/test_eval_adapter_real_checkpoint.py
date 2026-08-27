import gc
import os
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from interfaces.abot_recon import (
    infer_cameras_c2w,
    infer_custom_reconstruction,
    infer_mv_pointclouds,
)
from models.abot_recon import ABotReconEval


CHECKPOINT = os.environ.get("ABOT_RECON_CHECKPOINT")
IMAGE_DIR = os.environ.get("ABOT_RECON_IMAGE_DIR")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@pytest.mark.skipif(
    not CHECKPOINT or not IMAGE_DIR,
    reason="set checkpoint and image directory for eval-adapter integration",
)
@pytest.mark.parametrize("backend", ["paged", "sdpa"])
def test_eval_adapter_runs_pose_and_reconstruction(backend, tmp_path):
    paths = sorted(
        str(path) for path in Path(IMAGE_DIR).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )[:3]
    assert len(paths) == 3
    device = os.environ.get("ABOT_RECON_DEVICE", "cuda")
    model = ABotReconEval(
        CHECKPOINT,
        device=device,
        attention_backend=backend,
        max_frames=128,
        num_frames_cap=128,
    ).eval()
    config = DictConfig(
        {
            "device": device,
            "output_dir": str(tmp_path),
            "measure_forward_fps": False,
            "pointmap_resize_mode": "nearest",
            "pi3_pc_world_source": "points",
            "pc_alignment_depth_max": None,
        }
    )

    poses, intrinsics = infer_cameras_c2w(paths, model, config)
    assert poses.shape == (3, 4, 4)
    assert intrinsics is None
    assert model.attention_backend == backend

    recon_poses, local_points, confidence, _ = infer_custom_reconstruction(paths, model, config)
    assert recon_poses.shape == (3, 4, 4)
    assert local_points.shape == (3, 280, 504, 3)
    assert confidence.shape == (3, 280, 504)
    assert all(torch.isfinite(value).all() for value in (recon_poses, local_points, confidence))

    config.mv_recon_output_indices = [0, 2]
    world, all_poses, pred_mask, observation_mask, alignment_mask, extras = infer_mv_pointclouds(
        paths, model, config, data_size=(280, 504)
    )
    assert world.shape == (2, 280, 504, 3)
    assert all_poses.shape == (3, 4, 4)
    assert pred_mask.shape == (2, 280, 504)
    assert observation_mask.shape == (2, 280, 504)
    assert alignment_mask is None
    assert extras["pred_depth"].shape == (2, 280, 504)
    assert extras["pred_local_points"].shape == (2, 280, 504, 3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
