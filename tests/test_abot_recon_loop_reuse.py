import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


def test_abot_recon_pose_both_reuses_one_noloop_forward(tmp_path):
    checkpoint = tmp_path / "abot_recon.safetensors"
    salad = tmp_path / "dino_salad.ckpt"
    dino = tmp_path / "dinov2.pth"
    for path in (checkpoint, salad, dino):
        path.write_bytes(b"test")
    runner = ROOT / "scripts/run_long_pose_protocol.sh"
    common = [
        "bash",
        str(runner),
        "--method",
        "abot_recon",
        "--dataset",
        "kitti",
        "--out-dir",
        str(tmp_path / "outputs"),
        "--ckpt",
        str(checkpoint),
        "--dry-run",
    ]
    env = {
        **os.environ,
        "PY": sys.executable,
        "HORIZON_SALAD_CKPT": str(salad),
        "HORIZON_SALAD_DINO_WEIGHTS": str(dino),
    }

    both = subprocess.run(
        [*common, "--abot-recon-loop-mode", "both"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    assert both.count("relpose/eval_dist.py") == 1
    assert both.count("relpose/eval_abot_recon_loop.py") == 1
    assert "abot_recon_loop_enabled=false" in both
    assert "abot_recon_kitti_s1_sim3/abot_recon_kitti_s1_sim3" in both
    assert "abot_recon_kitti_s1_sim3_loop" in both

    no_loop = subprocess.run(
        [*common, "--abot-recon-loop-mode", "off"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout
    assert no_loop.count("relpose/eval_dist.py") == 1
    assert "relpose/eval_abot_recon_loop.py" not in no_loop


def test_abot_recon_loop_evaluator_is_cache_driven():
    source = (ROOT / "relpose/eval_abot_recon_loop.py").read_text()
    assert 'base_pose_path = osp.join(base_seq_dir, "pred_poses.npy")' in source
    assert "extract_abot_recon_loop_descriptors" in source
    assert "_run_stream_inference" not in source


@pytest.mark.parametrize(
    "method",
    [
        "cut3r",
        "ttt3r",
        "longstream",
        "infinitevggt",
        "ovggt",
        "stream3r_window5",
        "horizon",
        "lingbot",
    ],
)
def test_default_abot_loop_mode_does_not_reject_other_methods(tmp_path, method):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_long_pose_protocol.sh"),
            "--method",
            method,
            "--dataset",
            "kitti",
            "--out-dir",
            str(tmp_path / "outputs"),
            "--ckpt",
            str(checkpoint),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PY": sys.executable},
    )
    assert "relpose/eval_dist.py" in result.stdout
