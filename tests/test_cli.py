from types import SimpleNamespace

import numpy as np
import torch

from abot_recon.cli import _save_pose_outputs, build_parser
from abot_recon.checkpoint import DEFAULT_MODEL_ID


def test_cli_enables_loop_by_default_and_can_disable_it():
    parser = build_parser()
    assert parser.parse_args([]).loop_closure is True
    assert parser.parse_args(["--no-loop-closure"]).loop_closure is False
    assert parser.parse_args([]).checkpoint == DEFAULT_MODEL_ID


def test_cli_saves_noloop_and_loop_pose_outputs(tmp_path):
    poses = torch.eye(4).repeat(2, 1, 1)
    result = SimpleNamespace(
        camera_poses=poses + 1,
        relative_poses=poses[:1] + 1,
        camera_poses_noloop=poses,
        relative_poses_noloop=poses[:1],
        camera_poses_loop=poses + 1,
        relative_poses_loop=poses[:1] + 1,
    )
    _save_pose_outputs(tmp_path, result)
    expected = {
        "camera_poses.npy",
        "relative_poses.npy",
        "camera_poses_noloop.npy",
        "relative_poses_noloop.npy",
        "camera_poses_loop.npy",
        "relative_poses_loop.npy",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    np.testing.assert_array_equal(np.load(tmp_path / "camera_poses_noloop.npy"), poses)
