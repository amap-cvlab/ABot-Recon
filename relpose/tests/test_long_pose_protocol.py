from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relpose.long_pose_protocol import (
    bind_dataset_context,
    load_resumable_poses,
    trajectory_length,
    validate_predicted_poses,
)


def _identity_poses(count):
    return torch.eye(4).repeat(count, 1, 1)


def test_validate_predicted_poses_accepts_3x4_and_4x4():
    assert validate_predicted_poses(_identity_poses(3), 3).shape == (3, 4, 4)
    assert validate_predicted_poses(_identity_poses(2)[:, :3], 2).shape == (2, 3, 4)


@pytest.mark.parametrize("bad", ["count", "nan", "rotation"])
def test_validate_predicted_poses_rejects_corruption(bad):
    poses = _identity_poses(3)
    expected = 3
    if bad == "count":
        expected = 4
    elif bad == "nan":
        poses[1, 0, 0] = float("nan")
    else:
        poses[1, :3, :3] *= 2.0
    with pytest.raises(ValueError):
        validate_predicted_poses(poses, expected)


def test_resume_only_reuses_complete_valid_pose_cache(tmp_path):
    np.save(tmp_path / "pred_poses.npy", _identity_poses(5).numpy())
    loaded = load_resumable_poses(tmp_path, 5, enabled=True)
    assert loaded.shape == (5, 4, 4)
    assert load_resumable_poses(tmp_path, 5, enabled=False) is None
    with pytest.raises(ValueError, match="Pose/frame count mismatch"):
        load_resumable_poses(tmp_path, 4, enabled=True)


def test_bind_dataset_context_is_explicit():
    model = SimpleNamespace()
    info = {"pose_eval_stride": 1}
    bind_dataset_context(model, "vbr-long", info)
    assert model.eval_dataset_name == "vbr-long"
    assert model.eval_dataset_info is info


def test_trajectory_length_supports_loader_tuple_and_evo_object():
    poses = np.zeros((7, 7))
    timestamps = np.arange(7)
    assert trajectory_length((poses, timestamps)) == 7
    assert trajectory_length(SimpleNamespace(positions_xyz=np.zeros((5, 3)))) == 5
