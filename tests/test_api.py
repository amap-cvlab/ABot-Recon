from pathlib import Path

import pytest
import torch

from abot_recon import ABotRecon, InferenceConfig


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def infer_paths(self, paths, *, output_points, output_confidence, dense_output_indices=None):
        poses = torch.eye(4).repeat(len(paths), 1, 1)
        result = {"camera_poses": poses}
        dense_count = len(paths) if dense_output_indices is None else len(dense_output_indices)
        if output_points:
            result["local_points"] = torch.zeros(dense_count, 1, 1, 3)
        if output_confidence:
            result["confidence"] = torch.ones(dense_count, 1, 1)
        return result


def _images(tmp_path, count):
    paths = []
    for index in range(count):
        path = tmp_path / f"{index}.jpg"
        path.touch()
        paths.append(path)
    return paths


def test_public_api_builds_consistent_outputs(tmp_path):
    paths = _images(tmp_path, 2)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(paths, output_points=True, output_confidence=True)
    assert result.camera_poses.shape == (2, 4, 4)
    assert result.relative_poses.shape == (1, 4, 4)
    assert torch.equal(result.camera_poses, result.camera_poses_noloop)
    assert result.camera_poses_loop is None
    assert result.world_points.shape == (2, 1, 1, 3)
    assert result.confidence.shape == (2, 1, 1)


def test_default_loop_returns_before_and_after_poses(monkeypatch, tmp_path):
    paths = _images(tmp_path, 2)

    def fake_refine(image_paths, poses, model, config):
        assert image_paths == paths
        refined = poses.clone()
        refined[:, 0, 3] = torch.arange(len(poses), dtype=poses.dtype)
        return refined

    monkeypatch.setattr("abot_recon.loop_closure.refine_trajectory", fake_refine)
    model = ABotRecon(FakeModel(), InferenceConfig())
    result = model.infer(paths, output_points=True)
    assert torch.equal(result.camera_poses_noloop, torch.eye(4).repeat(2, 1, 1))
    assert torch.equal(result.camera_poses, result.camera_poses_loop)
    assert torch.equal(result.relative_poses, result.relative_poses_loop)
    assert result.metadata["loop_closure"] is True
    assert result.metadata["pose_outputs"] == ["noloop", "loop"]
    assert result.local_points.shape == (2, 1, 1, 3)
    assert result.world_points[1, 0, 0, 0].item() == 1.0


def test_explicit_no_loop_overrides_default(monkeypatch, tmp_path):
    path = _images(tmp_path, 1)[0]

    def should_not_run(*args, **kwargs):
        raise AssertionError("loop closure should be disabled")

    monkeypatch.setattr("abot_recon.loop_closure.refine_trajectory", should_not_run)
    result = ABotRecon(FakeModel(), InferenceConfig()).infer([path], loop_closure=False)
    assert result.camera_poses_loop is None
    assert result.metadata["pose_outputs"] == ["noloop"]


def test_confidence_can_be_requested_without_saving_points(tmp_path):
    path = _images(tmp_path, 1)[0]
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer([path], output_points=False, output_confidence=True)
    assert result.local_points is None
    assert result.world_points is None
    assert result.confidence.shape == (1, 1, 1)


def test_sparse_dense_outputs_keep_all_poses(tmp_path):
    paths = _images(tmp_path, 4)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(paths, output_points=True, dense_output_indices=[0, 3])
    assert result.camera_poses.shape[0] == 4
    assert result.local_points.shape[0] == 2
    assert result.metadata["dense_output_indices"] == [0, 3]


def test_public_api_rejects_missing_images(tmp_path):
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    with pytest.raises(FileNotFoundError):
        model.infer([Path(tmp_path / "missing.jpg")])
