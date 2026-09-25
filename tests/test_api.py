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

    def infer_paths(
        self,
        paths,
        *,
        output_local_points,
        output_world_points,
        output_confidence,
        dense_output_indices=None,
    ):
        poses = torch.eye(4).repeat(len(paths), 1, 1)
        result = {"camera_poses": poses}
        dense_count = len(paths) if dense_output_indices is None else len(dense_output_indices)
        if output_local_points or output_world_points:
            result["local_points"] = torch.zeros(dense_count, 1, 1, 3)
        if output_world_points:
            result["world_points"] = torch.zeros(dense_count, 1, 1, 3)
        if output_confidence:
            result["confidence"] = torch.linspace(0.0, 1.0, dense_count).reshape(
                dense_count, 1, 1
            )
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
    result = model.infer(
        paths,
        output_local_points=True,
        output_world_points=True,
        output_confidence=True,
    )
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
    result = model.infer(paths)
    assert torch.equal(result.camera_poses_noloop, torch.eye(4).repeat(2, 1, 1))
    assert torch.equal(result.camera_poses, result.camera_poses_loop)
    assert torch.equal(result.relative_poses, result.relative_poses_loop)
    assert result.metadata["loop_closure"] is True
    assert result.metadata["pose_outputs"] == ["noloop", "loop"]
    assert result.local_points.shape == (2, 1, 1, 3)
    assert result.world_points is None
    assert result.confidence.shape == (2, 1, 1)
    assert result.confidence_mask.all()
    assert result.metadata["dense_outputs"]["local_points"] is True


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
    result = model.infer(
        [path],
        output_local_points=False,
        output_world_points=False,
        output_confidence=True,
    )
    assert result.local_points is None
    assert result.world_points is None
    assert result.confidence.shape == (1, 1, 1)


def test_sparse_dense_outputs_keep_all_poses(tmp_path):
    paths = _images(tmp_path, 4)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(paths, dense_output_indices=[0, 3])
    assert result.camera_poses.shape[0] == 4
    assert result.local_points.shape[0] == 2
    assert result.metadata["dense_output_indices"] == [0, 3]


def test_world_points_can_be_requested_without_returning_local_points(tmp_path):
    paths = _images(tmp_path, 2)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(paths, output_local_points=False, output_world_points=True)
    assert result.local_points is None
    assert result.world_points.shape == (2, 1, 1, 3)


def test_legacy_output_points_alias_enables_local_and_world_points(tmp_path):
    paths = _images(tmp_path, 2)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(paths, output_points=True)
    assert result.local_points is not None
    assert result.world_points is not None


def test_confidence_threshold_masks_local_and_world_points(tmp_path):
    paths = _images(tmp_path, 2)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(
        paths,
        output_world_points=True,
        confidence_threshold=0.5,
    )
    assert result.confidence[:, 0, 0].tolist() == [0.0, 1.0]
    assert result.confidence_mask[:, 0, 0].tolist() == [False, True]
    assert torch.isnan(result.local_points[0]).all()
    assert torch.isnan(result.world_points[0]).all()
    assert torch.isfinite(result.local_points[1]).all()
    assert torch.isfinite(result.world_points[1]).all()


def test_filtering_can_run_without_returning_confidence(tmp_path):
    paths = _images(tmp_path, 2)
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    result = model.infer(paths, output_confidence=False, confidence_threshold=0.5)
    assert result.confidence is None
    assert result.confidence_mask is not None
    assert torch.isnan(result.local_points[0]).all()


def test_public_api_rejects_missing_images(tmp_path):
    model = ABotRecon(FakeModel(), InferenceConfig(loop_closure=False))
    with pytest.raises(FileNotFoundError):
        model.infer([Path(tmp_path / "missing.jpg")])
