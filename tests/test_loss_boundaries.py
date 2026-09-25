import copy

import pytest
import torch
import torch.nn.functional as F

from abot_recon.training.loss import (
    CameraLoss,
    CameraLossConfig,
    Pi3Loss,
    PointLoss,
    _sample_valid,
    align_points_scale,
    prepare_ground_truth,
)


def _batch(batch_size=2, frames=3):
    generator = torch.Generator().manual_seed(123)
    height, width = 4, 5
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    points = torch.stack((x * 0.1, y * 0.1, torch.ones_like(x)), dim=-1).float()
    points = points.expand(batch_size, frames, height, width, 3).clone()
    poses = torch.eye(4).repeat(batch_size, frames, 1, 1)
    poses[:, :, 0, 3] = torch.arange(frames) * 0.1
    views = [
        {"pts3d": points[:, i].clone(), "valid_mask": torch.ones(batch_size, height, width, dtype=torch.bool),
         "camera_pose": poses[:, i].clone(), "dataset": ["tartanground"] * batch_size}
        for i in range(frames)
    ]
    prediction = {
        "local_points": (points + 0.01 * torch.randn(points.shape, generator=generator)).requires_grad_(),
        "camera_poses": (poses + 0.0).requires_grad_(),
        "conf": torch.randn(batch_size, frames, height, width, 1, generator=generator, requires_grad=True),
    }
    return prediction, views


@pytest.mark.parametrize("batch_size", [1, 2])
def test_empty_sample_preserves_pose_scale_and_has_finite_gradients(batch_size):
    prediction, views = _batch(batch_size)
    for view in views:
        view["valid_mask"][0] = False
        view["pts3d"][0] = float("nan")
    target = prepare_ground_truth(views)
    expected_poses = torch.stack([view["camera_pose"] for view in views], dim=1)
    torch.testing.assert_close(target["camera_poses"][0], expected_poses[0])
    local, poses = prediction["local_points"], prediction["camera_poses"]
    loss, details = Pi3Loss()(prediction, views)
    assert torch.isfinite(loss) and loss < 10
    loss.backward()
    for parameter in (local, poses):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(local.grad[0], torch.zeros_like(local.grad[0]))
    torch.testing.assert_close(prediction["camera_poses"][0], poses.detach()[0])
    if batch_size == 1:
        assert details["local_pts_loss"] == 0
        assert details["normal_loss"] == 0


def test_empty_sample_does_not_dilute_valid_point_or_normal_loss():
    prediction, views = _batch()
    for view in views:
        view["valid_mask"][0] = False
    single_prediction = {key: value[1:].detach().clone() for key, value in prediction.items()}
    single_views = [
        {key: value[1:] for key, value in view.items()} for view in views
    ]
    _, mixed_details = Pi3Loss()(prediction, views)
    _, single_details = Pi3Loss()(single_prediction, single_views)
    for key in ("local_pts_loss", "normal_loss"):
        torch.testing.assert_close(mixed_details[key], single_details[key])


@pytest.mark.parametrize("invalid_as_zero", [False, True])
def test_all_empty_confidence_loss_stays_connected(invalid_as_zero):
    prediction, views = _batch(1)
    for view in views:
        view["valid_mask"].zero_()
    local, logits = prediction["local_points"], prediction["conf"]
    criterion = PointLoss(train_confidence=True, confidence_invalid_as_zero=invalid_as_zero)
    loss, details, scale = criterion(prediction, prepare_ground_truth(views))
    torch.testing.assert_close(scale, torch.ones_like(scale))
    assert details["local_pts_loss"] == 0 and details["normal_loss"] == 0
    expected = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)) if invalid_as_zero else logits.sum() * 0
    torch.testing.assert_close(loss, 0.05 * expected)
    loss.backward()
    for parameter in (local, logits):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(local.grad, torch.zeros_like(local.grad))


@pytest.mark.parametrize("train_confidence", [False, True])
@pytest.mark.parametrize("missing_pixels", [False, True])
def test_valid_batch_preserves_original_point_formula_and_gradients(train_confidence, missing_pixels):
    prediction, views = _batch()
    if missing_pixels:
        for view in views:
            view["valid_mask"][:, 0, 0] = False
    original = copy.deepcopy(prediction)
    target = prepare_ground_truth(views)
    points, mask = original["local_points"], target["valid_masks"]
    valid_points = points.clone()
    valid_points[~mask] = 0
    norm = (valid_points.reshape(2, 3, -1, 3).norm(dim=-1).sum((1, 2)) / mask.sum((1, 2, 3)).clamp_min(1)).clamp_min(1e-8)
    normalized = points / norm[:, None, None, None, None]
    weights = target["local_points"][..., 2].clamp_min(1e-6)
    mean_depth = (weights * mask).sum((-2, -1), keepdim=True) / mask.sum((-2, -1), keepdim=True).clamp_min(1)
    weights = 1 / weights.clamp_min(0.1 * mean_depth)
    with torch.no_grad():
        scale = align_points_scale(
            _sample_valid(normalized, mask, 4096),
            _sample_valid(target["local_points"], mask, 4096),
            _sample_valid(weights[..., None], mask, 4096)[..., 0],
        )
    aligned = normalized * scale[:, None, None, None, None]
    residual = (aligned.float() - target["local_points"].float()).abs() * weights[..., None]
    expected = residual[mask].mean() + PointLoss.normal_loss(aligned, target["local_points"], mask)
    if train_confidence:
        labels = (residual.detach().mean(-1) < 0.02).float()
        expected = expected + 0.05 * F.binary_cross_entropy_with_logits(original["conf"][..., 0][mask], labels[mask])
        if missing_pixels:
            invalid_logits = original["conf"][..., 0][~mask]
            expected = expected + 0.05 * F.binary_cross_entropy_with_logits(invalid_logits, torch.zeros_like(invalid_logits))
    local, logits = prediction["local_points"], prediction["conf"]
    loss, _, actual_scale = PointLoss(train_confidence=train_confidence)(prediction, target)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(local.grad, original["local_points"].grad, rtol=1e-6, atol=1e-7)
    if train_confidence:
        torch.testing.assert_close(logits.grad, original["conf"].grad, rtol=0, atol=0)


@pytest.mark.parametrize("pair_mode", ["causal_upper", "dense"])
def test_single_frame_camera_loss_returns_connected_zero(pair_mode):
    poses = torch.eye(4).repeat(2, 1, 1, 1).requires_grad_()
    loss, details = CameraLoss(CameraLossConfig(pair_mode=pair_mode))(
        {"camera_poses": poses}, {"camera_poses": poses.detach()}, torch.ones(2)
    )
    assert loss == 0 and details["camera_pairs"] == 0
    assert details["trans_loss"] == 0 and details["rot_loss"] == 0
    loss.backward()
    assert poses.grad is not None
    torch.testing.assert_close(poses.grad, torch.zeros_like(poses.grad))
