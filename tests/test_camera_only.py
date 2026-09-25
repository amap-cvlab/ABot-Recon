"""Camera-only regressions using the real data/loss interfaces.

The fixtures contain no dataset files and do not require CUDA or checkpoints.
Camera-only is a sequence-level flag read from the first collated view.
"""

from __future__ import annotations

import copy

import numpy as np
import PIL.Image
import pytest
import torch

from abot_recon.training.data import collate_views
from abot_recon.training.datasets.base import MultiViewDataset
from abot_recon.training import loss as loss_module
from abot_recon.training.loss import (
    CameraLoss,
    CameraLossConfig,
    Pi3Loss,
    PointLoss,
    prepare_ground_truth,
)


_MISSING = object()


def _poses(x, y=None, *, batch=1):
    x = torch.as_tensor(x, dtype=torch.float32)
    poses = torch.eye(4).repeat(batch, x.numel(), 1, 1)
    poses[..., 0, 3] = x
    if y is not None:
        poses[..., 1, 3] = torch.as_tensor(y, dtype=torch.float32)
    return poses


def _views(batch=2, frames=4, flag=_MISSING):
    height, width = 4, 5
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    local = torch.stack((x * 0.1, y * 0.1, torch.ones_like(x) * 2), dim=-1).float()
    poses = _poses(torch.linspace(0, 1, frames), batch=batch)
    views = []
    for frame in range(frames):
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        valid[:, -1, -1] = False  # Exercise invalid-pixel confidence too.
        view = {
            "pts3d": local.expand(batch, -1, -1, -1).clone() + poses[:, frame, None, None, :3, 3],
            "valid_mask": valid,
            "camera_pose": poses[:, frame].clone(),
            "dataset": ["tartanground"] * batch,
        }
        if flag is not _MISSING:
            view["camera_only"] = copy.deepcopy(flag)
        views.append(view)
    return views


def _prediction(views):
    batch, height, width = views[0]["valid_mask"].shape
    frames = len(views)
    generator = torch.Generator().manual_seed(819)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    plane = torch.stack((x * 0.1, y * 0.1, torch.ones_like(x) * 2), dim=-1).float()
    local = plane.expand(batch, frames, -1, -1, -1).clone()
    local += 0.035 * torch.randn(local.shape, generator=generator)
    poses = _poses(
        torch.linspace(0, 1.3, frames),
        torch.sin(torch.linspace(0, 2.2, frames)) * 0.2,
        batch=batch,
    )
    return {
        "local_points": local.requires_grad_(),
        "camera_poses": poses.requires_grad_(),
        "conf": torch.randn(batch, frames, height, width, 1, generator=generator).requires_grad_(),
    }


def _clone_prediction(prediction):
    return {key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in prediction.items()}


def _subset_views(views, sample):
    output = []
    for view in views:
        result = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                result[key] = value[sample:sample + 1].clone()
            elif isinstance(value, list):
                result[key] = value[sample:sample + 1]
            else:
                result[key] = value
        output.append(result)
    return output


def _assert_no_gradient(parameter, sample=None):
    if parameter.grad is None:
        return
    gradient = parameter.grad if sample is None else parameter.grad[sample]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


class _SyntheticDataset(MultiViewDataset):
    dataset_name = "synthetic_camera_only"

    def __init__(self, root, flag=_MISSING):
        self.flag = flag
        super().__init__(
            root=str(root), num_views=3, resolution=(5, 4),
            aug_crop=0, aug_focal=1.0, preserve_fov_prob=0.0,
            sequence_consistent_aug_prob=0.0, train_augmentation=False,
            max_refetch=1,
        )

    def __len__(self):
        return 1

    def _get_views(self, idx, resolution, rng, num_views, preserve_fov, sequence_aug):
        views = []
        for frame in range(num_views):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = 0.2 * frame
            view = {
                "img": PIL.Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)),
                "depthmap": np.ones((4, 5), dtype=np.float32),
                "camera_intrinsics": np.array([[4, 0, 2], [0, 4, 1.5], [0, 0, 1]], dtype=np.float32),
                "camera_pose": pose,
            }
            if self.flag is not _MISSING:
                view["camera_only"] = self.flag
            views.append(view)
        return views


def test_base_collate_and_ground_truth_propagate_camera_only(tmp_path):
    camera = _SyntheticDataset(tmp_path, True)[0]
    rgbd = _SyntheticDataset(tmp_path, False)[0]
    assert all(view["camera_only"] is True for view in camera)
    assert all(view["camera_only"] is False for view in rgbd)
    collated = collate_views([camera, rgbd])
    gt = prepare_ground_truth(collated)
    assert gt["camera_only"].dtype == torch.bool
    assert gt["camera_only"].shape == (2,)
    torch.testing.assert_close(gt["camera_only"], torch.tensor([True, False]))
    # GT must retain incoming validity; the objective excludes camera-only later.
    torch.testing.assert_close(gt["valid_masks"], torch.stack([v["valid_mask"] for v in collated], dim=1))


def test_base_missing_flag_defaults_to_rgbd(tmp_path):
    sample = _SyntheticDataset(tmp_path)[0]
    assert all(view["camera_only"] is False for view in sample)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (True, [True, True]),
        (False, [False, False]),
        (torch.tensor(True), [True, True]),
        ([True, False], [True, False]),
        ((False, True), [False, True]),
        (torch.tensor([True, False]), [True, False]),
    ],
)
def test_ground_truth_flag_scalar_list_and_tensor(flag, expected):
    gt = prepare_ground_truth(_views(flag=flag))
    assert gt["camera_only"].shape == (2,)
    assert gt["camera_only"].dtype == torch.bool
    assert gt["camera_only"].device == gt["camera_poses"].device
    torch.testing.assert_close(gt["camera_only"], torch.tensor(expected))


@pytest.mark.parametrize("flag", [[True], [True, False, True], torch.tensor([True]), torch.tensor([True, False, True])])
def test_ground_truth_rejects_non_scalar_flag_with_wrong_batch_length(flag):
    with pytest.raises(ValueError):
        prepare_ground_truth(_views(batch=2, flag=flag))


def test_ground_truth_missing_flag_is_false():
    gt = prepare_ground_truth(_views())
    torch.testing.assert_close(gt["camera_only"], torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize("placeholder", [10000.0, float("nan"), float("inf")])
def test_camera_only_gt_normalization_does_not_use_placeholder_points(placeholder):
    views = _views(flag=[True, False])
    baseline = prepare_ground_truth(views)
    modified = copy.deepcopy(views)
    for view in modified:
        view["pts3d"][0].fill_(placeholder)
    changed = prepare_ground_truth(modified)
    torch.testing.assert_close(changed["camera_poses"], baseline["camera_poses"])
    torch.testing.assert_close(changed["local_points"][1], baseline["local_points"][1])
    torch.testing.assert_close(changed["valid_masks"], baseline["valid_masks"])
    expected = torch.stack([v["camera_pose"][0] for v in views])
    # First pose is identity; mean translation radius of [0,1/3,2/3,1] is .5.
    expected[:, :3, 3] /= 0.5
    torch.testing.assert_close(changed["camera_poses"][0], expected)


@pytest.mark.parametrize("invalid_as_zero", [False, True])
@pytest.mark.parametrize("nonfinite_points", [False, True])
def test_pure_camera_only_has_only_camera_gradients(invalid_as_zero, nonfinite_points):
    views = _views(batch=1, flag=True)
    prediction = _prediction(views)
    if nonfinite_points:
        for view in views:
            view["pts3d"].fill_(float("nan"))
        with torch.no_grad():
            prediction["local_points"].fill_(float("nan"))
    local, poses, logits = (prediction[k] for k in ("local_points", "camera_poses", "conf"))
    gt = prepare_ground_truth(views)
    point, details, scale = PointLoss(
        align_resolution=32, train_confidence=True,
        confidence_invalid_as_zero=invalid_as_zero,
    )(prediction, gt)
    camera, _ = CameraLoss(CameraLossConfig())(prediction, gt, scale)
    assert not scale.requires_grad
    assert torch.isfinite(scale).all() and torch.isfinite(camera)
    assert point == 0
    for key in ("local_pts_loss", "normal_loss", "confidence_loss", "confidence_valid_loss", "confidence_invalid_loss"):
        assert details[key] == 0, key
    (point + camera).backward()
    _assert_no_gradient(local)
    _assert_no_gradient(logits)
    assert poses.grad is not None and torch.isfinite(poses.grad).all()
    assert poses.grad[..., :3, 3].abs().sum() > 0


@pytest.mark.parametrize("invalid_as_zero", [False, True])
def test_mixed_camera_only_does_not_dilute_rgbd_losses_or_gradients(invalid_as_zero):
    views = _views(flag=[True, False])
    prediction = _prediction(views)
    single_views = _subset_views(views, 1)
    single_prediction = {key: value[1:].detach().clone().requires_grad_() for key, value in prediction.items()}
    for view in views:
        view["pts3d"][0].fill_(float("nan"))
    with torch.no_grad():
        prediction["local_points"][0].fill_(float("nan"))
    local, logits = prediction["local_points"], prediction["conf"]
    single_local, single_logits = single_prediction["local_points"], single_prediction["conf"]
    criterion = PointLoss(align_resolution=32, train_confidence=True, confidence_invalid_as_zero=invalid_as_zero)
    mixed_loss, mixed_details, mixed_scale = criterion(prediction, prepare_ground_truth(views))
    single_loss, single_details, single_scale = criterion(single_prediction, prepare_ground_truth(single_views))
    torch.testing.assert_close(mixed_loss, single_loss)
    torch.testing.assert_close(mixed_scale[1:], single_scale)
    for key in ("local_pts_loss", "normal_loss", "confidence_loss", "confidence_valid_loss", "confidence_invalid_loss"):
        torch.testing.assert_close(mixed_details[key], single_details[key])
    mixed_loss.backward()
    single_loss.backward()
    _assert_no_gradient(local, 0)
    _assert_no_gradient(logits, 0)
    assert local.grad is not None and logits.grad is not None
    torch.testing.assert_close(local.grad[1:], single_local.grad)
    torch.testing.assert_close(logits.grad[1:], single_logits.grad)
    assert local.grad[1].abs().sum() > 0
    assert logits.grad[1].abs().sum() > 0


@pytest.mark.parametrize("placeholder", [100000.0, float("nan"), float("inf")])
def test_camera_only_prediction_scale_and_camera_gradient_ignore_placeholder(placeholder):
    views = _views(flag=[True, False])
    gt = prepare_ground_truth(views)
    original = _prediction(views)
    modified = _clone_prediction(original)
    with torch.no_grad():
        modified["local_points"][0].fill_(placeholder)
    pose_leaves = [original["camera_poses"], modified["camera_poses"]]
    camera_results = []
    for prediction in (original, modified):
        _, _, scale = PointLoss(align_resolution=32)(prediction, gt)
        loss, details = CameraLoss(CameraLossConfig())(prediction, gt, scale)
        camera_results.append((scale, prediction["camera_poses"].detach(), loss.detach(), details["trans_loss"].detach()))
        loss.backward()
    for actual, expected in zip(camera_results[1], camera_results[0], strict=True):
        torch.testing.assert_close(actual, expected)
    for leaf in pose_leaves:
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    torch.testing.assert_close(pose_leaves[0].grad, pose_leaves[1].grad)


@pytest.mark.parametrize("train_confidence", [False, True])
def test_missing_flag_preserves_explicit_rgbd_losses_and_gradients(train_confidence):
    missing_views = _views()
    explicit_views = copy.deepcopy(missing_views)
    for view in explicit_views:
        view["camera_only"] = [False, False]
    missing_prediction = _prediction(missing_views)
    explicit_prediction = _clone_prediction(missing_prediction)
    first_leaves = dict(missing_prediction)
    second_leaves = dict(explicit_prediction)
    criterion = Pi3Loss(train_confidence=train_confidence)
    first_loss, first_details = criterion(missing_prediction, missing_views)
    second_loss, second_details = criterion(explicit_prediction, explicit_views)
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    assert first_details.keys() == second_details.keys()
    for key in first_details:
        torch.testing.assert_close(first_details[key], second_details[key], rtol=0, atol=0)
    first_loss.backward()
    second_loss.backward()
    for key in first_leaves:
        left, right = first_leaves[key].grad, second_leaves[key].grad
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_camera_normalizer_uses_frame_zero_relative_translation():
    poses = _poses([7, 9, 11])
    translated = poses.clone()
    translated[..., :3, 3] += torch.tensor([20.0, -9.0, 3.0])
    expected = torch.tensor([2.0])
    torch.testing.assert_close(loss_module._camera_normalizer(poses), expected)
    torch.testing.assert_close(loss_module._camera_normalizer(translated), expected)


@pytest.mark.parametrize(
    ("pred_x", "gt_x", "expected"),
    [
        ([0], [0], 1.0),                  # No adjacent pairs.
        ([0, 0, 0], [0, 0, 0], 1.0),      # Static trajectories.
        ([0, 1], [0, 2], 1.0),            # One pair is below the minimum.
        ([0, 1, 2], [0, 2, 4], 2.0),      # Exactly two valid pairs.
        ([0, 1, 1, 2], [0, 2, 2, 4], 2.0),  # Static gap excluded, two valid pairs remain.
        ([0, 0, 0], [0, 1, 2], 1.0),      # Zero predicted motion cannot define scale.
        ([0, 1, 2], [0, 0, 0], 1.0),      # Zero target motion cannot define scale.
        ([0, 1, 2, 3, 4], [0, 2, 4, 104, 106], 2.0),  # median/MAD removes a 100x outlier.
    ],
)
def test_align_camera_scale_robust_pairs_and_degenerate_fallbacks(pred_x, gt_x, expected):
    prediction = _poses(pred_x).requires_grad_()
    target = _poses(gt_x).requires_grad_()
    scale = loss_module.align_camera_scale(prediction, target)
    assert isinstance(scale, torch.Tensor) and scale.shape == (1,)
    assert torch.isfinite(scale).all() and not scale.requires_grad
    torch.testing.assert_close(scale, torch.tensor([expected]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("pair_mode", ["causal_upper", "dense"])
def test_zero_motion_camera_only_full_objective_is_finite(frames, pair_mode):
    views = _views(batch=1, frames=frames, flag=True)
    for view in views:
        view["camera_pose"] = torch.eye(4)[None]
    prediction = _prediction(views)
    with torch.no_grad():
        prediction["camera_poses"].copy_(torch.eye(4).repeat(1, frames, 1, 1))
    local, poses, logits = (prediction[k] for k in ("local_points", "camera_poses", "conf"))
    criterion = Pi3Loss(camera=CameraLossConfig(pair_mode=pair_mode), train_confidence=True)
    total, details = criterion(prediction, views)
    assert torch.isfinite(total)
    for value in details.values():
        assert torch.isfinite(value).all()
    total.backward()
    _assert_no_gradient(local)
    _assert_no_gradient(logits)
    assert poses.grad is not None and torch.isfinite(poses.grad).all()
    if frames == 1:
        assert details["camera_pairs"] == 0
        assert details["camera_loss"] == 0
