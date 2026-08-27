import numpy as np
import pytest

from mv_recon.depth_metrics import evaluate_scale_aligned_depth, evaluate_scale_aligned_depth_maps


def _translation(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def test_sparse_pose_selection_and_scale_recovery():
    gt_c2w = np.stack([_translation(0.0), _translation(2.0)])
    full_pred_c2w = np.stack([_translation(x) for x in (0.0, 0.5, 1.0, 1.5, 2.0)])
    local_gt = np.zeros((2, 10, 10, 3), dtype=np.float64)
    local_gt[..., 2] = 4.0
    gt_world = local_gt.copy()
    gt_world[1, ..., 0] += 2.0
    # Prediction has exactly half the depth scale, while poses remain full-length.
    pred_world = local_gt.copy() * 0.5
    pred_world[1, ..., 0] += 2.0
    gt_w2c = np.linalg.inv(gt_c2w)
    result = evaluate_scale_aligned_depth(
        pred_world=pred_world,
        pred_c2w=full_pred_c2w,
        gt_world=gt_world,
        gt_w2c=gt_w2c,
        gt_mask=np.ones((2, 10, 10), dtype=bool),
        metric_indices=[0, 4],
    )
    assert result["depth_scale"] == pytest.approx(2.0)
    assert result["abs_rel"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["delta1"] == pytest.approx(100.0)


def test_common_masks_exclude_invalid_outlier():
    world = np.zeros((1, 11, 10, 3), dtype=np.float64)
    world[..., 2] = 3.0
    pred = world.copy()
    pred[0, 0, 0, 2] = 300.0
    pred_mask = np.ones((1, 11, 10), dtype=bool)
    pred_mask[0, 0, 0] = False
    result = evaluate_scale_aligned_depth(
        pred_world=pred,
        pred_c2w=np.eye(4)[None],
        gt_world=world,
        gt_w2c=np.eye(4)[None],
        gt_mask=np.ones((1, 11, 10), dtype=bool),
        pred_mask=pred_mask,
    )
    assert result["num_depth_pixels"] == 109
    assert result["abs_rel"] == pytest.approx(0.0)


def test_rejects_ambiguous_pose_count():
    world = np.zeros((2, 10, 10, 3), dtype=np.float64)
    world[..., 2] = 1.0
    with pytest.raises(ValueError, match="Pose count"):
        evaluate_scale_aligned_depth(
            pred_world=world,
            pred_c2w=np.repeat(np.eye(4)[None], 3, axis=0),
            gt_world=world,
            gt_w2c=np.repeat(np.eye(4)[None], 2, axis=0),
            gt_mask=np.ones((2, 10, 10), dtype=bool),
        )


def test_explicit_depth_map_scale_is_pose_independent():
    gt = np.full((2, 10, 10), 8.0)
    pred = gt / 4.0
    result = evaluate_scale_aligned_depth_maps(
        pred_depth=pred,
        gt_depth=gt,
        gt_mask=np.ones_like(gt, dtype=bool),
        alignment_gt_depth_max=None,
    )
    assert result["depth_scale"] == pytest.approx(4.0)
    assert result["abs_rel"] == pytest.approx(0.0)
    assert result["delta1"] == pytest.approx(100.0)
    assert not any(key.startswith("raw_") for key in result)


def test_scale_uses_near_40m_but_metrics_keep_far_pixels():
    gt = np.full((1, 11, 10), 20.0)
    pred = gt / 2.0
    gt[:, 0, :] = 80.0
    pred[:, 0, :] = 10.0
    result = evaluate_scale_aligned_depth_maps(
        pred_depth=pred,
        gt_depth=gt,
        gt_mask=np.ones_like(gt, dtype=bool),
        alignment_gt_depth_max=40.0,
    )
    assert result["depth_scale"] == pytest.approx(2.0)
    assert result["num_scale_pixels"] == 100
    assert result["num_depth_pixels"] == 110
    assert result["abs_rel"] > 0.0
    assert result["rmse"] > 0.0
