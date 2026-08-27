"""Unit tests for lingbot-aligned mv_recon point-cloud protocol.

Run from the evaluation repository root:
    python -m pytest mv_recon/tests/test_lingbot_protocol.py -v
or:
    python mv_recon/tests/test_lingbot_protocol.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Allow importing mv_recon without hydra/rootutils.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mv_recon.lingbot_protocol import (  # noqa: E402
    DEFAULT_EVAL_THRESHOLD_INDOOR,
    DEFAULT_EVAL_THRESHOLD_TUM,
    DEFAULT_VOXEL_SIZE,
    METRIC_KEYS,
    accuracy,
    apply_transform,
    chamfer_distance,
    completeness,
    evaluate_pointcloud,
    evaluate_reconstruction,
    f1_from_pr,
    get_dataset_eval_options,
    icp_registration,
    metrics_to_csv_row,
    nearest_neighbor_distances,
    precision_at_thresholds,
    recall_at_thresholds,
    resolve_dataset_eval_options,
    restrict_masks_to_observed_fov,
    umeyama_registration,
    voxel_downsample,
)


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _sim3(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


class TestDatasetOptions(unittest.TestCase):
    def test_indoor_defaults(self):
        for name in ("7scenes-dense", "7scenes-sparse"):
            opts = get_dataset_eval_options(name)
            self.assertAlmostEqual(opts["icp_threshold"], 0.1)
            self.assertAlmostEqual(opts["voxel_size"], DEFAULT_VOXEL_SIZE)
            self.assertAlmostEqual(opts["eval_threshold"], DEFAULT_EVAL_THRESHOLD_INDOOR)

    def test_hs_paper_7scenes_f1_025(self):
        opts = get_dataset_eval_options("7scenes-hs-paper")
        self.assertAlmostEqual(opts["icp_threshold"], 0.1)
        self.assertAlmostEqual(opts["voxel_size"], DEFAULT_VOXEL_SIZE)
        self.assertAlmostEqual(opts["eval_threshold"], DEFAULT_EVAL_THRESHOLD_TUM)

    def test_explicit_f1_threshold_override(self):
        opts = resolve_dataset_eval_options("7scenes-hs-paper", 0.25)
        self.assertAlmostEqual(opts["eval_threshold"], 0.25)
        self.assertAlmostEqual(opts["icp_threshold"], 0.1)
        self.assertAlmostEqual(opts["voxel_size"], DEFAULT_VOXEL_SIZE)

    def test_invalid_f1_threshold_override(self):
        with self.assertRaises(ValueError):
            resolve_dataset_eval_options("7scenes-hs-paper", 0.0)

class TestNearestNeighborMetrics(unittest.TestCase):
    def test_identical_clouds_zero_error(self):
        pts = np.random.RandomState(0).randn(200, 3).astype(np.float64)
        self.assertAlmostEqual(accuracy(pts, pts), 0.0, places=7)
        self.assertAlmostEqual(completeness(pts, pts), 0.0, places=7)
        m = evaluate_pointcloud(pts, pts, thresholds=[0.05])
        self.assertAlmostEqual(m["accuracy"], 0.0, places=7)
        self.assertAlmostEqual(m["completeness"], 0.0, places=7)
        self.assertAlmostEqual(m["chamfer"], 0.0, places=7)
        self.assertAlmostEqual(m["precision"], 100.0, places=5)
        self.assertAlmostEqual(m["recall"], 100.0, places=5)
        self.assertAlmostEqual(m["f1"], 100.0, places=5)

    def test_uniform_translation_accuracy(self):
        gt = np.random.RandomState(1).randn(150, 3)
        offset = 0.03
        pred = gt + np.array([offset, 0.0, 0.0])
        acc = accuracy(pred, gt)
        comp = completeness(pred, gt)
        self.assertAlmostEqual(acc, offset, places=5)
        self.assertAlmostEqual(comp, offset, places=5)
        self.assertAlmostEqual(chamfer_distance(acc, comp), offset, places=5)

    def test_precision_recall_threshold(self):
        gt = np.zeros((10, 3))
        gt[:, 0] = np.arange(10) * 0.1
        # 5 points within 0.05 of a GT point, 5 far away
        pred = gt.copy()
        pred[5:] += 1.0
        prec = precision_at_thresholds(pred, gt, [0.05])[0]
        rec = recall_at_thresholds(pred, gt, [0.05])[0]
        self.assertAlmostEqual(prec, 50.0, places=5)
        self.assertAlmostEqual(rec, 50.0, places=5)
        self.assertAlmostEqual(f1_from_pr([prec], [rec])[0], 50.0, places=5)

    def test_f1_zero_when_both_zero(self):
        self.assertEqual(f1_from_pr([0.0], [0.0])[0], 0.0)

    def test_multi_threshold_keys(self):
        pts = np.random.RandomState(2).randn(50, 3)
        m = evaluate_pointcloud(pts, pts + 0.001, thresholds=[0.01, 0.05])
        self.assertIn("precision_0.01", m)
        self.assertIn("recall_0.05", m)
        self.assertIn("f1_0.01", m)
        self.assertNotIn("precision", m)

    def test_chamfer_is_mean_of_acc_comp(self):
        a = np.random.RandomState(3).randn(80, 3)
        b = a + 0.02
        m = evaluate_pointcloud(a, b, thresholds=[0.05])
        self.assertAlmostEqual(
            m["chamfer"], (m["accuracy"] + m["completeness"]) / 2.0, places=7
        )

    def test_nn_distances_shape(self):
        src = np.random.RandomState(4).randn(17, 3)
        tgt = np.random.RandomState(5).randn(23, 3)
        d = nearest_neighbor_distances(src, tgt)
        self.assertEqual(d.shape, (17,))


class TestRegistration(unittest.TestCase):
    def test_apply_transform_translation(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        T = np.eye(4)
        T[:3, 3] = [1.0, -1.0, 0.5]
        out = apply_transform(pts, T)
        np.testing.assert_allclose(out, pts + np.array([1.0, -1.0, 0.5]), atol=1e-8)

    def test_apply_transform_preserves_extra_channels(self):
        pts = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        T = np.eye(4)
        T[:3, 3] = [1, 0, 0]
        out = apply_transform(pts, T)
        np.testing.assert_allclose(out[0, 3:], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(out[0, :3], [2.0, 2.0, 3.0])

    def test_umeyama_recovers_sim3(self):
        rng = np.random.RandomState(6)
        src = rng.randn(300, 3)
        scale = 1.7
        R = _rotation_z(0.4)
        t = np.array([0.5, -0.2, 0.1])
        T_gt = _sim3(scale, R, t)
        tgt = apply_transform(src, T_gt)
        T_est = umeyama_registration(src, tgt)
        aligned = apply_transform(src, T_est)
        np.testing.assert_allclose(aligned, tgt, atol=1e-6)
        np.testing.assert_allclose(T_est, T_gt, atol=1e-6)

    def test_umeyama_se3_keeps_unit_scale(self):
        rng = np.random.RandomState(61)
        src = rng.randn(300, 3)
        R = _rotation_z(0.4)
        t = np.array([0.5, -0.2, 0.1])
        # GT differs by RT only; SE3 must recover it with scale==1.
        T_rt = _sim3(1.0, R, t)
        tgt = apply_transform(src, T_rt)
        T_est = umeyama_registration(src, tgt, with_scale=False)
        s = np.linalg.norm(T_est[:3, 0])
        self.assertAlmostEqual(s, 1.0, places=6)
        aligned = apply_transform(src, T_est)
        np.testing.assert_allclose(aligned, tgt, atol=1e-6)

    def test_umeyama_se3_does_not_absorb_scale(self):
        rng = np.random.RandomState(62)
        src = rng.randn(400, 3)
        T_gt = _sim3(1.7, _rotation_z(0.2), np.array([0.1, 0.0, -0.2]))
        tgt = apply_transform(src, T_gt)
        T_est = umeyama_registration(src, tgt, with_scale=False)
        s = np.linalg.norm(T_est[:3, 0])
        self.assertAlmostEqual(s, 1.0, places=6)
        # Residual should remain large vs Sim3 recovery.
        err = np.mean(np.linalg.norm(apply_transform(src, T_est) - tgt, axis=1))
        self.assertGreater(err, 0.1)

    def test_umeyama_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            umeyama_registration(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_icp_aligns_small_offset(self):
        rng = np.random.RandomState(7)
        tgt = rng.randn(500, 3)
        src = tgt + np.array([0.02, -0.01, 0.015])
        T = icp_registration(src, tgt, icp_threshold=0.1, max_iterations=50)
        aligned = apply_transform(src, T)
        err = np.mean(np.linalg.norm(aligned - tgt, axis=1))
        self.assertLess(err, 1e-3)

    def test_voxel_downsample_reduces_count(self):
        rng = np.random.RandomState(8)
        pts = rng.rand(2000, 3).astype(np.float32) * 0.5
        down = voxel_downsample(pts, voxel_size=0.05)
        self.assertLess(len(down), len(pts))
        self.assertEqual(down.shape[1], 3)

    def test_voxel_disabled(self):
        pts = np.random.RandomState(9).rand(100, 3).astype(np.float32)
        down = voxel_downsample(pts, voxel_size=0.0)
        np.testing.assert_array_equal(down, pts)


class TestEvaluateReconstruction(unittest.TestCase):
    def _make_grid(self, n=2, h=8, w=8, seed=0):
        rng = np.random.RandomState(seed)
        gt = rng.randn(n, h, w, 3).astype(np.float64)
        mask = np.ones((n, h, w), dtype=bool)
        # Punch a hole in GT
        mask[0, 0, 0] = False
        return gt, mask

    def test_perfect_prediction(self):
        gt, mask = self._make_grid()
        pred = gt.copy()
        m = evaluate_reconstruction(
            pred, gt, mask,
            icp_threshold=0.1,
            voxel_size=0.0,
            eval_threshold=0.05,
        )
        self.assertAlmostEqual(m["accuracy"], 0.0, places=5)
        self.assertAlmostEqual(m["completeness"], 0.0, places=5)
        self.assertAlmostEqual(m["chamfer"], 0.0, places=5)
        self.assertAlmostEqual(m["f1"], 100.0, places=3)
        self.assertEqual(m["num_gt"], int(mask.sum()))
        self.assertEqual(m["num_pred"], int(mask.sum()))  # common == gt when pred all valid

    def test_sim3_misalignment_recovered(self):
        gt, mask = self._make_grid(seed=10)
        T = _sim3(1.25, _rotation_z(0.3), np.array([0.2, -0.1, 0.05]))
        pred = apply_transform(gt, np.linalg.inv(T))
        m = evaluate_reconstruction(
            pred, gt, mask,
            icp_threshold=0.5,
            voxel_size=0.0,
            eval_threshold=0.05,
        )
        self.assertLess(m["chamfer"], 0.01)
        self.assertGreater(m["f1"], 95.0)
        self.assertTrue(m["with_scale"])

    def test_se3_recovers_rt_only(self):
        gt, mask = self._make_grid(seed=12)
        T = _sim3(1.0, _rotation_z(0.25), np.array([0.15, -0.05, 0.08]))
        pred = apply_transform(gt, np.linalg.inv(T))
        m = evaluate_reconstruction(
            pred, gt, mask,
            icp_threshold=0.5,
            voxel_size=0.0,
            eval_threshold=0.05,
            with_scale=False,
        )
        self.assertFalse(m["with_scale"])
        self.assertLess(m["chamfer"], 0.01)
        self.assertGreater(m["f1"], 95.0)

    def test_common_mask_excludes_invalid_pred(self):
        gt, mask = self._make_grid(seed=11)
        pred = gt.copy()
        pred_mask = mask.copy()
        # Invalidate half of pred pixels (still GT-valid)
        pred_mask[:, :, :4] = False
        pred[:, :, :4] = np.nan
        m = evaluate_reconstruction(
            pred, gt, mask,
            pred_mask=pred_mask,
            icp_threshold=0.1,
            voxel_size=0.0,
            eval_threshold=0.05,
        )
        # Pred eval uses common_mask only; GT uses full gt_mask.
        self.assertEqual(m["num_correspondences"], int((mask & pred_mask).sum()))
        self.assertEqual(m["num_gt"], int(mask.sum()))
        self.assertLess(m["num_pred"], m["num_gt"])

    def test_alignment_mask_does_not_remove_points_from_metrics(self):
        rng = np.random.RandomState(123)
        gt = rng.uniform(-1.0, 1.0, size=(1, 4, 4, 3)).astype(np.float64)
        pred = gt / 2.0
        pred[0, 0, 0] = np.array([1000.0, 1000.0, 1000.0])
        valid = np.ones((1, 4, 4), dtype=bool)
        align_valid = valid.copy()
        align_valid[0, 0, 0] = False

        m = evaluate_reconstruction(
            pred,
            gt,
            valid,
            pred_mask=valid,
            alignment_pred_mask=align_valid,
            icp_threshold=0.01,
            voxel_size=0.0,
            eval_threshold=0.05,
            with_scale=True,
        )

        self.assertEqual(m["num_correspondences"], 16)
        self.assertEqual(m["num_alignment_correspondences"], 15)
        self.assertEqual(m["num_pred"], 16)
        recovered_scale = np.linalg.norm(m["T_umeyama"][:3, 0])
        self.assertAlmostEqual(recovered_scale, 2.0, places=5)
        # The excluded alignment outlier is still scored, so Accuracy is bad.
        self.assertGreater(m["accuracy"], 10.0)

    def test_gt_alignment_mask_only_changes_umeyama_correspondences(self):
        rng = np.random.RandomState(124)
        gt = rng.uniform(-1.0, 1.0, size=(1, 4, 4, 3)).astype(np.float64)
        pred = gt / 3.0
        pred[0, 0, 0] = np.array([500.0, 500.0, 500.0])
        valid = np.ones((1, 4, 4), dtype=bool)
        alignment_gt = valid.copy()
        alignment_gt[0, 0, 0] = False

        m = evaluate_reconstruction(
            pred,
            gt,
            valid,
            pred_mask=valid,
            alignment_gt_mask=alignment_gt,
            icp_threshold=0.01,
            voxel_size=0.0,
            eval_threshold=0.05,
            with_scale=True,
        )

        self.assertEqual(m["num_correspondences"], 16)
        self.assertEqual(m["num_alignment_correspondences"], 15)
        self.assertEqual(m["num_pred"], 16)
        self.assertAlmostEqual(np.linalg.norm(m["T_umeyama"][:3, 0]), 3.0, places=5)
        self.assertGreater(m["accuracy"], 10.0)

    def test_unobserved_crop_is_excluded_from_completeness(self):
        gt, raw_gt_mask = self._make_grid(n=1, h=8, w=8, seed=21)
        pred = gt.copy()
        pred_mask = np.ones_like(raw_gt_mask)
        observed = np.zeros_like(raw_gt_mask)
        observed[:, 2:6, :] = True

        # Deliberately corrupt pixels outside the model's observed crop. They
        # must affect neither the GT cloud used by Comp nor prediction points.
        pred[~observed] += 1000.0
        raw_gt_before = raw_gt_mask.copy()
        pred_mask_before = pred_mask.copy()
        eval_gt_mask, eval_pred_mask = restrict_masks_to_observed_fov(
            raw_gt_mask, pred_mask, observed
        )

        np.testing.assert_array_equal(raw_gt_mask, raw_gt_before)
        np.testing.assert_array_equal(pred_mask, pred_mask_before)
        np.testing.assert_array_equal(eval_gt_mask, raw_gt_mask & observed)
        np.testing.assert_array_equal(eval_pred_mask, pred_mask & observed)

        m = evaluate_reconstruction(
            pred,
            gt,
            eval_gt_mask,
            pred_mask=eval_pred_mask,
            icp_threshold=0.1,
            voxel_size=0.0,
            eval_threshold=0.05,
        )
        self.assertEqual(m["num_gt"], int((raw_gt_mask & observed).sum()))
        self.assertAlmostEqual(m["accuracy"], 0.0, places=5)
        self.assertAlmostEqual(m["completeness"], 0.0, places=5)

    def test_insufficient_correspondences(self):
        gt = np.zeros((1, 2, 2, 3))
        mask = np.zeros((1, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        pred = gt.copy()
        with self.assertRaises(ValueError):
            evaluate_reconstruction(pred, gt, mask, voxel_size=0.0)

    def test_indoor_voxel_path_runs(self):
        gt, mask = self._make_grid(n=1, h=16, w=16, seed=12)
        # Spread points so voxel keeps more than a few
        coords = np.stack(
            np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16), indexing="ij"),
            axis=-1,
        )
        gt[0, :, :, 0] = coords[..., 0]
        gt[0, :, :, 1] = coords[..., 1]
        gt[0, :, :, 2] = 0.0
        pred = gt + 0.001
        m = evaluate_reconstruction(
            pred, gt, mask,
            icp_threshold=0.1,
            voxel_size=DEFAULT_VOXEL_SIZE,
            eval_threshold=0.05,
        )
        for key in METRIC_KEYS:
            self.assertIn(key, m)
        self.assertLess(m["chamfer"], 0.02)
        self.assertGreater(m["f1"], 80.0)

    def test_csv_row_keys(self):
        gt, mask = self._make_grid(seed=13)
        m = evaluate_reconstruction(gt, gt, mask, voxel_size=0.0)
        row = metrics_to_csv_row("seq0", m)
        self.assertEqual(row["seq"], "seq0")
        for key in METRIC_KEYS:
            self.assertIn(key, row)

    def test_multiple_f1_thresholds_share_one_reconstruction(self):
        gt, mask = self._make_grid(n=1, h=8, w=8, seed=17)
        m = evaluate_reconstruction(
            gt,
            gt,
            mask,
            voxel_size=0.0,
            eval_threshold=0.25,
            eval_thresholds=[0.05, 0.25],
        )
        self.assertIn("f1_0.05", m)
        self.assertIn("f1_0.25", m)
        self.assertEqual(m["f1"], m["f1_0.25"])
        row = metrics_to_csv_row("seq0", m)
        self.assertNotIn("precision_0.05", row)
        self.assertNotIn("recall_0.25", row)
        self.assertIn("f1_0.05", row)
        self.assertIn("f1_0.25", row)

        raw = evaluate_pointcloud(
            gt.reshape(-1, 3),
            (gt + np.array([0.10, 0.0, 0.0])).reshape(-1, 3),
            thresholds=[0.05, 0.25],
        )
        self.assertLess(raw["f1_0.05"], raw["f1_0.25"])

    def test_shape_mismatch(self):
        gt = np.zeros((1, 4, 4, 3))
        pred = np.zeros((1, 4, 5, 3))
        mask = np.ones((1, 4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            evaluate_reconstruction(pred, gt, mask, voxel_size=0.0)


class TestAgainstReferenceFormulas(unittest.TestCase):
    """Sanity checks that match lingbot points.py definitions literally."""

    def test_accuracy_is_mean_not_median(self):
        # One outlier: mean >> median
        gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 10.0]])
        dists = nearest_neighbor_distances(pred, gt)
        self.assertAlmostEqual(accuracy(pred, gt), float(np.mean(dists)))
        self.assertNotAlmostEqual(accuracy(pred, gt), float(np.median(dists)))

    def test_completeness_direction(self):
        gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        pred = np.array([[0.0, 0.0, 0.0]])  # covers only first GT point
        # Acc: single pred point → near GT (0)
        self.assertAlmostEqual(accuracy(pred, gt), 0.0, places=7)
        # Comp: average of 0 and ~10
        self.assertAlmostEqual(completeness(pred, gt), 5.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
