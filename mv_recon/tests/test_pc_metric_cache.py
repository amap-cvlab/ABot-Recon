import json

import numpy as np

from mv_recon.lingbot_protocol import evaluate_pointcloud
from mv_recon.pc_metric_cache import (
    load_pc_metric_cache,
    metrics_from_distances,
    recompute_cached_metrics,
    save_pc_metric_cache,
)


def _full_metrics():
    pred = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    metrics = evaluate_pointcloud(pred, gt, thresholds=[0.5, 1.5, 3.0])
    metrics.update(
        eval_threshold=0.5,
        eval_thresholds=[0.5, 1.5, 3.0],
        icp_threshold=0.5,
        voxel_size=0.05,
        with_scale=True,
        num_pred=len(pred),
        num_gt=len(gt),
        T_umeyama=np.eye(4),
        T_icp=np.eye(4),
    )
    return metrics


def test_metric_cache_roundtrip_and_threshold_sweep(tmp_path):
    path = tmp_path / "seq.npz"
    original = _full_metrics()
    save_pc_metric_cache(
        str(path),
        metrics=original,
        sequence_name="scene/seq-01",
        dataset_name="test",
        model_name="model",
    )
    loaded = load_pc_metric_cache(str(path))
    assert loaded["metadata"]["sequence"] == "scene/seq-01"
    assert loaded["pred_xyz"].shape == (2, 3)
    recomputed = recompute_cached_metrics(str(path), [0.5, 1.5, 3.0], 0.5)
    for key in ("accuracy", "completeness", "chamfer", "f1_0.5", "f1_1.5", "f1_3"):
        assert np.isclose(recomputed[key], original[key])


def test_metrics_from_distances_primary_aliases():
    metrics = metrics_from_distances(
        np.array([0.1, 2.0]), np.array([0.2, 0.3]), [0.25, 3.0], 0.25
    )
    assert metrics["precision"] == metrics["precision_0.25"]
    assert metrics["recall"] == metrics["recall_0.25"]
    assert metrics["f1"] == metrics["f1_0.25"]
    assert metrics["f1_3"] == 100.0
