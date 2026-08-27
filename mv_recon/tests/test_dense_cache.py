import json

import numpy as np
import pytest

from mv_recon.dense_cache import save_metric_frame_cache


def test_cache_preserves_sparse_frames_without_precision_loss(tmp_path):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(2, 3, 4, 3)).astype(np.float32)
    poses = np.repeat(np.eye(4)[None], 11, axis=0)
    poses[:, 0, 3] = np.arange(11)
    mask = rng.random((2, 3, 4)) > 0.2
    path = tmp_path / "seq.npz"
    save_metric_frame_cache(
        str(path),
        pred_world=points,
        pred_c2w=poses,
        pred_mask=mask,
        observation_mask=None,
        metric_indices=[0, 10],
        metric_frame_ids=[100, 110],
        sequence_name="sample",
        pred_depth=points[..., 2],
        pred_local_points=points,
    )
    with np.load(path, allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached["pred_world"], points)
        np.testing.assert_array_equal(cached["pred_c2w"], poses[[0, 10]])
        np.testing.assert_array_equal(cached["pred_mask"], mask)
        assert cached["pred_world"].dtype == np.float32
        assert cached["pred_c2w"].dtype == np.float64
        assert cached["observation_mask"].all()
        np.testing.assert_array_equal(cached["pred_depth"], points[..., 2])
        np.testing.assert_array_equal(cached["pred_local_points"], points)
        assert json.loads(str(cached["metadata_json"]))["sequence"] == "sample"


def test_cache_rejects_bad_index_mapping(tmp_path):
    with pytest.raises(ValueError, match="Index count"):
        save_metric_frame_cache(
            str(tmp_path / "bad.npz"),
            pred_world=np.zeros((2, 2, 2, 3), dtype=np.float32),
            pred_c2w=np.repeat(np.eye(4)[None], 4, axis=0),
            pred_mask=None,
            observation_mask=None,
            metric_indices=[0],
            metric_frame_ids=[0, 1],
            sequence_name="bad",
        )
