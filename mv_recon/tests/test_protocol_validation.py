import numpy as np
import pytest

from mv_recon.protocol_validation import validate_formal_pointcloud_protocol


def _poses(count=3):
    return np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)


def test_formal_oxford_contract_accepts_documented_horizon_runtime():
    validate_formal_pointcloud_protocol(
        model_name="horizonstream",
        dataset_name="Oxford-Spires-S1-I10",
        runtime={"input_h": 378, "input_w": 518, "forward_compute_dtype": "float16"},
        pred_c2w=_poses(),
        with_scale=True,
        metric_frame_ids=[0, 10, 20],
        eval_threshold=4.0,
        eval_thresholds=[2.0, 4.0],
        voxel_size=0.05,
        icp_threshold=0.5,
        nearest_depth_to_gt=True,
        pointmap_resize_mode="nearest",
        alignment_depth_max=80.0,
        observation_mask=np.ones((3, 2, 2), dtype=bool),
    )


@pytest.mark.parametrize("model_name", ["horizonstream", "horizonstream_sim3", "horizonstream_se3"])
def test_formal_tum_contract_accepts_horizon_aliases(model_name):
    validate_formal_pointcloud_protocol(
        model_name=model_name,
        dataset_name="TUM-Dynamics-Full",
        runtime={"input_h": 378, "input_w": 518, "forward_compute_dtype": "fp16"},
        pred_c2w=_poses(),
        with_scale=True,
        metric_frame_ids=None,
        eval_thresholds=[0.05, 0.25],
        nearest_depth_to_gt=True,
        pointmap_resize_mode="nearest",
        alignment_depth_max=40.0,
        observation_mask=np.ones((3, 2, 2), dtype=bool),
    )


@pytest.mark.parametrize(
    "model,runtime",
    [
        ("cut3r", {"input_h": 384, "input_w": 512, "forward_compute_dtype": "bf16"}),
        ("horizonstream", {"input_h": 378, "input_w": 518, "forward_compute_dtype": "bf16"}),
        ("horizonstream_sim3", {"input_h": 378, "input_w": 518, "forward_compute_dtype": "bf16"}),
    ],
)
def test_formal_contract_rejects_wrong_forward_dtype(model, runtime):
    with pytest.raises(RuntimeError, match="formal point-cloud forward must"):
        validate_formal_pointcloud_protocol(
            model_name=model,
            dataset_name="Oxford-Spires-S1-I10",
            runtime=runtime,
            pred_c2w=_poses(),
            with_scale=True,
            metric_frame_ids=[0, 10],
        )


def test_formal_contract_rejects_wrong_shape_or_oxford_indices():
    runtime = {"input_h": 392, "input_w": 518, "forward_compute_dtype": "bf16"}
    with pytest.raises(RuntimeError, match="input shape mismatch"):
        validate_formal_pointcloud_protocol(
            model_name="lingbot_map",
            dataset_name="Oxford-Spires-S1-I10",
            runtime=runtime,
            pred_c2w=_poses(),
            with_scale=True,
            metric_frame_ids=[0, 10],
        )
    runtime = {"input_h": 378, "input_w": 518, "forward_compute_dtype": "bf16"}
    with pytest.raises(RuntimeError, match="exactly 0,10,20"):
        validate_formal_pointcloud_protocol(
            model_name="lingbot_map",
            dataset_name="Oxford-Spires-S1-I10",
            runtime=runtime,
            pred_c2w=_poses(),
            with_scale=True,
            eval_threshold=4.0,
            metric_frame_ids=[0, 11],
        )


def test_formal_contract_rejects_non_so3_pose_and_se3_alignment():
    runtime = {
        "input_h": 280,
        "input_w": 504,
        "forward_compute_dtype": "bf16",
        "online_state": "causal_stream+paged-kv-true",
    }
    with pytest.raises(RuntimeError, match="requires Sim3"):
        validate_formal_pointcloud_protocol(
            model_name="abot_recon",
            dataset_name="7scenes-hs-paper",
            runtime=runtime,
            pred_c2w=_poses(),
            with_scale=False,
            metric_frame_ids=None,
        )
    bad = _poses()
    bad[1, 0, 0] = 2.0
    with pytest.raises(RuntimeError, match=r"not proper SO\(3\)"):
        validate_formal_pointcloud_protocol(
            model_name="abot_recon",
            dataset_name="7scenes-hs-paper",
            runtime=runtime,
            pred_c2w=bad,
            with_scale=True,
            metric_frame_ids=None,
            observation_mask=np.ones((3, 2, 2), dtype=bool),
        )


def test_formal_contract_rejects_wrong_oxford_geometry_or_f1_protocol():
    runtime = {"input_h": 378, "input_w": 518, "forward_compute_dtype": "fp16"}
    common = dict(
        model_name="horizonstream",
        dataset_name="Oxford-Spires-S1-I10",
        runtime=runtime,
        pred_c2w=_poses(),
        with_scale=True,
        metric_frame_ids=[0, 10, 20],
        alignment_depth_max=80.0,
        eval_threshold=4.0,
        observation_mask=np.ones((3, 2, 2), dtype=bool),
    )
    with pytest.raises(RuntimeError, match="primary F1 threshold must be 4.0"):
        validate_formal_pointcloud_protocol(
            **{**common, "eval_threshold": 2.0},
            eval_thresholds=[2.0, 4.0], voxel_size=0.05, icp_threshold=0.5
        )
    with pytest.raises(RuntimeError, match="F1 thresholds mismatch"):
        validate_formal_pointcloud_protocol(
            **common, eval_thresholds=[1.0, 2.0], voxel_size=0.05, icp_threshold=0.5
        )
    with pytest.raises(RuntimeError, match="voxel mismatch"):
        validate_formal_pointcloud_protocol(
            **common, eval_thresholds=[2.0, 4.0], voxel_size=0.01, icp_threshold=0.5
        )


def test_formal_abot_recon_requires_paged_kv_runtime():
    runtime = {
        "input_h": 280,
        "input_w": 504,
        "forward_compute_dtype": "bf16",
        "online_state": "causal_stream+paged-kv-false",
    }
    with pytest.raises(RuntimeError, match="requires paged KV"):
        validate_formal_pointcloud_protocol(
            model_name="abot_recon",
            dataset_name="7scenes-hs-paper",
            runtime=runtime,
            pred_c2w=_poses(),
            with_scale=True,
            metric_frame_ids=None,
        )


def test_formal_contract_rejects_wrong_resize_alignment_or_missing_roi():
    runtime = {"input_h": 378, "input_w": 518, "forward_compute_dtype": "fp16"}
    common = dict(
        model_name="horizonstream",
        dataset_name="Oxford-Spires-S1-I10",
        runtime=runtime,
        pred_c2w=_poses(),
        with_scale=True,
        metric_frame_ids=[0, 10, 20],
        eval_threshold=4.0,
        eval_thresholds=[2.0, 4.0],
        voxel_size=0.05,
        icp_threshold=0.5,
    )
    with pytest.raises(RuntimeError, match="nearest XYZ resize"):
        validate_formal_pointcloud_protocol(
            **common,
            nearest_depth_to_gt=True,
            pointmap_resize_mode="bilinear",
            alignment_depth_max=80.0,
            observation_mask=np.ones((3, 2, 2), dtype=bool),
        )
    with pytest.raises(RuntimeError, match="alignment depth cutoff mismatch"):
        validate_formal_pointcloud_protocol(
            **common,
            nearest_depth_to_gt=True,
            pointmap_resize_mode="nearest",
            alignment_depth_max=40.0,
            observation_mask=np.ones((3, 2, 2), dtype=bool),
        )
    with pytest.raises(RuntimeError, match="observed-FOV mask"):
        validate_formal_pointcloud_protocol(
            **common,
            nearest_depth_to_gt=True,
            pointmap_resize_mode="nearest",
            alignment_depth_max=80.0,
            observation_mask=None,
        )
