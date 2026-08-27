from pathlib import Path

import json
import numpy as np
import pytest
import torch
from PIL import Image

from datasets.oxford_spires_mv_recon import (
    OxfordSpiresMVRecon,
    resize_sparse_depth_and_intrinsic,
)
from mv_recon.eval import (
    resolve_model_alignment_depth_max,
    select_metric_dense_outputs,
)
from omegaconf import OmegaConf
from mv_recon.lingbot_protocol import (
    get_dataset_eval_options,
    resolve_lingbot_prepare_width,
)
from interfaces.lingbot_map import (
    LazyLingBotImages,
    _area_budget_hw,
    _assert_expected_input_hw,
    _prepare_width_aligned_hw,
    _load_area_budget,
)
from interfaces.abot_recon import (
    _local_points_model_res,
    _poses_c2w_from_out,
    _world_points_model_res,
)


def _write_fixture(root: Path):
    image_root = root / "images"
    gt_root = root / "gt"
    name = "keble-college-test"
    processed = image_root / name
    gt = gt_root / name
    (processed / "images").mkdir(parents=True)
    (gt / "depth").mkdir(parents=True)
    poses = []
    for frame_id in range(21):
        Image.fromarray(np.full((6, 8, 3), frame_id, np.uint8)).save(
            processed / "images" / f"{frame_id:06d}.png"
        )
        pose = np.eye(4)
        pose[0, 3] = frame_id
        poses.append(pose)
    poses = np.stack(poses)
    np.savetxt(processed / "poses_c2w.txt", poses.reshape(21, 16))
    np.savetxt(processed / "intrinsics.txt", [4, 4, 3.5, 2.5, 8, 6])
    frame_ids = np.array([0, 10, 20])
    np.savetxt(gt / "frame_ids.txt", frame_ids, fmt="%d")
    np.savetxt(gt / "poses_c2w.txt", poses[frame_ids].reshape(3, 16))
    for frame_id in frame_ids:
        depth = np.zeros((6, 8), np.float32)
        depth[2, 2] = 2.0 + frame_id
        np.save(gt / "depth" / f"{frame_id:06d}.npy", depth)
    (gt / "DONE.json").write_text(
        json.dumps(
            {
                "source_frame_count": 21,
                "sampled_frame_count": 3,
                "interval": 10,
                "width": 8,
                "height": 6,
            }
        )
    )
    return image_root, gt_root, name


def test_stride1_inference_and_interval10_metric_mapping(tmp_path):
    image_root, gt_root, name = _write_fixture(tmp_path)
    dataset = OxfordSpiresMVRecon(
        str(image_root), str(gt_root), load_img_size=4, expected_interval=10, depth_max=50
    )
    data = dataset.get_data(sequence_name=name, ids=list(range(21)))
    assert len(data["image_paths"]) == 21
    assert data["metric_indices"].tolist() == [0, 10, 20]
    assert data["metric_frame_ids"].tolist() == [0, 10, 20]
    assert data["pointclouds"].shape == (3, 3, 4, 3)
    assert data["valid_mask"].sum(axis=(1, 2)).tolist() == [1, 1, 1]
    assert data["alignment_gt_mask"].sum(axis=(1, 2)).tolist() == [1, 1, 1]
    assert data["images"] is None
    assert data["image_hw"] == (3, 4)
    np.testing.assert_allclose(data["extrs"][10, 0, 3], -10.0)


def test_nonzero_requested_start_maps_positions_not_source_ids(tmp_path):
    image_root, gt_root, name = _write_fixture(tmp_path)
    dataset = OxfordSpiresMVRecon(str(image_root), str(gt_root), load_img_size=4)
    data = dataset.get_data(sequence_name=name, ids=list(range(5, 16)))
    assert data["metric_indices"].tolist() == [5]
    assert data["metric_frame_ids"].tolist() == [10]


def test_alignment_gt_depth_cutoff_does_not_change_metric_validity(tmp_path):
    image_root, gt_root, name = _write_fixture(tmp_path)
    depth_path = gt_root / name / "depth" / "000020.npy"
    depth = np.load(depth_path)
    depth[2, 2] = 120.0
    np.save(depth_path, depth)
    dataset = OxfordSpiresMVRecon(
        str(image_root),
        str(gt_root),
        load_img_size=4,
        depth_max=200.0,
        alignment_depth_max=80.0,
    )
    data = dataset.get_data(sequence_name=name, ids=list(range(21)))
    assert data["valid_mask"].sum() == 3
    assert data["alignment_gt_mask"].sum() == 2


def test_sparse_depth_resize_and_pixel_center_intrinsics():
    depth = np.zeros((6, 8), np.float32)
    depth[2, 2] = 4.0
    K = np.array([[4, 0, 3.5], [0, 4, 2.5], [0, 0, 1]], np.float32)
    resized, scaled = resize_sparse_depth_and_intrinsic(depth, K, 4)
    assert resized.shape == (3, 4)
    assert np.count_nonzero(resized) == 1
    np.testing.assert_allclose(scaled, [[2, 0, 1.5], [0, 2, 1.0], [0, 0, 1]])


def test_metric_output_selection_supports_full_and_preselected_outputs():
    full = np.arange(5 * 2 * 3 * 3).reshape(5, 2, 3, 3)
    mask = np.ones((5, 2, 3), dtype=bool)
    selected = select_metric_dense_outputs(full, mask, mask, None, [0, 2, 4], 5)
    np.testing.assert_array_equal(selected[0], full[[0, 2, 4]])
    assert selected[1].shape[0] == 3
    already = select_metric_dense_outputs(full[[0, 2, 4]], mask[[0, 2, 4]], None, None, [0, 2, 4], 5)
    np.testing.assert_array_equal(already[0], full[[0, 2, 4]])
    with pytest.raises(ValueError, match="neither inference count"):
        select_metric_dense_outputs(full[:2], mask[:2], None, None, [0, 2, 4], 5)


def test_oxford_metric_defaults_are_outdoor_scale():
    assert get_dataset_eval_options("Oxford-Spires-S1-I10") == {
        "icp_threshold": 0.5,
        "voxel_size": 0.05,
        "eval_threshold": 4.0,
    }


def test_prediction_alignment_cutoff_is_horizon_only_and_state_independent():
    cfg = OmegaConf.create(
        {
            "pc_alignment_depth_max": None,
            "pc_alignment_depth_max_by_model": {"horizonstream": 80.0},
        }
    )
    base = cfg.pc_alignment_depth_max
    assert resolve_model_alignment_depth_max(cfg, "abot_recon", base) is None
    cfg.pc_alignment_depth_max = 80.0  # simulate active Horizon state
    assert resolve_model_alignment_depth_max(cfg, "horizonstream", base) == 80.0
    assert resolve_model_alignment_depth_max(cfg, "cut3r", base) is None


def test_lingbot_oxford_uses_official_378x518_prepare_before_area_budget():
    assert resolve_lingbot_prepare_width("Oxford-Spires-S1-I10") == 518
    prepared_wh = _prepare_width_aligned_hw(1440, 1080, 518)
    assert prepared_wh == (518, 378)
    assert _area_budget_hw(*prepared_wh, area_budget=255000, align=14) == prepared_wh


def test_lingbot_oxford_pil_lanczos_loader_and_runtime_shape_assertion(tmp_path):
    source = np.arange(1080 * 1440 * 3, dtype=np.uint8).reshape(1080, 1440, 3)
    image_path = tmp_path / "frame.png"
    Image.fromarray(source).save(image_path)
    loaded = _load_area_budget(
        [str(image_path)],
        area_budget=255000,
        align=14,
        device="cpu",
        prepare_width=518,
        prepare_interpolation="pil_lanczos",
    )
    assert tuple(loaded.shape) == (1, 1, 3, 378, 518)
    expected = np.asarray(
        Image.fromarray(source).resize((518, 378), Image.Resampling.LANCZOS)
    )
    actual = np.rint(loaded[0, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)
    cfg = OmegaConf.create({"lingbot_expected_input_hw": [378, 518]})
    _assert_expected_input_hw(loaded, cfg)
    with pytest.raises(RuntimeError, match="input shape mismatch"):
        _assert_expected_input_hw(loaded, OmegaConf.create({"lingbot_expected_input_hw": [434, 574]}))


def test_abot_recon_selects_dense_frames_but_keeps_full_camera_trajectory():
    points = torch.arange(1 * 5 * 2 * 3 * 3, dtype=torch.float32).reshape(1, 5, 2, 3, 3)
    poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 5, 1, 1)
    poses[0, :, 0, 3] = torch.arange(5, dtype=torch.float32)
    out = {"points": points, "local_points": points + 1000, "camera_poses": poses}
    indices = torch.tensor([0, 2, 4])
    world = _world_points_model_res(out, "points", frame_indices=indices)
    local = _local_points_model_res(out, frame_indices=indices)
    cameras = _poses_c2w_from_out(out)
    assert world.shape[0] == local.shape[0] == 3
    assert cameras.shape[0] == 5
    torch.testing.assert_close(world, points[0, indices])
    torch.testing.assert_close(local, points[0, indices] + 1000)
    assert cameras[:, 0, 3].tolist() == [0, 1, 2, 3, 4]
