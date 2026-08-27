import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

import interfaces.horizonstream as horizon_interface
from interfaces.horizonstream import (
    _load_and_preprocess as horizon_load,
    horizon_model_roi_on_gt,
    postprocess_horizon_points_to_gt,
)
from interfaces.lingbot_map import _load_area_budget
from interfaces.abot_recon import postprocess_fov_points_to_gt
from mv_recon.fov_preprocess import load_filelist_fov
from mv_recon.lingbot_protocol import resolve_pc_align_with_scale
from mv_recon.traj_vis import unpack_infer_mv_result


def test_tum_640x480_uses_model_specific_preprocessing_and_common_gt_grid(tmp_path):
    image_path = tmp_path / "tum.png"
    Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(image_path)
    filelist = [str(image_path)]

    # ABot-Recon: training-aligned width lock + center crop for 4:3 input.
    pi_images, pad = load_filelist_fov(
        filelist,
        target_h=280,
        target_w=504,
        pad_rgb=[0.485, 0.456, 0.406],
    )
    assert tuple(pi_images.shape) == (1, 1, 3, 280, 504)
    pi_world = torch.ones((1, 280, 504, 3))
    pi_points, pi_mask = postprocess_fov_points_to_gt(
        pi_world,
        pad,
        (480, 640),
        local=pi_world.clone(),
        nearest=True,
    )
    assert pi_points.shape == (1, 480, 640, 3)
    assert pi_mask.shape == (1, 480, 640)
    assert 0 < int(pi_mask.sum()) < 480 * 640

    _, _, pi_observed = postprocess_fov_points_to_gt(
        pi_world,
        pad,
        (480, 640),
        local=pi_world.clone(),
        nearest=True,
        return_observation_mask=True,
    )
    assert int(pi_observed.sum()) == int(pi_mask.sum())

    # LingBot official area-budget path preserves the complete FOV.
    lingbot_images = _load_area_budget(filelist, 255000, 14, "cpu", 0)
    assert tuple(lingbot_images.shape) == (1, 1, 3, 434, 574)

    # Horizon official loader preserves aspect ratio then center-crops; its
    # output must be placed only in the corresponding native-GT ROI.
    horizon_images = horizon_load(filelist, 518, 14, True, "cpu")
    model_hw = tuple(horizon_images.shape[-2:])
    rows, cols = horizon_model_roi_on_gt(
        model_hw, (480, 640), img_size=518, crop=True
    )
    horizon_depth = torch.ones((1, *model_hw))
    horizon_world = torch.ones((1, *model_hw, 3))
    horizon_points, horizon_mask = postprocess_horizon_points_to_gt(
        horizon_world,
        horizon_depth,
        (480, 640),
        img_size=518,
        crop=True,
        nearest=True,
    )
    assert horizon_points.shape == (1, 480, 640, 3)
    assert horizon_mask.shape == (1, 480, 640)
    assert int(horizon_mask.sum()) == (rows.stop - rows.start) * (cols.stop - cols.start)
    assert 0 < int(horizon_mask.sum()) <= 480 * 640

    _, _, horizon_observed = postprocess_horizon_points_to_gt(
        horizon_world,
        horizon_depth,
        (480, 640),
        img_size=518,
        crop=True,
        nearest=True,
        return_observation_mask=True,
    )
    assert np.array_equal(horizon_observed, horizon_mask)


def test_observation_mask_is_independent_of_depth_validity():
    world = torch.ones((1, 14, 14, 3))
    depth = torch.ones((1, 14, 14))
    depth[:, 3:5, 4:7] = 0.0
    _, pred_mask, observed = postprocess_horizon_points_to_gt(
        world,
        depth,
        (480, 640),
        img_size=518,
        crop=True,
        nearest=True,
        return_observation_mask=True,
    )
    assert int(observed.sum()) > int(pred_mask.sum())
    assert np.all(pred_mask <= observed)


def test_unpack_infer_result_preserves_observation_mask():
    points = np.zeros((1, 2, 3, 3), dtype=np.float32)
    poses = np.eye(4, dtype=np.float64)[None]
    pred_mask = np.ones((1, 2, 3), dtype=bool)
    observed = pred_mask.copy()
    observed[:, 0] = False
    out = unpack_infer_mv_result((points, poses, pred_mask, observed))
    assert len(out) == 6
    np.testing.assert_array_equal(out[2], pred_mask)
    np.testing.assert_array_equal(out[3], observed)
    assert out[4] is None
    assert out[5] is None


def test_tum_comparison_uses_per_model_scale_alignment():
    assert resolve_pc_align_with_scale("abot_recon", None) is True
    assert resolve_pc_align_with_scale("lingbot_map", None) is True
    assert resolve_pc_align_with_scale("horizonstream", None) is False
    assert resolve_pc_align_with_scale("horizonstream_se3", False) is False
    assert resolve_pc_align_with_scale("horizonstream_sim3", True) is True


def test_horizon_bfloat16_predictions_use_float32_geometry(monkeypatch):
    class FakeHorizon:
        img_size = 518
        patch_size = 14
        crop = True
        window_size = 10
        sliding_size = 21
        abs_pose_source = "online"

        def _autocast_settings(self, device):
            return True, torch.bfloat16

        def __call__(self, images, dense_output_indices=None):
            dtype = torch.bfloat16
            extrinsic = torch.eye(4, dtype=dtype)[None, :3]
            intrinsic = torch.eye(3, dtype=dtype)[None]
            depth = torch.ones((1, 14, 14), dtype=dtype)
            return {
                "extrinsic_w2c": extrinsic,
                "intrinsic": intrinsic,
                "depth": depth,
            }

    monkeypatch.setattr(
        horizon_interface,
        "_load_and_preprocess",
        lambda *args, **kwargs: torch.zeros((1, 1, 3, 14, 14)),
    )
    cfg = OmegaConf.create({"device": "cpu", "nearest_depth_to_gt": True})
    points, poses, mask, observed, alignment_mask, dense_aux = horizon_interface.infer_mv_pointclouds(
        ["unused.png"], FakeHorizon(), cfg, (480, 640)
    )
    assert points.dtype == np.float32
    assert points.shape == (1, 480, 640, 3)
    assert poses.shape == (1, 4, 4)
    assert mask.shape == (1, 480, 640)
    assert observed.shape == mask.shape
    assert np.all(mask <= observed)
    assert np.isfinite(points[mask]).all()
    assert alignment_mask is None
    assert dense_aux["pred_depth"].shape == mask.shape
    np.testing.assert_array_equal(dense_aux["pred_depth"] > 1e-4, mask)
