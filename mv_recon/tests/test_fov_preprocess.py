"""Unit tests for reconstruction FOV preprocessing and padding removal.

Run from the evaluation repository root:
    python mv_recon/tests/test_fov_preprocess.py
    python -m unittest mv_recon.tests.test_fov_preprocess -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mv_recon.fov_preprocess import (  # noqa: E402
    DEFAULT_PAD_RGB,
    FovPadInfo,
    load_filelist_fov,
    resize_width_crop_or_pad_mean,
    strip_vertical_pad_hwc,
    strip_vertical_pad_map,
)
from mv_recon.lingbot_protocol import (  # noqa: E402
    colored_aligned_clouds_for_ply,
    save_xyzrgb_ply,
    subsample_points,
)
from mv_recon.pc_infer_utils import (  # noqa: E402
    pointmap_resize_mode,
    pred_mask_from_depth,
    resize_map_to_hw,
)


def _solid_rgb(h: int, w: int, color=(0.2, 0.4, 0.6)) -> torch.Tensor:
    t = torch.zeros(3, h, w)
    for c, v in enumerate(color):
        t[c] = float(v)
    return t


def _training_pad_geometry(src_h: int, src_w: int, target_h: int = 280, target_w: int = 504):
    """Mirror ``BaseDataset._resize_width_crop_or_pad_mean`` geometry (no pixels)."""
    scale = float(target_w) / float(src_w)
    resized_h = max(1, int(round(float(src_h) * scale)))
    if resized_h > target_h:
        return 0, 0, target_h, False
    if resized_h < target_h:
        pad_top = int((target_h - resized_h) // 2)
        pad_bottom = int(target_h - resized_h - pad_top)
        return pad_top, pad_bottom, resized_h, True
    return 0, 0, target_h, False


class TestFovResize(unittest.TestCase):
    def test_wide_short_gets_vertical_mean_pad(self):
        # Landscape short: after W→504, H is small → pad to 280.
        src = _solid_rgb(100, 800, color=(0.1, 0.2, 0.3))
        fr = resize_width_crop_or_pad_mean(src, target_h=280, target_w=504)
        self.assertEqual(tuple(fr.image.shape), (3, 280, 504))
        self.assertTrue(fr.pad.has_pad)
        self.assertGreater(fr.pad.pad_top, 0)
        self.assertGreater(fr.pad.pad_bottom, 0)
        self.assertEqual(fr.pad.pad_top + fr.pad.content_h + fr.pad.pad_bottom, 280)
        # Pad band ≈ ImageNet mean
        top_band = fr.image[:, : fr.pad.pad_top, :]
        for c, v in enumerate(DEFAULT_PAD_RGB):
            self.assertTrue(torch.allclose(top_band[c], torch.tensor(v), atol=1e-5))
        # Content band keeps source color
        mid = fr.image[:, fr.pad.pad_top : fr.pad.pad_top + fr.pad.content_h, :]
        self.assertTrue(torch.allclose(mid[0], torch.tensor(0.1), atol=1e-3))

    def test_tall_gets_center_crop_no_pad(self):
        # After W→504, H >> 280 → center crop, no pad.
        src = _solid_rgb(900, 600, color=(0.7, 0.1, 0.1))
        fr = resize_width_crop_or_pad_mean(src, target_h=280, target_w=504)
        self.assertEqual(tuple(fr.image.shape), (3, 280, 504))
        self.assertFalse(fr.pad.has_pad)
        self.assertTrue(fr.pad.has_crop)
        self.assertEqual(fr.pad.resized_h, 756)
        self.assertEqual(fr.pad.crop_top + 280 + fr.pad.crop_bottom, 756)
        self.assertEqual(fr.pad.pad_top, 0)
        self.assertEqual(fr.pad.pad_bottom, 0)
        self.assertEqual(fr.pad.content_h, 280)

    def test_exact_aspect_no_pad_no_crop_change(self):
        # 504×280 already: identity geometry.
        src = _solid_rgb(280, 504, color=(0.5, 0.5, 0.5))
        fr = resize_width_crop_or_pad_mean(src, target_h=280, target_w=504)
        self.assertEqual(tuple(fr.image.shape), (3, 280, 504))
        self.assertFalse(fr.pad.has_pad)

    def test_matches_training_width_lock_formula(self):
        # W=640 H=480 (7S-like) → scale=504/640, resized_h=round(480*scale)=378 → crop.
        src_h, src_w = 480, 640
        scale = 504.0 / src_w
        expected_h = max(1, int(round(src_h * scale)))
        self.assertEqual(expected_h, 378)
        src = _solid_rgb(src_h, src_w)
        fr = resize_width_crop_or_pad_mean(src)
        self.assertFalse(fr.pad.has_pad)
        self.assertEqual(fr.pad.content_h, 280)
        self.assertEqual(fr.pad.crop_top, 49)
        self.assertEqual(fr.pad.crop_bottom, 49)
        self.assertEqual(fr.pad.gt_content_row_slice(480), slice(62, 418))

    def test_kitti_like_pad_content_height(self):
        # 1242×375 → scale=504/1242, resized_h≈152 → pad.
        src_h, src_w = 375, 1242
        scale = 504.0 / src_w
        expected_content = max(1, int(round(src_h * scale)))
        src = _solid_rgb(src_h, src_w)
        fr = resize_width_crop_or_pad_mean(src)
        self.assertEqual(fr.pad.content_h, expected_content)
        self.assertEqual(fr.pad.pad_top + fr.pad.content_h + fr.pad.pad_bottom, 280)

    def test_pad_geometry_matches_training_for_common_sensors(self):
        cases = [
            (480, 640),   # 7Scenes / TUM → crop
            (375, 1242),  # KITTI-like → pad
            (1080, 1920),
            (768, 1024),
            (280, 504),   # already target
        ]
        for src_h, src_w in cases:
            with self.subTest(hw=(src_h, src_w)):
                pt, pb, ch, has_pad = _training_pad_geometry(src_h, src_w)
                fr = resize_width_crop_or_pad_mean(_solid_rgb(src_h, src_w))
                self.assertEqual(fr.pad.pad_top, pt)
                self.assertEqual(fr.pad.pad_bottom, pb)
                self.assertEqual(fr.pad.content_h, ch)
                self.assertEqual(fr.pad.has_pad, has_pad)
                self.assertEqual(tuple(fr.image.shape), (3, 280, 504))

    def test_bicubic_boundary_values_are_not_clamped(self):
        image = np.zeros((376, 1241, 3), dtype=np.uint8)
        image[:, 600:] = 255
        frame = resize_width_crop_or_pad_mean(image)
        self.assertLess(float(frame.image.min()), 0.0)
        self.assertGreater(float(frame.image.max()), 1.0)


class TestStripPad(unittest.TestCase):
    def test_strip_hwc_points(self):
        pad = FovPadInfo(pad_top=40, pad_bottom=40, target_h=280, target_w=504, content_h=200)
        pts = torch.zeros(2, 280, 504, 3)
        pts[:, 40:240, :, 2] = 1.5  # valid depth in content
        pts[:, :40, :, 2] = 99.0  # pad garbage
        out = strip_vertical_pad_hwc(pts, pad)
        self.assertEqual(tuple(out.shape), (2, 200, 504, 3))
        self.assertTrue(torch.all(out[..., 2] == 1.5))

    def test_strip_noop_when_no_pad(self):
        pad = FovPadInfo(pad_top=0, pad_bottom=0, target_h=280, target_w=504, content_h=280)
        pts = torch.randn(3, 280, 504, 3)
        out = strip_vertical_pad_hwc(pts, pad)
        self.assertTrue(torch.equal(out, pts))

    def test_strip_nchw_depth(self):
        pad = FovPadInfo(pad_top=10, pad_bottom=20, target_h=280, target_w=504, content_h=250)
        depth = torch.zeros(4, 1, 280, 504)
        depth[:, :, 10:260, :] = 2.0
        out = strip_vertical_pad_map(depth, pad, spatial_dims=(-2, -1))
        self.assertEqual(tuple(out.shape), (4, 1, 250, 504))
        self.assertTrue(torch.all(out == 2.0))

    def test_strip_rejects_wrong_height(self):
        pad = FovPadInfo(pad_top=1, pad_bottom=1, target_h=280, target_w=504, content_h=278)
        pts = torch.zeros(1, 100, 504, 3)
        with self.assertRaises(ValueError):
            strip_vertical_pad_hwc(pts, pad)


class TestPcPipelineHelpers(unittest.TestCase):
    def test_pointmap_resize_mode_is_explicit_and_backward_compatible(self):
        from types import SimpleNamespace

        self.assertEqual(pointmap_resize_mode(SimpleNamespace(nearest_depth_to_gt=True)), "nearest")
        self.assertEqual(pointmap_resize_mode(SimpleNamespace(nearest_depth_to_gt=False)), "bilinear")
        self.assertEqual(
            pointmap_resize_mode(SimpleNamespace(pointmap_resize_mode="bilinear")),
            "bilinear",
        )
        with self.assertRaises(ValueError):
            pointmap_resize_mode(SimpleNamespace(pointmap_resize_mode="bicubic"))

    def test_strip_then_nearest_to_gt_indoor(self):
        """Simulate the reconstruction path: strip padding, then resize."""
        pad = FovPadInfo(pad_top=50, pad_bottom=50, target_h=280, target_w=504, content_h=180)
        world = torch.zeros(2, 280, 504, 3)
        # Content: constant depth 3m, varying x along height
        ys = torch.linspace(0, 1, 180).view(1, 180, 1).expand(2, 180, 504)
        world[:, 50:230, :, 0] = ys
        world[:, 50:230, :, 2] = 3.0
        world[:, :50, :, 2] = 0.0  # pad
        world[:, 230:, :, 2] = 0.0

        stripped = strip_vertical_pad_hwc(world, pad)
        self.assertEqual(stripped.shape[1], 180)
        depth = stripped[..., 2]
        depth_gt = resize_map_to_hw(depth, (480, 640), nearest=True)
        self.assertEqual(tuple(depth_gt.shape), (2, 480, 640))
        mask = pred_mask_from_depth(depth_gt)
        self.assertTrue(mask.all())
        self.assertTrue(torch.allclose(depth_gt, torch.tensor(3.0), atol=1e-5))

        world_nchw = stripped.permute(0, 3, 1, 2)
        world_gt = resize_map_to_hw(world_nchw, (480, 640), nearest=True)
        self.assertEqual(tuple(world_gt.shape), (2, 3, 480, 640))

    def test_pad_depth_not_in_mask_before_strip(self):
        pad = FovPadInfo(pad_top=30, pad_bottom=30, target_h=280, target_w=504, content_h=220)
        depth = torch.zeros(1, 280, 504)
        depth[:, 30:250, :] = 1.0
        # If we forgot to strip, pad rows have depth 0 → mask False there
        full_mask = pred_mask_from_depth(depth)
        self.assertFalse(bool(full_mask[0, :30].any()))
        stripped = strip_vertical_pad_map(depth, pad)
        self.assertTrue(pred_mask_from_depth(stripped).all())


class TestPostprocessFovPointsToGt(unittest.TestCase):
    def test_pad_garbage_never_reaches_gt_grid(self):
        from interfaces.abot_recon import postprocess_fov_points_to_gt

        pad = FovPadInfo(pad_top=40, pad_bottom=40, target_h=280, target_w=504, content_h=200)
        world = torch.full((1, 280, 504, 3), 999.0)  # pad sentinel
        local = torch.zeros(1, 280, 504, 3)
        # Content only
        world[:, 40:240, :, :] = 1.0
        local[:, 40:240, :, 2] = 2.5
        # Pad local Z = 0 (invalid) but world pad = 999 — must not leak after strip
        local[:, :40, :, 2] = 0.0
        local[:, 240:, :, 2] = 0.0

        pts, mask = postprocess_fov_points_to_gt(
            world, pad, (480, 640), local=local, nearest=True
        )
        self.assertEqual(pts.shape, (1, 480, 640, 3))
        self.assertEqual(mask.shape, (1, 480, 640))
        self.assertTrue(mask.all())
        self.assertTrue(np.allclose(pts, 1.0, atol=1e-5))
        self.assertFalse(np.any(np.isclose(pts, 999.0)))

    def test_mask_uses_local_z_not_world_z(self):
        from interfaces.abot_recon import postprocess_fov_points_to_gt

        # No pad; world Z huge but local Z zero on half → those pixels invalid.
        pad = FovPadInfo(pad_top=0, pad_bottom=0, target_h=280, target_w=504, content_h=280)
        world = torch.ones(1, 280, 504, 3) * 10.0
        local = torch.zeros(1, 280, 504, 3)
        local[:, :, :252, 2] = 1.0  # left half valid
        pts, mask = postprocess_fov_points_to_gt(
            world, pad, (280, 504), local=local, nearest=True
        )
        self.assertTrue(mask[0, :, :252].all())
        self.assertFalse(mask[0, :, 252:].any())

    def test_alignment_depth_max_uses_local_z_only(self):
        from interfaces.abot_recon import postprocess_fov_points_to_gt

        pad = FovPadInfo(
            pad_top=0,
            pad_bottom=0,
            target_h=2,
            target_w=2,
            content_h=2,
        )
        # World Z is deliberately unrelated to local camera depth.
        world = torch.zeros(1, 2, 2, 3)
        world[..., 2] = 1000.0
        local = torch.zeros_like(world)
        local[..., 2] = torch.tensor([[2.0, 50.0], [3.0, 40.0]])

        _, pred_mask, observed, alignment_mask = postprocess_fov_points_to_gt(
            world,
            pad,
            (2, 2),
            local=local,
            nearest=True,
            return_observation_mask=True,
            alignment_depth_max=40.0,
            return_alignment_mask=True,
        )

        self.assertTrue(pred_mask.all())
        self.assertTrue(observed.all())
        np.testing.assert_array_equal(
            alignment_mask,
            np.array([[[True, False], [True, True]]]),
        )

    def test_7scenes_crop_maps_only_to_matching_gt_roi(self):
        from interfaces.abot_recon import postprocess_fov_points_to_gt

        # 7S FOV: crop only → pad zeros; model out is full 280×504 content.
        fr = resize_width_crop_or_pad_mean(_solid_rgb(480, 640))
        self.assertFalse(fr.pad.has_pad)
        world = torch.randn(2, 280, 504, 3)
        local = torch.ones(2, 280, 504, 3)
        local[..., 2] = 1.2
        pts, mask = postprocess_fov_points_to_gt(
            world, fr.pad, (480, 640), local=local, nearest=True
        )
        self.assertEqual(pts.shape, (2, 480, 640, 3))
        self.assertFalse(mask[:, :62].any())
        self.assertTrue(mask[:, 62:418].all())
        self.assertFalse(mask[:, 418:].any())
        self.assertTrue(np.allclose(pts[:, :62], 0.0))
        self.assertTrue(np.allclose(pts[:, 418:], 0.0))

    def test_crop_row_coordinates_land_in_correct_gt_rows(self):
        from interfaces.abot_recon import postprocess_fov_points_to_gt

        fr = resize_width_crop_or_pad_mean(_solid_rgb(480, 640))
        rows = torch.arange(280, dtype=torch.float32).view(1, 280, 1)
        world = torch.zeros(1, 280, 504, 3)
        world[..., 0] = rows
        local = torch.zeros_like(world)
        local[..., 2] = 1.0
        pts, mask = postprocess_fov_points_to_gt(
            world, fr.pad, (480, 640), local=local, nearest=True
        )
        self.assertTrue(mask[0, 62:418].all())
        self.assertEqual(float(pts[0, 62, 0, 0]), 0.0)
        self.assertEqual(float(pts[0, 417, 0, 0]), 279.0)


class TestHorizonCropRoi(unittest.TestCase):
    def test_predicted_intrinsics_scale_anisotropically_to_roi(self):
        from interfaces.horizonstream import scale_intrinsics_to_hw

        k = torch.tensor(
            [[[500.0, 0.0, 259.0], [0.0, 480.0, 189.0], [0.0, 0.0, 1.0]]]
        )
        scaled = scale_intrinsics_to_hw(k, (378, 518), (468, 640))
        self.assertAlmostEqual(float(scaled[0, 0, 0]), 500.0 * 640.0 / 518.0, delta=1e-4)
        self.assertAlmostEqual(float(scaled[0, 0, 2]), 259.0 * 640.0 / 518.0, delta=1e-4)
        self.assertAlmostEqual(float(scaled[0, 1, 1]), 480.0 * 468.0 / 378.0, delta=1e-4)
        self.assertAlmostEqual(float(scaled[0, 1, 2]), 189.0 * 468.0 / 378.0, delta=1e-4)
        self.assertEqual(float(scaled[0, 2, 2]), 1.0)

    def test_depth_resize_then_unproject_preserves_crop_mask(self):
        from interfaces.horizonstream import postprocess_horizon_depth_to_gt

        depth = torch.full((1, 378, 518), 2.0)
        intrinsic = torch.tensor(
            [[[500.0, 0.0, 259.0], [0.0, 500.0, 189.0], [0.0, 0.0, 1.0]]]
        )
        w2c = torch.eye(4)[None, :3]
        pts, mask, observed = postprocess_horizon_depth_to_gt(
            depth,
            intrinsic,
            w2c,
            (480, 640),
            img_size=518,
            crop=True,
            nearest=True,
            return_observation_mask=True,
        )
        self.assertEqual(pts.shape, (1, 480, 640, 3))
        self.assertFalse(mask[:, :6].any())
        self.assertTrue(mask[:, 6:474].all())
        self.assertFalse(mask[:, 474:].any())
        self.assertTrue(np.array_equal(mask, observed))
        self.assertTrue(np.allclose(pts[:, 6:474, :, 2], 2.0, atol=1e-5))

    def test_depth_max_only_changes_alignment_mask(self):
        from interfaces.horizonstream import postprocess_horizon_depth_to_gt

        depth = torch.tensor([[[2.0, 50.0], [3.0, 40.0]]])
        intrinsic = torch.eye(3)[None]
        w2c = torch.eye(4)[None, :3]
        _, pred_mask, observed, alignment_mask = postprocess_horizon_depth_to_gt(
            depth,
            intrinsic,
            w2c,
            (2, 2),
            img_size=2,
            crop=False,
            nearest=True,
            return_observation_mask=True,
            alignment_depth_max=40.0,
            return_alignment_mask=True,
        )

        self.assertTrue(pred_mask.all())
        self.assertTrue(observed.all())
        np.testing.assert_array_equal(
            alignment_mask,
            np.array([[[True, False], [True, True]]]),
        )

    def test_7scenes_official_crop_maps_to_center_gt_rows(self):
        from interfaces.horizonstream import horizon_model_roi_on_gt

        # Official loader: 640x480 -> 518x388 -> center crop 518x378.
        rows, cols = horizon_model_roi_on_gt(
            (378, 518), (480, 640), img_size=518, crop=True
        )
        self.assertEqual(rows, slice(6, 474))
        self.assertEqual(cols, slice(0, 640))

    def test_horizon_crop_never_fills_unobserved_gt_rows(self):
        from interfaces.horizonstream import postprocess_horizon_points_to_gt

        world = torch.ones(2, 378, 518, 3)
        depth = torch.full((2, 378, 518), 2.0)
        pts, mask = postprocess_horizon_points_to_gt(
            world,
            depth,
            (480, 640),
            img_size=518,
            crop=True,
            nearest=True,
        )
        self.assertEqual(pts.shape, (2, 480, 640, 3))
        self.assertFalse(mask[:, :6].any())
        self.assertTrue(mask[:, 6:474].all())
        self.assertFalse(mask[:, 474:].any())
        self.assertTrue(np.allclose(pts[:, :6], 0.0))
        self.assertTrue(np.allclose(pts[:, 474:], 0.0))

    def test_no_crop_maps_to_full_gt(self):
        from interfaces.horizonstream import horizon_model_roi_on_gt

        rows, cols = horizon_model_roi_on_gt(
            (378, 518), (480, 640), img_size=518, crop=False
        )
        self.assertEqual(rows, slice(0, 480))
        self.assertEqual(cols, slice(0, 640))


class TestColoredPly(unittest.TestCase):
    def test_colored_aligned_clouds_and_rgb_ply(self):
        s, h, w = 2, 8, 10
        pred = np.random.randn(s, h, w, 3).astype(np.float32)
        gt = pred + 0.01
        gt_mask = np.ones((s, h, w), dtype=bool)
        gt_mask[0, 0, 0] = False
        imgs = np.zeros((s, 3, h, w), dtype=np.float32)
        imgs[:, 0] = 0.8
        imgs[:, 1] = 0.2
        imgs[:, 2] = 0.1
        T = np.eye(4)
        px, pr, gx, gr = colored_aligned_clouds_for_ply(
            pred, gt, gt_mask, imgs, T, T, pred_mask=gt_mask
        )
        self.assertEqual(px.shape, pr.shape)
        self.assertTrue(np.allclose(pr[:, 0], 0.8, atol=1e-5))
        ps, pc = subsample_points(px, 50, seed=0, colors=pr)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "c.ply")
            save_xyzrgb_ply(path, ps, pc)
            self.assertGreater(os.path.getsize(path), 0)
            # PLY header must declare RGB properties
            with open(path, "rb") as f:
                head = f.read(512).decode("latin1", errors="ignore")
            self.assertIn("property uchar red", head)
            self.assertIn("property uchar green", head)
            self.assertIn("property uchar blue", head)


class TestLoadFilelistFov(unittest.TestCase):
    def test_interface_preserves_caller_file_order(self):
        from types import SimpleNamespace
        from interfaces.abot_recon import _load_filelist_fov_tensor

        with tempfile.TemporaryDirectory() as td:
            z_path = os.path.join(td, "z_last_by_name.png")
            a_path = os.path.join(td, "a_first_by_name.png")
            Image.fromarray(np.full((280, 504, 3), (220, 0, 0), dtype=np.uint8)).save(z_path)
            Image.fromarray(np.full((280, 504, 3), (10, 0, 0), dtype=np.uint8)).save(a_path)
            model = SimpleNamespace(height=280, width=504, fov_pad_rgb=DEFAULT_PAD_RGB)
            batch, _ = _load_filelist_fov_tensor([z_path, a_path], model, "cpu")
            self.assertGreater(float(batch[0, 0, 0].mean()), 0.8)
            self.assertLess(float(batch[0, 1, 0].mean()), 0.1)

    def test_load_from_pngs(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(3):
                # Wide short → will pad
                arr = np.full((120, 640, 3), 40, dtype=np.uint8)
                p = os.path.join(td, f"frame_{i:02d}.png")
                Image.fromarray(arr).save(p)
                paths.append(p)
            batch, pad = load_filelist_fov(paths, target_h=280, target_w=504)
            self.assertEqual(tuple(batch.shape), (1, 3, 3, 280, 504))
            self.assertTrue(pad.has_pad)


class TestWorldFromLocalCompose(unittest.TestCase):
    def test_explicit_source_selects_direct_points_or_local_pose(self):
        from interfaces.abot_recon import _world_points_model_res

        direct = torch.full((1, 1, 2, 2, 3), 9.0)
        local = torch.zeros(1, 1, 2, 2, 3)
        local[..., 2] = 2.0
        poses = torch.eye(4).view(1, 1, 4, 4).clone()
        poses[0, 0, 0, 3] = 3.0
        out = {"points": direct, "local_points": local, "camera_poses": poses}

        world_direct = _world_points_model_res(out, source="points")
        world_local = _world_points_model_res(out, source="local_pose")

        self.assertTrue(torch.all(world_direct == 9.0))
        self.assertTrue(torch.allclose(world_local[..., 0], torch.tensor(3.0)))
        self.assertTrue(torch.allclose(world_local[..., 2], torch.tensor(2.0)))

    def test_identity_c2w_matches_local(self):
        """Sanity for interface helper: identity pose → world == local."""
        from interfaces.abot_recon import _world_points_model_res

        s, h, w = 2, 8, 10
        local = torch.randn(1, s, h, w, 3)
        poses = torch.eye(4).view(1, 1, 4, 4).expand(1, s, 4, 4).clone()
        world = _world_points_model_res({"local_points": local, "camera_poses": poses})
        self.assertTrue(torch.allclose(world, local[0], atol=1e-5))

    def test_translation_c2w(self):
        from interfaces.abot_recon import _world_points_model_res

        local = torch.zeros(1, 1, 2, 2, 3)
        local[..., 2] = 1.0
        poses = torch.eye(4).view(1, 1, 4, 4).clone()
        poses[0, 0, 0, 3] = 5.0  # tx
        world = _world_points_model_res({"local_points": local, "camera_poses": poses})
        self.assertTrue(torch.allclose(world[..., 0], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(world[..., 2], torch.tensor(1.0)))




if __name__ == "__main__":
    unittest.main()
