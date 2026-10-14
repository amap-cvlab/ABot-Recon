"""CPU-only regression tests for the standalone published-output renderer.

Run at repository root with ``python -m unittest discover -s tests -p 'test_visualization.py' -v``. Synthetic
public-format results live only in temporary directories; no model weights,
dataset, display server, network, or GPU are needed.
"""
from __future__ import annotations

import gc
import ast
import hashlib
from importlib import import_module
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np
from PIL import Image
import torch

# Published layout is tests/ alongside visualization/.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION = REPOSITORY_ROOT / "visualization"
sys.path.insert(0, str(VISUALIZATION))
import render
import generate_sky_masks


def dependency_module(name, **attributes):
    """A fake importable dependency, including metadata used by find_spec()."""
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    module.__dict__.update(attributes)
    return module


def arguments(root=None, **overrides):
    """Use the real CLI defaults so tests catch changes to public options."""
    root = Path(root or "/unused")
    result = render.parser().parse_args([
        "--result-dir", str(root), "--output-dir", str(root / "rendered"),
        "--outputs", "trajectory",
    ])
    result.confidence = 0.15
    result.depth_edge = 0
    result.max_depth = 0
    result.device = "cpu"
    # Geometry-only tests intentionally require no generated sky masks.
    # OfflineSkyMaskTests checks the actual default-on CLI separately.
    result.sky = False
    for name, value in overrides.items():
        setattr(result, name, value)
    return result


def point_grid(height=4, width=6, depth=2.0):
    """An exact pinhole point map, with independent fx/fy and centered cx/cy."""
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return np.stack(((xx - (width - 1) / 2) / width * depth,
                     (yy - (height - 1) / 2) / height * depth,
                     np.full_like(xx, depth)), axis=-1)


class ArrayReader:
    """In-memory reader used to isolate filtering from source-file validation."""
    def __init__(self, points, confidence=None, source_hw=None, pose=None):
        self._points = points.astype(np.float32)
        self.indices = np.array([0], dtype=np.int64)
        self.poses = np.eye(4)[None] if pose is None else pose[None]
        self.confidence = None if confidence is None else torch.from_numpy(confidence[None].astype(np.float32))
        self.colors = torch.full(points.shape, 150, dtype=torch.uint8)[None]
        self.h, self.w = points.shape[:2]
        self.paths = None if source_hw is None else [Path("unused.png")]
        self._rgb = None if source_hw is None else np.full((*source_hw, 3), 200, dtype=np.uint8)
        self.geometry = None if source_hw is None else render.source_geometry(source_hw, (self.h, self.w))

    def local_points(self, row):
        assert row == 0
        return self._points.copy()

    def rgb(self, pose_index):
        assert pose_index == 0
        return self._rgb.copy()


class ResultFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.indices = [0, 2, 4]
        self.local = np.stack([point_grid(depth=2 + j) for j in range(3)])
        self.final = np.repeat(np.eye(4)[None], 5, axis=0).astype(np.float32)
        self.raw = self.final.copy()
        # Nontrivial rotations expose accidental row/column convention errors.
        for i in range(5):
            theta = i * 0.1
            c, s = np.cos(theta), np.sin(theta)
            self.final[i, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            self.final[i, :3, 3] = [10 + i, i * 0.3, -0.2 * i]
            self.raw[i, :3, 3] = [i, 0, 0]
        self.metadata = {"frames": 5, "dense_output_indices": self.indices,
                         "confidence_threshold": 0.0, "loop_closure": True}
        np.save(self.root / "camera_poses.npy", self.final)
        np.save(self.root / "camera_poses_loop.npy", self.final)
        np.save(self.root / "camera_poses_noloop.npy", self.raw)
        torch.save(torch.from_numpy(self.local), self.root / "local_points.pt")
        torch.save(torch.full(self.local.shape, 100, dtype=torch.uint8), self.root / "colors.pt")
        torch.save(torch.full(self.local.shape[:-1], 0.8), self.root / "confidence.pt")
        self.save_metadata()

    def tearDown(self):
        # Mock call/return cycles can retain mmap tensors after a test returns.
        # Release them before NFS turns open, unlinked files into .nfs* entries.
        gc.collect()
        self.tmp.cleanup()

    def save_metadata(self):
        (self.root / "metadata.json").write_text(json.dumps(self.metadata))

    def reader(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return render.ReconstructionReader(arguments(self.root, **kwargs))


class ReaderTests(ResultFixture, unittest.TestCase):
    def test_sparse_dense_indices_are_preserved(self):
        reader = self.reader()
        np.testing.assert_array_equal(reader.indices, [0, 2, 4])
        self.assertEqual(reader.m, 3)
        self.assertEqual(len(reader.poses), 5)
        np.testing.assert_allclose(reader.local_points(2), self.local[2])

    def test_local_geometry_uses_mapped_final_pose_exactly_once(self):
        reader = self.reader()
        world, colors, stats = render.prepare_cloud(reader, 1, arguments(), 1)
        pose = self.final[2]  # Dense row 1 corresponds to pose 2, not pose 1.
        expected = self.local[1].reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        np.testing.assert_allclose(world, expected, atol=1e-6)
        self.assertEqual(stats["pose_index"], 2)
        self.assertEqual(colors.shape, world.shape)

    def test_world_input_undoes_final_pose_before_raw_pose_selection(self):
        world = np.stack([p @ self.final[i, :3, :3].T + self.final[i, :3, 3]
                          for p, i in zip(self.local, self.indices)])
        (self.root / "local_points.pt").unlink()
        torch.save(torch.from_numpy(world), self.root / "world_points.pt")
        reader = self.reader(poses="noloop")
        self.assertFalse(reader.local)
        np.testing.assert_allclose(reader.local_points(1), self.local[1], atol=1e-6)
        actual, _, _ = render.prepare_cloud(reader, 1, arguments(), 1)
        expected = self.local[1].reshape(-1, 3) + self.raw[2, :3, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_world_input_final_mode_does_not_double_apply_loop(self):
        world = np.stack([p @ self.final[i, :3, :3].T + self.final[i, :3, 3]
                          for p, i in zip(self.local, self.indices)])
        (self.root / "local_points.pt").unlink()
        torch.save(torch.from_numpy(world), self.root / "world_points.pt")
        actual, _, _ = render.prepare_cloud(self.reader(), 2, arguments(), 1)
        np.testing.assert_allclose(actual, world[2].reshape(-1, 3), atol=1e-6)

    def test_invalid_dense_mappings_are_rejected(self):
        for mapping in ([0, 0, 4], [4, 2, 0], [0, 2], [0, 2, 5], [-1, 2, 4], [0.0, 2.0, 4.0], None):
            with self.subTest(mapping=mapping):
                self.metadata["dense_output_indices"] = mapping
                self.save_metadata()
                with self.assertRaises(ValueError):
                    self.reader()

    def test_missing_mapping_allowed_only_when_every_pose_has_dense_output(self):
        self.metadata["frames"] = 3
        self.metadata["dense_output_indices"] = None
        self.save_metadata()
        np.save(self.root / "camera_poses.npy", self.final[:3])
        reader = self.reader()
        np.testing.assert_array_equal(reader.indices, [0, 1, 2])

    def test_reflection_and_nonrigid_pose_are_rejected(self):
        for value in (-1.0, 1.2):
            with self.subTest(value=value):
                bad = self.final.copy()
                bad[0, 0, 0] = value
                np.save(self.root / "camera_poses.npy", bad)
                with self.assertRaises(ValueError):
                    self.reader()

    def test_missing_confidence_requires_explicit_disable(self):
        (self.root / "confidence.pt").unlink()
        with self.assertRaisesRegex(ValueError, "confidence.pt"):
            self.reader()
        self.assertIsNone(self.reader(confidence=0).confidence)

    def test_irreversible_inference_threshold_is_reported(self):
        self.metadata["confidence_threshold"] = 0.5
        self.save_metadata()
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            render.ReconstructionReader(arguments(self.root))
        self.assertTrue(any("cannot restore" in str(item.message) for item in messages))

    def test_source_list_count_mismatch_is_rejected_before_preprocessor_import(self):
        file = self.root / "frames.json"
        file.write_text("[]")
        with self.assertRaisesRegex(ValueError, "exactly N"):
            self.reader(frame_list=file)

    def test_end_to_end_frame_schedule_transition_and_aligned_pngs(self):
        """Exercise the real orchestration while replacing only FFmpeg I/O."""
        local = np.repeat(self.local[:1], 5, axis=0)
        torch.save(torch.from_numpy(local), self.root / "local_points.pt")
        torch.save(torch.full(local.shape, 150, dtype=torch.uint8), self.root / "colors.pt")
        torch.save(torch.ones(local.shape[:-1]), self.root / "confidence.pt")
        self.metadata["dense_output_indices"] = None
        self.save_metadata()
        written = []

        class MemoryVideoWriter:
            def __init__(self, path, width, height, a):
                self.frames = []
                self.shape = (height, width, 3)
                self.closed = False
                written.append(self)

            def write(self, rgb):
                if rgb.shape != self.shape:
                    raise AssertionError("Video dimensions unexpectedly changed")
                self.frames.append(rgb.copy())

            def close(self):
                self.closed = True

        a = arguments(self.root, width=64, height=48, point_size=1,
                      view_step=2, cloud_step=1, ground_frames=600,
                      transition_frames=2, bird_cloud_step=1, bird_pixel_step=1,
                      frustum_step=0, prune_every=0,
                      outputs=["video", "trajectory", "pointcloud"])
        with warnings.catch_warnings(), mock.patch.object(render, "VideoWriter", MemoryVideoWriter), mock.patch.object(render.shutil, "which", return_value="/mock/ffmpeg"):
            warnings.simplefilter("ignore", UserWarning)
            render.render(a)
        manifest = json.loads((a.output_dir / "render_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["ground_view_indices"], [0, 2, 4])
        self.assertEqual(manifest["ground_frame_count"], 3)
        self.assertEqual(manifest["transition_frame_count"], 2)
        self.assertEqual([row["added_dense_pose_indices"] for row in manifest["ground_frame_log"]], [[0], [1, 2], [3, 4]])
        self.assertEqual(len(written), 1)
        self.assertEqual(len(written[0].frames), 5)
        self.assertTrue(written[0].closed)
        with Image.open(a.output_dir / "last_frame.png") as im:
            np.testing.assert_array_equal(np.asarray(im), written[0].frames[-1])
        # The complete scene is shown by the last ground frame in this fixture,
        # so transparent transition endpoint and global trajectory must match.
        with Image.open(a.output_dir / "last_frame_transparent.png") as im:
            partial = np.asarray(im)
        with Image.open(a.output_dir / "birdseye_trajectory.png") as im:
            trajectory = np.asarray(im)
        np.testing.assert_array_equal(partial, trajectory)
        with Image.open(a.output_dir / "birdseye_pointcloud.png") as im:
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.size, (64, 48))

    def test_video_without_transition_never_loads_global_bird_cloud(self):
        """A short ground-only preview must not read the full sequence cloud."""
        a = arguments(self.root, width=64, height=48, point_size=1,
                      view_step=1, cloud_step=1, ground_frames=2,
                      transition_frames=0, bird_cloud_step=1,
                      frustum_step=0, prune_every=0, outputs=["video"])
        frames = []
        writer = mock.Mock()
        writer.write.side_effect = lambda rgb: frames.append(rgb.copy())
        with warnings.catch_warnings(), \
                mock.patch.object(render, "VideoWriter", return_value=writer), \
                mock.patch.object(render.shutil, "which", return_value="/mock/ffmpeg"), \
                mock.patch.object(render, "prepare_cloud", wraps=render.prepare_cloud) as prepare, \
                mock.patch.object(render, "bird_camera", side_effect=AssertionError("Bird camera is unnecessary for ground-only video")):
            warnings.simplefilter("ignore", UserWarning)
            render.render(a)
        # Dense rows map to [0,2,4], but the requested video shows poses [0,1].
        # Only row 0 is needed; a global bounds/image pass would add extra calls.
        try:
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(prepare.call_args.args[1], 0)
        finally:
            # Spy call records retain the reader, hence its torch mmap tensors.
            # Close those mappings before TemporaryDirectory removes input
            # files on NFS; otherwise unlinked open files become .nfs* entries.
            prepare.reset_mock()
            gc.collect()
        self.assertEqual(len(frames), 2)
        writer.close.assert_called_once()
        manifest = json.loads((a.output_dir / "render_manifest.json").read_text())
        self.assertEqual(manifest["ground_view_indices"], [0, 1])
        self.assertEqual(manifest["ground_cloud_indices"], [0])
        self.assertEqual(manifest["transition_frame_count"], 0)
        self.assertIsNone(manifest["bird_camera"])
        self.assertEqual(manifest["global_point_count"], 0)
        stats = [json.loads(line) for line in (a.output_dir / "filter_stats.jsonl").read_text().splitlines()]
        self.assertEqual([item["phase"] for item in stats], ["ground"])
        self.assertTrue((a.output_dir / "last_frame.png").is_file())
        self.assertFalse((a.output_dir / "last_frame_transparent.png").exists())
        self.assertFalse((a.output_dir / "birdseye_trajectory.png").exists())
        self.assertFalse((a.output_dir / "birdseye_pointcloud.png").exists())


class ArgumentValidationTests(unittest.TestCase):
    def test_nonfinite_float_arguments_are_rejected_before_rendering(self):
        # argparse accepts strings such as "nan" and "inf" for type=float.
        # Ordinary comparison checks are insufficient: NaN < 0 is False.
        fields = ("confidence", "min_depth", "max_depth",
                  "depth_edge", "fps", "ground_fov", "bird_fov",
                  "follow_distance", "distance_ratio", "distance_min",
                  "distance_max", "height_ratio", "look_ahead", "frustum_size")
        with tempfile.TemporaryDirectory() as directory:
            for name in fields:
                for value in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(parameter=name, value=value):
                        a = arguments(directory, **{name: value})
                        with self.assertRaises(ValueError):
                            render.validate_args(a)

    def test_finite_legal_float_arguments_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            a = arguments(directory, confidence=0,
                          min_depth=0, max_depth=0, depth_edge=0,
                          follow_distance=0, height_ratio=0, look_ahead=0)
            render.validate_args(a)


class OfflineSkyMaskTests(ResultFixture, unittest.TestCase):
    """Rendering consumes verified, already-thresholded native masks only."""
    def setUp(self):
        super().setUp()
        self.mask_dir = self.root / "sky_masks"
        self.mask_dir.mkdir()
        self.masks = []
        color = np.full((4, 6, 3), 100, np.uint8)
        self.manifest = {
            "schema_version": 1, "status": "complete", "mask_kind": "sky_rejected",
            "native_hw": [4, 6], "frames": 5,
            "dense_output_indices": self.indices,
            "color_sha256": [hashlib.sha256(color.tobytes()).hexdigest()] * 3,
            "mask_sha256": [],
        }
        for row, index in enumerate(self.indices):
            mask = np.zeros((4, 6), np.uint8)
            mask[:, :row + 1] = 255
            self.masks.append(mask)
            path = self.mask_dir / f"{index:08d}.png"
            Image.fromarray(mask).save(path)
            self.manifest["mask_sha256"].append(hashlib.sha256(path.read_bytes()).hexdigest())
        self.save_mask_manifest()

    def save_mask_manifest(self):
        (self.mask_dir / "manifest.json").write_text(json.dumps(self.manifest))

    def cli(self, *extra):
        return render.parser().parse_args([
            "--result-dir", str(self.root), "--output-dir", str(self.root / "rendered"),
            "--outputs", "trajectory", *extra,
        ])

    def assert_render_rejects(self):
        a = arguments(self.root, sky=True, sky_mask_dir=self.mask_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaises((ValueError, FileNotFoundError, OSError)):
                render.render(a)
        self.assertFalse(a.output_dir.exists(), "Mask validation must precede output creation")

    def test_cli_default_on_default_directory_and_explicit_disable(self):
        a = self.cli()
        self.assertTrue(a.sky)
        render.validate_args(a)
        self.assertEqual(a.sky_mask_dir.resolve(), self.mask_dir.resolve())
        self.assertFalse(self.cli("--no-sky").sky)
        self.assertTrue(self.cli("--no-sky", "--sky").sky)
        explicit = self.cli("--sky-mask-dir", str(self.mask_dir))
        self.assertEqual(explicit.sky_mask_dir, self.mask_dir)

    def test_missing_manifest_fails_before_loading_reconstruction(self):
        (self.mask_dir / "manifest.json").unlink()
        a = self.cli()
        with mock.patch.object(render, "ReconstructionReader") as reader:
            with self.assertRaisesRegex(FileNotFoundError, "generate_sky_masks|sky-mask-dir|no-sky"):
                render.render(a)
            reader.assert_not_called()
        self.assertFalse(a.output_dir.exists())

    def test_no_sky_does_not_need_manifest_or_load_masks(self):
        (self.mask_dir / "manifest.json").unlink()
        a = arguments(self.root, width=64, height=48, bird_cloud_step=1,
                      bird_pixel_step=1, frustum_step=0)
        with mock.patch.object(render, "SkyMasks", side_effect=AssertionError("sky disabled")), \
                mock.patch.dict("sys.modules", {"onnxruntime": None, "huggingface_hub": None}), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            render.render(a)
        manifest = json.loads((a.output_dir / "render_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")

    def test_renderer_has_no_onnx_or_download_imports(self):
        tree = ast.parse(Path(render.__file__).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(name.split(".")[0] in {"onnxruntime", "huggingface_hub"}
                             for name in imports), imports)
        self.assertFalse(hasattr(render, "SkyMask"), "Runtime inference class must not remain")

    def test_sparse_masks_use_pose_indices_and_do_not_read_source_rgb(self):
        reader = self.reader()
        a = arguments(self.root, sky=True, sky_mask_dir=self.mask_dir)
        sky = render.SkyMasks(a, reader)
        with mock.patch.object(reader, "rgb", side_effect=AssertionError("no source images")):
            for row, index in enumerate(self.indices):
                expected = self.masks[row] != 0
                actual = sky.rejected(index)
                self.assertEqual(actual.dtype, np.bool_)
                np.testing.assert_array_equal(actual, expected)
                _, _, stats = render.prepare_cloud(reader, row, a, 1, sky=sky)
                self.assertEqual(stats["pose_index"], index)
                self.assertEqual(stats["after_padding_sky"], int((~expected).sum()))
        with self.assertRaises((ValueError, KeyError, IndexError)):
            sky.rejected(1)  # Not a saved dense pose; must not silently use row 1.

    def test_invalid_manifest_metadata_rejected_before_outputs(self):
        baseline = json.loads(json.dumps(self.manifest))
        cases = {
            "schema_version": [2, "1"],
            "status": ["running", "failed"],
            "mask_kind": ["non_sky"],
            "native_hw": [[6, 4], [4, 7]],
            "frames": [3, 6],
            "dense_output_indices": [[0, 1, 4], [0, 2], [0.0, 2.0, 4.0]],
            "color_sha256": [[], ["bad"] * 3],
            "mask_sha256": [[], ["bad"] * 3],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(field=key, value=value):
                    self.manifest = {**baseline, key: value}
                    self.save_mask_manifest()
                    self.assert_render_rejects()
        self.manifest = baseline
        self.save_mask_manifest()

    def test_changed_native_rgb_is_rejected_before_outputs(self):
        colors = torch.full(self.local.shape, 100, dtype=torch.uint8)
        colors[1, 0, 0, 0] = 101
        torch.save(colors, self.root / "colors.pt")
        self.assert_render_rejects()

    def test_missing_png_is_rejected_before_outputs(self):
        (self.mask_dir / "00000002.png").unlink()
        self.assert_render_rejects()

    def test_checksum_mismatch_is_rejected_before_outputs(self):
        path = self.mask_dir / "00000002.png"
        Image.fromarray(255 - self.masks[1]).save(path)
        self.assert_render_rejects()

    def test_invalid_mask_encoding_size_and_binary_values_rejected(self):
        path = self.mask_dir / "00000002.png"
        for mask in (np.full((4, 6), 128, np.uint8),
                     np.zeros((6, 4), np.uint8),
                     np.zeros((4, 6, 3), np.uint8)):
            with self.subTest(shape=mask.shape, value=int(mask.max())):
                Image.fromarray(mask).save(path)
                self.manifest["mask_sha256"][1] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.save_mask_manifest()
                self.assert_render_rejects()

    def test_corrupt_png_is_rejected_even_when_checksum_matches(self):
        path = self.mask_dir / "00000002.png"
        path.write_bytes(b"not a PNG")
        self.manifest["mask_sha256"][1] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_mask_manifest()
        self.assert_render_rejects()

    def test_palette_png_and_renamed_non_png_are_rejected(self):
        path = self.mask_dir / "00000002.png"
        for image_format, mode in (("PNG", "P"), ("BMP", "L")):
            with self.subTest(image_format=image_format, mode=mode):
                image = Image.fromarray(np.zeros((4, 6), np.uint8)).convert(mode)
                image.save(path, format=image_format)
                self.manifest["mask_sha256"][1] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.save_mask_manifest()
                self.assert_render_rejects()


class SkyGeneratorTests(ResultFixture, unittest.TestCase):
    """Model execution is isolated to the preprocessing command, with no weights."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # patch.dict(sys.modules) restores the entire module table on exit.
        # Load real preprocessing/torchvision first so that a later restore
        # cannot unload modules whose native operators stay registered.
        import_module("abot_recon.preprocessing")

    def setUp(self):
        super().setUp()
        self.model = self.root / "skyseg.onnx"
        self.model.write_bytes(b"mock ONNX model")
        self.raw = np.zeros((9, 9), np.float32)
        self.raw[4, 4] = 1
        self.session = mock.Mock()
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        self.session.get_inputs.return_value = [SimpleNamespace(name="input")]
        self.session.run.side_effect = lambda *args, **kwargs: [self.raw[None, None].copy()]
        self.ort = dependency_module("onnxruntime",
            get_available_providers=mock.Mock(return_value=["CPUExecutionProvider"]),
            SessionOptions=SimpleNamespace,
            InferenceSession=mock.Mock(return_value=self.session),
        )
        patcher = mock.patch.dict("sys.modules", {"onnxruntime": self.ort})
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *extra):
        return generate_sky_masks.parser().parse_args([
            "--result-dir", str(self.root), "--model-path", str(self.model), *extra])

    def test_cli_defaults_and_explicit_download_opt_in(self):
        a = self.args()
        self.assertEqual(a.device, "cpu")
        self.assertEqual(a.non_sky_threshold, .3)
        self.assertEqual(a.sky_dilate, 3)
        self.assertFalse(a.download_model)
        self.assertTrue(self.args("--download-model").download_model)

    def test_missing_model_never_downloads_without_opt_in(self):
        a = self.args("--model-path", str(self.root / "missing.onnx"))
        hub = dependency_module("huggingface_hub", hf_hub_download=mock.Mock(side_effect=AssertionError("no automatic download")))
        with mock.patch.dict("sys.modules", {"huggingface_hub": hub}):
            with self.assertRaisesRegex(FileNotFoundError, "model-path|download-model"):
                generate_sky_masks.load_session(a)
        hub.hf_hub_download.assert_not_called()

    def test_cuda_provider_requirement_and_device_index(self):
        a = self.args("--device", "cuda:2")
        with self.assertRaisesRegex(RuntimeError, "CUDA provider unavailable"):
            generate_sky_masks.load_session(a)
        self.ort.InferenceSession.assert_not_called()
        self.ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.assertIs(generate_sky_masks.load_session(a), self.session)
        providers = self.ort.InferenceSession.call_args.kwargs["providers"]
        self.assertEqual(providers[0], ("CUDAExecutionProvider", {"device_id": 2}))
        self.session.disable_fallback.assert_called_once_with()

    def test_explicit_download_honors_model_path_and_records_provenance(self):
        destination = self.root / "checkpoints" / "custom-name.onnx"
        hub = dependency_module("huggingface_hub", hf_hub_download=mock.Mock(return_value=str(self.model)))
        with mock.patch.dict("sys.modules", {"huggingface_hub": hub}), \
                mock.patch.dict("os.environ", {"HF_ENDPOINT": "https://mirror.invalid"}), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            manifest = generate_sky_masks.generate(self.args(
                "--model-path", str(destination), "--download-model"))
        hub.hf_hub_download.assert_called_once_with(
            repo_id="JianyuanWang/skyseg", filename="skyseg.onnx",
            local_dir=str(destination.parent))
        self.assertEqual(destination.read_bytes(), self.model.read_bytes())
        self.assertEqual(manifest["model"]["path"], str(destination))
        self.assertTrue(manifest["model"]["downloaded_this_run"])
        self.assertEqual(manifest["model"]["hf_endpoint"], "https://mirror.invalid")

    def test_cuda_constructor_cpu_fallback_is_rejected(self):
        self.ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with self.assertRaises(RuntimeError):
            generate_sky_masks.load_session(self.args("--device", "cuda:0"))

    def test_native_default_dilation_is_radius_three_ellipse(self):
        reader = SimpleNamespace(h=9, w=9, paths=None, geometry=None)
        rgb = np.full((9, 9, 3), 150, np.uint8)
        mask = generate_sky_masks.predict_mask(self.session, "input", rgb, reader)
        expected = np.zeros((9, 9), np.uint8)
        expected[1:8, 1:8] = render.cv2.getStructuringElement(render.cv2.MORPH_ELLIPSE, (7, 7)) * 255
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(mask.dtype, np.uint8)
        feed = self.session.run.call_args.args[1]["input"]
        self.assertEqual(feed.shape, (1, 3, 320, 320))
        self.assertEqual(feed.dtype, np.float32)
        self.assertTrue(np.all(rgb == 150))

    def test_source_score_mapping_preserves_preprocessing_padding(self):
        reader = SimpleNamespace(h=6, w=4, paths=[Path("source.png")],
                                 geometry=render.source_geometry((2, 8), (6, 4)))
        self.raw = np.zeros((2, 8), np.float32)
        mask = generate_sky_masks.predict_mask(
            self.session, "input", np.full((2, 8, 3), 150, np.uint8),
            reader, threshold=.3, dilation_radius=0)
        expected = np.full((6, 4), 255, np.uint8)
        expected[2] = 0
        np.testing.assert_array_equal(mask, expected)

    def test_nonfinite_model_scores_are_rejected(self):
        reader = SimpleNamespace(h=9, w=9, paths=None, geometry=None)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.raw[4, 4] = value
                with self.assertRaisesRegex(ValueError, "invalid scores"):
                    generate_sky_masks.predict_mask(
                        self.session, "input", np.full((9, 9, 3), 150, np.uint8), reader)

    def test_generator_writes_complete_sparse_manifest_consumable_by_renderer(self):
        a = self.args()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            manifest = generate_sky_masks.generate(a)
        mask_dir = self.root / "sky_masks"
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["dense_output_indices"], [0, 2, 4])
        self.assertEqual(manifest["native_hw"], [4, 6])
        self.assertEqual(manifest["frames"], 5)
        self.assertEqual(manifest["input_mode"], "native")
        self.assertEqual(manifest["threshold"], .3)
        self.assertEqual(manifest["dilation_radius"], 3)
        self.assertEqual(manifest["provider"], "CPUExecutionProvider")
        self.assertEqual(self.session.run.call_count, 3)
        self.assertFalse((mask_dir / "00000001.png").exists())
        self.assertEqual(json.loads((mask_dir / "manifest.json").read_text()), manifest)
        sky = render.SkyMasks(arguments(self.root, sky=True, sky_mask_dir=mask_dir), self.reader())
        for row, index in enumerate(self.indices):
            path = mask_dir / f"{index:08d}.png"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["mask_sha256"][row])
            with Image.open(path) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (6, 4))
                self.assertTrue(set(np.unique(np.asarray(image))).issubset({0, 255}))
                np.testing.assert_array_equal(sky.rejected(index), np.asarray(image) != 0)

    def test_generator_failure_never_leaves_complete_manifest(self):
        self.session.run.side_effect = RuntimeError("inference failed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                generate_sky_masks.generate(self.args())
        manifest = json.loads((self.root / "sky_masks" / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        a = arguments(self.root, sky=True, sky_mask_dir=self.root / "sky_masks")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaises(ValueError):
                render.render(a)
        self.assertFalse(a.output_dir.exists())

    def test_mask_write_failure_never_commits_a_partial_sequence(self):
        atomic_write = generate_sky_masks.atomic_write

        def fail_second_mask(path, data):
            if path.name == "00000002.png":
                raise OSError("synthetic mask write failure")
            return atomic_write(path, data)

        with mock.patch.object(generate_sky_masks, "atomic_write", side_effect=fail_second_mask), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with self.assertRaisesRegex(OSError, "synthetic mask write failure"):
                generate_sky_masks.generate(self.args())
        mask_dir = self.root / "sky_masks"
        manifest = json.loads((mask_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["color_sha256"]), 1)
        self.assertEqual(len(manifest["mask_sha256"]), 1)
        self.assertTrue((mask_dir / "00000000.png").is_file())
        self.assertFalse((mask_dir / "00000002.png").exists())

    def test_source_generation_reads_mapped_source_frames_but_binds_native_rgb(self):
        reader = self.reader()
        reader.paths = []
        for index in range(5):
            path = self.root / f"source_{index}.png"
            Image.fromarray(np.full((4, 6, 3), 100, np.uint8)).save(path)
            reader.paths.append(path)
        reader.source_hw = (4, 6)
        reader.geometry = render.source_geometry((4, 6), (4, 6))
        reader.rgb = mock.Mock(side_effect=lambda index: np.full((4, 6, 3), 100, np.uint8))
        with mock.patch.object(generate_sky_masks, "ReconstructionReader", return_value=reader) as reader_factory, \
                mock.patch.object(generate_sky_masks, "predict_mask", wraps=generate_sky_masks.predict_mask) as predict:
            try:
                manifest = generate_sky_masks.generate(self.args())
                self.assertEqual(manifest["input_mode"], "source")
                self.assertEqual([call.args[0] for call in reader.rgb.call_args_list], [0, 2, 4])
                self.assertEqual([int(call.args[2][0, 0, 0]) for call in predict.call_args_list], [100, 100, 100])
                expected = hashlib.sha256(reader.colors[0].numpy().tobytes()).hexdigest()
                self.assertEqual(manifest["color_sha256"], [expected] * 3)
            finally:
                # Both spy arguments and the factory return value retain the
                # reader; clear both paths before the temporary input cleanup.
                predict.reset_mock()
                reader_factory.reset_mock(return_value=True)

    def test_wrong_source_rgb_at_nonsampled_dense_row_fails_before_prediction(self):
        # Reader's quick association checks cover rows 0, 2, 4. Generator must
        # additionally validate row 1, not hash its saved RGB beside a wrong mask.
        points = np.repeat(self.local[:1], 5, axis=0)
        torch.save(torch.from_numpy(points), self.root / "local_points.pt")
        torch.save(torch.full(points.shape, 100, dtype=torch.uint8), self.root / "colors.pt")
        torch.save(torch.ones(points.shape[:-1]), self.root / "confidence.pt")
        self.metadata["dense_output_indices"] = list(range(5))
        self.save_metadata()
        paths = []
        for index in range(5):
            path = self.root / f"source_{index}.png"
            value = 200 if index == 1 else 100
            Image.fromarray(np.full((4, 6, 3), value, np.uint8)).save(path)
            paths.append(str(path))
        frame_list = self.root / "frames.json"
        frame_list.write_text(json.dumps(paths))
        with self.assertRaisesRegex(ValueError, "dense row 1"):
            generate_sky_masks.generate(self.args("--frame-list", str(frame_list)))
        self.assertEqual(self.session.run.call_count, 1)
        mask_dir = self.root / "sky_masks"
        manifest = json.loads((mask_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["color_sha256"]), 1)
        self.assertTrue((mask_dir / "00000000.png").is_file())
        self.assertFalse((mask_dir / "00000001.png").exists())

    def test_overwrite_shrink_then_expand_removes_only_obsolete_owned_masks(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            generate_sky_masks.generate(self.args())
        mask_dir = self.root / "sky_masks"
        unrelated = mask_dir / "00000003.png"
        unrelated.write_bytes(b"unrelated image must survive")
        note = mask_dir / "notes.txt"
        note.write_text("user note")

        def save_dense_rows(count):
            points = self.local[:count]
            torch.save(torch.from_numpy(points), self.root / "local_points.pt")
            torch.save(torch.full(points.shape, 100, dtype=torch.uint8), self.root / "colors.pt")
            torch.save(torch.ones(points.shape[:-1]), self.root / "confidence.pt")
            self.metadata["dense_output_indices"] = self.indices[:count]
            self.save_metadata()

        save_dense_rows(2)
        write_manifest = generate_sky_masks.write_manifest
        obsolete_present_on_commit = []

        def record_manifest(path, manifest):
            write_manifest(path, manifest)
            obsolete_present_on_commit.append((manifest["status"], (mask_dir / "00000004.png").exists()))

        with mock.patch.object(generate_sky_masks, "write_manifest", side_effect=record_manifest), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            shrunk = generate_sky_masks.generate(self.args("--overwrite"))
        self.assertEqual(shrunk["dense_output_indices"], [0, 2])
        self.assertEqual(obsolete_present_on_commit, [("running", True), ("complete", False)])
        self.assertFalse((mask_dir / "00000004.png").exists())

        save_dense_rows(3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expanded = generate_sky_masks.generate(self.args("--overwrite"))
        self.assertEqual(expanded["status"], "complete")
        self.assertEqual(expanded["dense_output_indices"], [0, 2, 4])
        self.assertTrue((mask_dir / "00000004.png").is_file())
        self.assertEqual(unrelated.read_bytes(), b"unrelated image must survive")
        self.assertEqual(note.read_text(), "user note")


class FramePlanTests(unittest.TestCase):
    def test_ground_frame_count_and_view_stride_are_independent(self):
        views, end = render.frame_plan(arguments(start=3, end=30, ground_frames=4, view_step=2), 100)
        np.testing.assert_array_equal(views, [3, 5, 7, 9])
        self.assertEqual(end, 30)

    def test_short_input_is_not_repeated_to_fill_video(self):
        views, end = render.frame_plan(arguments(ground_frames=600, view_step=2), 7)
        np.testing.assert_array_equal(views, [0, 2, 4, 6])
        self.assertEqual(end, 7)

    def test_selection_uses_inference_indices_not_dense_row_numbers(self):
        reader = SimpleNamespace(indices=np.array([0, 2, 3, 5, 6, 9]))
        rows = render.selected_rows(reader, 3, 10, 3)
        np.testing.assert_array_equal(rows, [2, 4, 5])
        np.testing.assert_array_equal(reader.indices[rows], [3, 6, 9])

    def test_invalid_ranges_raise_instead_of_silently_clamping(self):
        for start, end in ((-1, 5), (5, 5), (7, 5), (0, 11)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    render.frame_plan(arguments(start=start, end=end), 10)


class GeometryTests(unittest.TestCase):
    def test_identity_source_pixel_centers(self):
        mx, my, valid, rh, crop, pad = render.source_geometry((4, 6), (4, 6))
        yy, xx = np.mgrid[:4, :6]
        np.testing.assert_array_equal(mx, xx)
        np.testing.assert_array_equal(my, yy)
        self.assertTrue(valid.all())
        self.assertEqual((rh, crop, pad), (4, 0, 0))

    def test_wide_source_padding_has_exact_valid_rows(self):
        mx, my, valid, rh, crop, pad = render.source_geometry((2, 8), (6, 4))
        self.assertEqual((rh, crop, pad), (1, 0, 2))
        self.assertEqual(np.flatnonzero(valid.any(1)).tolist(), [2])
        np.testing.assert_allclose(mx[0], (np.arange(8) + .5) * .5 - .5)
        np.testing.assert_allclose(my[:, 0], [1.75, 2.25])

    def test_tall_source_center_crop_does_not_claim_removed_rows(self):
        mx, my, valid, rh, crop, pad = render.source_geometry((8, 4), (4, 4))
        self.assertEqual((rh, crop, pad), (8, 2, 0))
        self.assertTrue(valid.all())
        keep = (my[:, 0] >= -.5) & (my[:, 0] < 3.5)
        self.assertEqual(np.flatnonzero(keep).tolist(), [2, 3, 4, 5])

    def test_rounded_resize_uses_distinct_x_y_scales(self):
        mx, my, _, rh, crop, pad = render.source_geometry((7, 10), (4, 6))
        self.assertEqual(rh, 4)
        self.assertAlmostEqual(float(mx[0, 1] - mx[0, 0]), 6 / 10, places=6)
        self.assertAlmostEqual(float(my[1, 0] - my[0, 0]), 4 / 7, places=6)

    def test_finite_geometry_confidence_and_depth_filters(self):
        p = point_grid(3, 4)
        p[0, 0, 0] = np.nan
        p[0, 1, 2] = 0
        p[0, 2, 2] = -1
        p[0, 3, 2] = 8
        conf = np.full((3, 4), .5, np.float32)
        conf[1, 0] = .1
        a = arguments(confidence=.5, min_depth=.1, max_depth=6)
        valid = render.native_depth_valid(p, conf, a)
        expected = np.ones((3, 4), bool)
        expected[0] = False
        expected[1, 0] = False
        np.testing.assert_array_equal(valid, expected)

    def test_invalid_sigmoid_scores_fail_loudly(self):
        for value in (float("nan"), float("inf"), -.01, 1.01):
            with self.subTest(value=value):
                conf = np.ones((4, 6), np.float32)
                conf[0, 0] = value
                with self.assertRaisesRegex(ValueError, "sigmoid"):
                    render.native_depth_valid(point_grid(), conf, arguments())

    def test_depth_edge_removes_both_sides_of_step_not_flat_region(self):
        points = point_grid(5, 8)
        points[:, 4:, 2] = 4
        valid = render.native_depth_valid(points, None, arguments(depth_edge=.15))
        expected = np.ones((5, 8), bool)
        expected[:, 3:5] = False
        np.testing.assert_array_equal(valid, expected)

    def test_native_pixel_stride_is_applied_on_both_axes(self):
        points = point_grid(4, 6)
        world, colors, _ = render.prepare_cloud(ArrayReader(points), 0, arguments(confidence=0), 2)
        np.testing.assert_allclose(world, points[::2, ::2].reshape(-1, 3))
        self.assertEqual(len(colors), 6)

    def test_native_padding_removed_when_source_geometry_is_known(self):
        points = point_grid(6, 4)
        reader = ArrayReader(points, source_hw=(2, 8))
        world, _, stats = render.prepare_cloud(reader, 0, arguments(confidence=0), 1)
        self.assertEqual(len(world), 4)
        self.assertEqual(stats["after_padding_sky"], 4)
        np.testing.assert_allclose(world, points[2])

    def test_source_crop_produces_only_observed_pixels(self):
        reader = ArrayReader(point_grid(4, 4), source_hw=(8, 4))
        world, _, _ = render.prepare_cloud(reader, 0, arguments(confidence=0, resolution="source"), 1)
        self.assertEqual(len(world), 16)
        np.testing.assert_allclose(world, reader._points.reshape(-1, 3))

    def test_identity_upsample_preserves_original_xyz_not_refitted_intrinsics(self):
        points = point_grid(4, 6)
        points[1, 2, 0] += .17  # Deliberately deviate from a single pinhole model.
        reader = ArrayReader(points, source_hw=(4, 6))
        world, colors, _ = render.prepare_cloud(reader, 0, arguments(confidence=0, resolution="source"), 1)
        np.testing.assert_allclose(world, points.reshape(-1, 3), atol=1e-7)
        self.assertTrue(np.all(colors == 200))

    def test_bilinear_upsample_never_reintroduces_invalid_footprints(self):
        p = point_grid(4, 4)
        p[1, 1] = np.nan
        reader = ArrayReader(p, source_hw=(8, 8))
        world, _, _ = render.prepare_cloud(reader, 0, arguments(confidence=0, resolution="source"), 1)
        # The invalid native pixel participates in the 4x4 source footprint
        # [1:5,1:5]. All 16 corresponding outputs must remain absent.
        self.assertEqual(len(world), 64 - 16)
        self.assertTrue(np.isfinite(world).all())
        self.assertTrue(np.all(world[:, 2] > 0))

    def test_sky_rejection_uses_same_path_for_native_and_upsampled_clouds(self):
        class LeftSky:
            def rejected(self, pose_index):
                mask = np.zeros((4, 4), bool)
                mask[:, :2] = True
                return mask
        reader = ArrayReader(point_grid(4, 4), source_hw=(4, 4))
        for resolution in ("native", "source"):
            with self.subTest(resolution=resolution):
                world, _, stats = render.prepare_cloud(reader, 0, arguments(confidence=0, resolution=resolution), 1, sky=LeftSky())
                self.assertEqual(len(world), 8)
                self.assertEqual(stats["after_padding_sky"], 8)

    def test_intrinsics_fit_is_accurate_for_exact_pinhole_map(self):
        fitted, residual = render.fit_intrinsics(point_grid(12, 16))
        np.testing.assert_allclose(fitted, [16, 7.5, 12, 5.5], atol=1e-5)
        self.assertLess(residual, 1e-5)


class CameraAndRasterTests(unittest.TestCase):
    def camera(self):
        return render.Camera(np.zeros(3), np.array([0., 0., 1.]), np.array([0., -1., 0.]), 90)

    def raster(self, size=1, chunk=100):
        return render.Rasterizer(arguments(width=17, height=17, point_size=size, chunk_points=chunk))

    def test_camera_axes_are_proper_c2w_and_image_y_points_down(self):
        camera = self.camera()
        np.testing.assert_allclose(camera.rotation, np.eye(3))
        xy, z = camera.project(np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]]), 17, 17)
        np.testing.assert_allclose(xy[0], [8, 8])
        self.assertGreater(xy[1, 1], xy[0, 1])
        self.assertGreater(xy[2, 0], xy[0, 0])
        self.assertAlmostEqual(np.linalg.det(camera.rotation), 1)

    def test_raster_depth_test_and_behind_camera_rejection(self):
        raster = self.raster()
        raster.add(np.array([[0, 0, 2], [0, 0, 1], [0, 0, -1]], np.float32),
                   np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8), self.camera())
        rgba = raster.rgba()
        np.testing.assert_array_equal(rgba[8, 8], [0, 255, 0, 255])
        self.assertEqual(np.count_nonzero(rgba[..., 3]), 1)

    def test_point_size_is_exact_square_diameter(self):
        for size in (1, 2, 3, 4):
            with self.subTest(size=size):
                raster = self.raster(size)
                raster.add(np.array([[0, 0, 1]], np.float32), np.array([[12, 34, 56]], np.uint8), self.camera())
                self.assertEqual(np.count_nonzero(raster.rgba()[..., 3]), size * size)

    def test_equal_depth_same_pixel_preserves_first_color(self):
        points = np.array([[0, 0, 1], [0, 0, 1]], np.float32)
        colors = np.array([[10, 20, 30], [90, 80, 70]], np.uint8)
        for chunk in (1, 10):
            with self.subTest(chunk=chunk):
                raster = self.raster(chunk=chunk)
                raster.add(points, colors, self.camera())
                np.testing.assert_array_equal(raster.rgba()[8, 8], [10, 20, 30, 255])

    def test_equal_depth_overlapping_splats_are_chunk_invariant(self):
        # Pixel centers are x=7 and x=8. Size-2 squares overlap in column 7.
        # Changing chunk size must not change which equally deep point wins.
        points = np.array([[-1 / 8.5, 0, 1], [0, 0, 1]], np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
        images = []
        for chunk in (1, 10):
            raster = self.raster(size=2, chunk=chunk)
            raster.add(points, colors, self.camera())
            images.append(raster.rgba())
        np.testing.assert_array_equal(images[0], images[1])
        np.testing.assert_array_equal(images[0][7, 7], [255, 0, 0, 255])

    def test_in_frustum_matches_camera_convention(self):
        pts = np.array([[0, 0, 1], [0, 0, -1], [10, 0, 1], [0, 10, 1], [np.nan, 0, 1]])
        np.testing.assert_array_equal(render.in_frustum(pts, self.camera(), 17, 17), [True, False, False, False, False])

    def test_follow_camera_does_not_mutate_reconstructed_poses(self):
        poses = np.repeat(np.eye(4)[None], 8, axis=0)
        poses[:, 0, 3] = np.arange(8)
        before = poses.copy()
        a = arguments(height_ratio=.2, look_ahead=.35, smooth_frames=6)
        camera = render.follow_camera(poses, 7, a, 1.0)
        np.testing.assert_allclose(camera.eye, [4, -.2, -1])
        np.testing.assert_allclose(camera.target, [4, 0, .35])
        np.testing.assert_array_equal(poses, before)

    def test_follow_camera_survives_exact_180_degree_average_cancellation(self):
        # Each matrix is a proper rotation: yaw flips forward only; roll flips
        # up only; pitch flips both. A two-frame mean can be exactly zero.
        rotations = {"yaw": np.diag([-1., 1., -1.]),
                     "roll": np.diag([-1., -1., 1.]),
                     "pitch": np.diag([1., -1., -1.])}
        a = arguments(smooth_frames=2, height_ratio=.2, look_ahead=.35)
        for name, rotation in rotations.items():
            with self.subTest(rotation=name):
                poses = np.repeat(np.eye(4)[None], 2, axis=0)
                poses[1, :3, :3] = rotation
                poses[1, :3, 3] = [1., 2., 3.]
                before = poses.copy()
                camera = render.follow_camera(poses, 1, a, 1)
                forward, up = rotation[:, 2], -rotation[:, 1]
                center = poses[:, :3, 3].mean(0)
                np.testing.assert_allclose(camera.eye, center - forward + .2 * up)
                np.testing.assert_allclose(camera.target, center + .35 * forward)
                np.testing.assert_allclose(camera.rotation.T @ camera.rotation, np.eye(3), atol=1e-7)
                self.assertAlmostEqual(np.linalg.det(camera.rotation), 1)
                np.testing.assert_array_equal(poses, before)

    def test_camera_parallel_or_zero_up_uses_finite_perpendicular_basis(self):
        for up in ([0., 0., 1.], [0., 0., -2.], [0., 0., 0.]):
            with self.subTest(up=up):
                camera = render.Camera(np.zeros(3), np.array([0., 0., 1.]), np.asarray(up), 60)
                rotation = camera.rotation
                self.assertTrue(np.isfinite(rotation).all())
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                np.testing.assert_allclose(rotation[:, 2], [0, 0, 1])
                self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_default_fov_matches_wangjing_vertical_60(self):
        a = arguments(width=1920, height=1080)
        self.assertEqual((a.ground_fov, a.bird_fov), (60, 60))
        camera = render.Camera(np.zeros(3), np.array([0., 0., 1.]), np.array([0., -1., 0.]), a.ground_fov)
        xy, _ = camera.project(np.array([[1., 0., 1.], [0., 1., 1.]]), a.width, a.height)
        focal = 1080 / (2 * np.tan(np.deg2rad(60)/2))
        np.testing.assert_allclose(xy, [[959.5+focal, 539.5], [959.5, 539.5+focal]])
        self.assertIn("vertical_fov", camera.as_dict())
        self.assertNotIn("horizontal_fov", camera.as_dict())

    def test_centered_follow_windows_match_wangjing_at_boundaries(self):
        poses = np.repeat(np.eye(4)[None], 25, axis=0)
        poses[:, :3, :3] = render.Rotation.from_euler("xyz", np.column_stack((np.linspace(0, .3, 25), np.linspace(0, .6, 25), np.linspace(0, .1, 25)))).as_matrix()
        poses[:, :3, 3] = np.column_stack((np.arange(25)**2, np.zeros(25), np.arange(25)))
        a = arguments()
        before = poses.copy()
        for i in (0, 1, 6, 12, 22, 24):
            window = poses[max(0, i-6):min(25, i+7)]
            center = window[:, :3, 3].mean(0)
            forward = render.unit(window[:, :3, 2].mean(0))
            lo = max(0, i-3)
            hi = min(25, lo+6)
            lo = max(0, hi-6)
            up = render.unit(-poses[lo:hi, :3, 1].mean(0))
            camera = render.follow_camera(poses, i, a, 1.3)
            np.testing.assert_allclose(camera.eye, center-forward*1.3+up*.26)
            np.testing.assert_allclose(camera.target, center+forward*1.3*.35)
            np.testing.assert_allclose(camera.up, up)
        np.testing.assert_array_equal(poses, before)

    def test_camera_zero_view_direction_is_rejected_not_invented(self):
        camera = render.Camera(np.zeros(3), np.zeros(3), np.array([0., -1., 0.]), 60)
        with self.assertRaises(ValueError):
            _ = camera.rotation

    def test_transition_endpoints_and_intermediate_camera_are_valid(self):
        start = self.camera()
        end = render.Camera(np.array([0., -10., 0.]), np.zeros(3), np.array([0., 0., 1.]), 60)
        self.assertIs(render.transition_camera(start, end, 0), start)
        self.assertIs(render.transition_camera(start, end, 1), end)
        for fraction in np.linspace(.05, .95, 9):
            camera = render.transition_camera(start, end, fraction)
            rot = camera.rotation
            self.assertTrue(np.isfinite(camera.eye).all())
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(rot), 1)
            self.assertGreaterEqual(camera.fov, 60)
            self.assertLessEqual(camera.fov, 90)
        for fraction, reference in ((1e-7, start), (1 - 1e-7, end)):
            camera = render.transition_camera(start, end, fraction)
            np.testing.assert_allclose(camera.eye, reference.eye, atol=1e-5)
            np.testing.assert_allclose(camera.rotation, reference.rotation, atol=1e-5)

    def test_bird_camera_fits_all_selected_camera_centers(self):
        poses = np.repeat(np.eye(4)[None], 20, axis=0)
        t = np.linspace(0, 2 * np.pi, 20)
        poses[:, :3, 3] = np.column_stack((4*np.cos(t), .1*np.sin(t), 2*np.sin(t)))
        a = arguments(width=640, height=360)
        camera = render.bird_camera(poses, self.camera(), a)
        self.assertTrue(render.in_frustum(poses[:, :3, 3], camera, a.width, a.height).all())

    def test_bird_median_up_cancellation_falls_back_to_reference_up(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, :3, :3] = np.diag([1., -1., -1.])
        poses[1, :3, 3] = [2., 0., 1.]
        before = poses.copy()
        reference = self.camera()
        a = arguments(width=640, height=360)
        bird = render.bird_camera(poses, reference, a)
        np.testing.assert_allclose(render.unit(bird.eye - bird.target), reference.up)
        np.testing.assert_allclose(bird.rotation.T @ bird.rotation, np.eye(3), atol=1e-7)
        self.assertTrue(render.in_frustum(poses[:, :3, 3], bird, a.width, a.height).all())
        np.testing.assert_array_equal(poses, before)

    def test_bird_reference_looking_along_world_up_uses_right_axis(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = [-1., 0., 1.]
        reference = render.Camera(np.zeros(3), np.array([0., -1., 0.]), np.array([0., 0., 1.]), 60)
        a = arguments(width=640, height=360)
        bird = render.bird_camera(poses, reference, a)
        self.assertTrue(np.isfinite(bird.rotation).all())
        np.testing.assert_allclose(bird.rotation.T @ bird.rotation, np.eye(3), atol=1e-7)
        self.assertTrue(render.in_frustum(poses[:, :3, 3], bird, a.width, a.height).all())

    def test_bird_static_trajectory_has_nonzero_observer_distance(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        a = arguments(width=640, height=360)
        bird = render.bird_camera(poses, self.camera(), a)
        self.assertGreater(np.linalg.norm(bird.eye - bird.target), 0)
        self.assertTrue(render.in_frustum(poses[:, :3, 3], bird, a.width, a.height).all())

    def test_straight_alpha_and_composite_do_not_add_white_halo(self):
        premult = np.array([[[50, 25, 0, 128], [0, 0, 0, 0]]], np.uint8)
        straight = render.straight_alpha(premult)
        np.testing.assert_array_equal(straight[0, 0], [100, 50, 0, 128])
        np.testing.assert_array_equal(straight[0, 1], [0, 0, 0, 0])
        cloud = np.zeros((1, 2, 4), np.uint8)
        actual = render.composite(cloud, premult, [0, 0, 0])
        np.testing.assert_array_equal(actual[0, 0], [50, 25, 0])
        np.testing.assert_array_equal(actual[0, 1], [0, 0, 0])

    def test_cloud_opacity_zero_leaves_trajectory_and_background(self):
        cloud = np.array([[[200, 100, 50, 255]]], np.uint8)
        line = np.zeros_like(cloud)
        np.testing.assert_array_equal(render.composite(cloud, line, [32, 32, 32], opacity=0), [[[32, 32, 32]]])

    def test_trajectory_export_alpha_is_zero_outside_drawn_geometry(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, :3, 3] = [[-.5, 0, 2], [.5, 0, 2]]
        a = arguments(width=64, height=64, frustum_step=0, center_size=0)
        layer = render.trajectory_layer(poses, render.time_colors(2), None, self.camera(), [0, 1], a)
        alpha = layer[..., 3]
        self.assertGreater(np.count_nonzero(alpha), 0)
        self.assertLess(np.count_nonzero(alpha), 64 * 64)
        self.assertTrue(np.all(layer[alpha == 0, :3] == 0))
        output = render.straight_alpha(layer)
        np.testing.assert_array_equal(output[..., 3], alpha)

    def test_gradient_endpoints_are_bright_blue_and_red(self):
        colors = render.time_colors(100)
        np.testing.assert_array_equal(colors[0], [37, 99, 235])
        np.testing.assert_array_equal(colors[-1], [239, 68, 68])


if __name__ == "__main__":
    unittest.main(verbosity=2)
