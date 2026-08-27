import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


MODULE_PATH = Path(__file__).parents[2] / "interfaces" / "official_streaming.py"
SPEC = importlib.util.spec_from_file_location("official_streaming_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


LONGSTREAM_PATH = Path(__file__).parents[2] / "interfaces" / "longstream.py"
LONGSTREAM_SPEC = importlib.util.spec_from_file_location(
    "longstream_adapter_under_test", LONGSTREAM_PATH
)
LONGSTREAM_MODULE = importlib.util.module_from_spec(LONGSTREAM_SPEC)
sys.modules[LONGSTREAM_SPEC.name] = LONGSTREAM_MODULE
LONGSTREAM_SPEC.loader.exec_module(LONGSTREAM_MODULE)


def test_crop_geometry_for_standard_indoor_frame_has_full_fov(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (640, 480)).save(path)
    geometry = MODULE.official_crop_geometry(str(path))

    assert geometry.resized_w == 518
    assert geometry.resized_h == 392
    assert geometry.native_rows() == slice(0, 480)
    assert geometry.rows_on_grid(388) == slice(0, 388)


def test_oxford_native_rows_are_scaled_to_metric_gt_grid(tmp_path):
    path = tmp_path / "oxford.png"
    Image.new("RGB", (1440, 1080)).save(path)
    geometry = MODULE.official_crop_geometry(str(path))
    points = np.ones((2, 392, 518, 3), dtype=np.float32)
    depth = np.ones((2, 392, 518), dtype=np.float32)

    mapped_points, observed = MODULE.map_world_points_to_native(points, geometry, (388, 518))
    mapped_depth = MODULE.map_depth_to_native(depth, geometry, (388, 518))
    assert mapped_points.shape == (2, 388, 518, 3)
    assert mapped_depth.shape == (2, 388, 518)
    assert observed.all()


def test_tall_crop_maps_only_observed_native_rows(tmp_path):
    path = tmp_path / "tall.png"
    Image.new("RGB", (400, 800)).save(path)
    geometry = MODULE.official_crop_geometry(str(path))
    points = np.ones((2, 518, 518, 3), dtype=np.float32)

    mapped, observed = MODULE.map_world_points_to_native(points, geometry, (800, 400))
    rows = geometry.native_rows()
    assert mapped.shape == (2, 800, 400, 3)
    assert observed[:, rows, :].all()
    assert not observed[:, : rows.start, :].any()
    assert not observed[:, rows.stop :, :].any()
    assert np.isnan(mapped[:, : rows.start]).all()


def test_w2c_to_c2w_inverts_rotation_and_translation():
    w2c = torch.tensor([[[0.0, -1.0, 0.0, 2.0], [1.0, 0.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0]]])
    c2w = MODULE.w2c_3x4_to_c2w(w2c)
    full_w2c = np.eye(4, dtype=np.float32)
    full_w2c[:3] = w2c[0].numpy()
    np.testing.assert_allclose(c2w[0] @ full_w2c, np.eye(4), atol=1e-6)


def test_lazy_frames_keeps_only_one_transferred_frame(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 6), color=(index, 0, 0)).save(path)
        paths.append(str(path))

    calls = []

    def loader(filelist, mode):
        calls.append((filelist[0], mode))
        return torch.zeros(1, 3, 6, 8)

    frames = MODULE.LazyOfficialFrames(paths, loader, "cpu")
    assert frames[0]["img"].shape == (1, 3, 6, 8)
    assert frames[0]["img"].shape == (1, 3, 6, 8)
    assert len(calls) == 1
    _ = frames[1]
    assert len(calls) == 2


def test_longstream_point_head_mask_uses_finite_xyz_and_positive_local_z():
    points = np.ones((1, 2, 3, 3), dtype=np.float32)
    local = np.ones_like(points)
    points[0, 1, 2, 0] = np.nan
    local[0, 0, 1, 2] = -1.0

    mask = LONGSTREAM_MODULE._point_head_valid_mask(points, local)

    assert mask.shape == (1, 2, 3)
    assert mask.sum() == 4
    assert not mask[0, 0, 1]
    assert not mask[0, 1, 2]


def test_depth_maps_to_same_native_roi_as_world_points(tmp_path):
    path = tmp_path / "tall.png"
    Image.new("RGB", (400, 800)).save(path)
    geometry = MODULE.official_crop_geometry(str(path))
    depth = np.ones((2, 518, 518), dtype=np.float32)
    mapped = MODULE.map_depth_to_native(depth, geometry, (800, 400))
    rows = geometry.native_rows()
    assert mapped[:, rows, :].shape == (2, rows.stop - rows.start, 400)
    assert np.isfinite(mapped[:, rows, :]).all()
    assert np.isnan(mapped[:, : rows.start]).all()


def test_recurrent_pose_collector_never_allocates_dense_sequence_array():
    class FakeModel:
        def inference(self, frames, frame_writer, cache_results):
            assert cache_results is False
            assert frame_writer.required_result_keys == frozenset({"camera_pose"})
            for index, frame in enumerate(frames):
                result = {
                    "camera_pose": torch.full((1, 9), float(index)),
                    "pts3d_in_other_view": torch.empty(1, 200, 300, 3),
                    "depth": torch.ones(1, 200, 300),
                }
                frame_writer(index, frame, result)

    frames = [{"img": torch.zeros(1, 3, 12, 16)} for _ in range(4)]
    pose_enc, image_hw = MODULE._collect_recurrent_pose_outputs(FakeModel(), frames, len(frames))
    assert pose_enc.shape == (4, 9)
    assert image_hw == (12, 16)
    assert pose_enc[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_recurrent_dense_collector_keeps_only_requested_frames():
    class FakeModel:
        def inference(self, frames, frame_writer, cache_results):
            # InfiniteVGGT/OVGGT only filter GPU->CPU fields when the pose-only
            # collector advertises required_result_keys. Point-cloud inference
            # must keep the official dense outputs numerically untouched.
            assert not hasattr(frame_writer, "required_result_keys")
            for index, frame in enumerate(frames):
                frame_writer(
                    index,
                    frame,
                    {
                        "camera_pose": torch.full((1, 9), float(index)),
                        "pts3d_in_other_view": torch.full((1, 2, 3, 3), float(index)),
                        "depth": torch.full((1, 2, 3), float(index + 1)),
                    },
                )

    frames = [{"img": torch.zeros(1, 3, 2, 3)} for _ in range(5)]
    points, depth, pose = MODULE._collect_recurrent_outputs(
        FakeModel(), frames, 5, output_indices=[0, 2, 4]
    )
    assert points.shape == (3, 2, 3, 3)
    assert depth.shape == (3, 2, 3)
    assert pose.shape == (5, 9)
    assert points[:, 0, 0, 0].tolist() == [0.0, 2.0, 4.0]


def test_longstream_sparse_dense_stitch_is_exact_subset_of_full_output():
    source_root = Path(__file__).parents[2] / "third_party" / "LongStream"
    if not source_root.is_dir():
        pytest.skip("third_party/LongStream is not installed")
    sys.path.insert(0, str(source_root))
    try:
        from longstream.streaming.refresh import (
            _append_batch_output,
            _finalize_stitched_batches,
        )

        def run(indices):
            tensors = {}
            scalars = {}
            for global_start, slice_start in ((0, 0), (4, 1)):
                frame_ids = torch.arange(global_start, global_start + 5)
                output = {
                    "pose_enc": frame_ids.view(1, 5, 1).float(),
                    "world_points": frame_ids.view(1, 5, 1, 1, 1).float(),
                    "depth": (frame_ids + 100).view(1, 5, 1, 1).float(),
                    "global_scale": torch.tensor(2.0),
                }
                _append_batch_output(
                    tensors,
                    scalars,
                    output,
                    actual_frames=5,
                    slice_start=slice_start,
                    global_start=global_start,
                    dense_output_indices=indices,
                )
                assert output == {}
            return _finalize_stitched_batches(tensors, scalars)

        full = run(None)
        selected_ids = [0, 3, 4, 7, 8]
        sparse = run(selected_ids)
        torch.testing.assert_close(sparse["world_points"], full["world_points"][:, selected_ids])
        torch.testing.assert_close(sparse["depth"], full["depth"][:, selected_ids])
        # Pose/state outputs remain dense because every frame is needed to
        # transform selected point maps into one world coordinate system.
        torch.testing.assert_close(sparse["pose_enc"], full["pose_enc"])
        torch.testing.assert_close(sparse["global_scale"], full["global_scale"])
    finally:
        sys.path.remove(str(source_root))


def test_infinite_pose_adapter_decodes_camera_without_point_sequence(monkeypatch):
    class Recurrent:
        def inference(self, frames, frame_writer, cache_results):
            for index, frame in enumerate(frames):
                frame_writer(
                    index,
                    frame,
                    {
                        "camera_pose": torch.zeros(1, 9),
                        "pts3d_in_other_view": torch.empty(1, 20, 30, 3),
                    },
                )

    def loader(paths, mode):
        return torch.zeros(len(paths), 3, 20, 30)

    def pose_decoder(pose_enc, image_hw):
        count = pose_enc.shape[1]
        extrinsic = torch.eye(4).repeat(1, count, 1, 1)[:, :, :3]
        return extrinsic, torch.eye(3).repeat(1, count, 1, 1)

    wrapper = SimpleNamespace(
        model=Recurrent(),
        family="infinitevggt",
        image_loader=loader,
        preprocess_mode="crop",
        pose_decoder=pose_decoder,
    )
    monkeypatch.setattr(MODULE.torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(MODULE.torch.amp, "autocast", lambda *a, **k: nullcontext())
    c2w = MODULE._infer_infinite_or_ovggt_poses(
        ["a", "b", "c"], wrapper, SimpleNamespace(device="cpu")
    )
    assert c2w.shape == (3, 4, 4)
    np.testing.assert_allclose(c2w, np.repeat(np.eye(4)[None], 3, axis=0), atol=1e-6)


def test_stream3r_pose_adapter_advances_official_cache_without_stacking_depth():
    class Session:
        def __init__(self):
            self.aggregator_kv_cache_list = []
            self.camera_head_kv_cache_list = []
            self.predictions = None
            self.update_count = 0
            self.model = self
            self.point_head = object()
            self.depth_head = object()

        def __call__(self, **kwargs):
            assert self.point_head is None
            assert self.depth_head is None
            return {
                "pose_enc": torch.zeros(1, 1, 9),
                "aggregator_kv_cache_list": [self.update_count],
                "camera_head_kv_cache_list": [self.update_count],
            }

        def _update_cache(self, aggregator, camera):
            assert self.predictions["depth"].shape == (0, 0, 12, 16)
            self.update_count += 1

    session = Session()

    def pose_decoder(pose_enc, image_hw):
        count = pose_enc.shape[1]
        extrinsic = torch.eye(4).repeat(1, count, 1, 1)[:, :, :3]
        return extrinsic, None

    wrapper = SimpleNamespace(
        make_session=lambda: session,
        chunk_size=1,
        image_loader=lambda paths, mode: torch.zeros(len(paths), 3, 12, 16),
        preprocess_mode="crop",
        mode="window",
        pose_decoder=pose_decoder,
    )
    c2w = MODULE._infer_stream3r_poses(["a", "b", "c", "d"], wrapper, SimpleNamespace(device="cpu"))
    assert session.update_count == 4
    assert c2w.shape == (4, 4, 4)
    assert session.point_head is not None
    assert session.depth_head is not None


def test_pose_only_head_contexts_restore_heads_after_exception():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.point_head = torch.nn.Identity()
            self.depth_head = torch.nn.Identity()

    for context_factory in (
        MODULE._infinitevggt_pose_only_heads,
        MODULE._stream3r_pose_only_heads,
    ):
        core = Core()
        original_point = core.point_head
        original_depth = core.depth_head
        try:
            with context_factory(core):
                raise RuntimeError("expected")
        except RuntimeError:
            pass
        assert core.point_head is original_point
        assert core.depth_head is original_depth


def test_discarded_dense_head_has_tiny_shape_compatible_outputs():
    images = torch.zeros(2, 3, 3, 12, 16)
    points, confidence = MODULE._DiscardedDenseHead(3)(None, images=images)
    depth, depth_confidence = MODULE._DiscardedDenseHead(1)(None, images=images)
    assert points.shape == (2, 3, 1, 1, 3)
    assert depth.shape == (2, 3, 1, 1, 1)
    assert confidence.shape == depth_confidence.shape == (2, 3, 1, 1)


def test_stream3r_dense_adapter_uses_direct_world_points_and_sparse_outputs():
    class Session:
        def __init__(self):
            self.aggregator_kv_cache_list = []
            self.camera_head_kv_cache_list = []
            self.predictions = None
            self.frame = 0
            self.model = self

        def __call__(self, images, **kwargs):
            count = len(images)
            values = torch.arange(self.frame, self.frame + count, dtype=torch.float32)
            self.frame += count
            return {
                "pose_enc": torch.zeros(1, count, 9),
                "world_points": values.view(1, count, 1, 1, 1).expand(1, count, 2, 3, 3),
                "depth": (values + 1).view(1, count, 1, 1, 1).expand(1, count, 2, 3, 1),
                "aggregator_kv_cache_list": [],
                "camera_head_kv_cache_list": [],
            }

        def _update_cache(self, aggregator, camera):
            pass

    def pose_decoder(pose_enc, image_hw):
        count = pose_enc.shape[1]
        return torch.eye(4).repeat(1, count, 1, 1)[:, :, :3], None

    wrapper = SimpleNamespace(
        make_session=Session,
        chunk_size=2,
        image_loader=lambda paths, mode: torch.zeros(len(paths), 3, 2, 3),
        preprocess_mode="crop",
        mode="window",
        pose_decoder=pose_decoder,
    )
    cfg = SimpleNamespace(device="cpu", mv_recon_output_indices=[0, 2, 4])
    points, c2w, depth = MODULE._infer_stream3r(["a", "b", "c", "d", "e"], wrapper, cfg)
    assert points.shape == (3, 2, 3, 3)
    assert depth.shape == (3, 2, 3)
    assert c2w.shape == (5, 4, 4)
    assert points[:, 0, 0, 0].tolist() == [0.0, 2.0, 4.0]
