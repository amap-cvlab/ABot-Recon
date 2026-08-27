from types import SimpleNamespace
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from PIL import Image

from interfaces.lingbot_map import (
    LazyLingBotImages,
    _assert_expected_input_hw,
    _load_long_pose_official,
)
from interfaces import lingbot_map as lingbot_interface
from interfaces.longstream import infer_cameras_c2w as infer_longstream_c2w
from interfaces.horizonstream import LazyHorizonImages, run_horizon_loop_from_c2w
from interfaces import horizonstream as horizon_interface
from interfaces.abot_recon import _run_stream_inference
from interfaces import abot_recon as abot_recon_interface
from interfaces.ttt3r import _forward_recurrent_predictions
from interfaces import ttt3r as ttt3r_interface
from interfaces.ttt3r import _poses_from_predictions
from models.fastmodel import (
    _DiscardedHorizonDepthHead,
    _horizon_pose_only_depth_readout,
    _temporarily_disable_dense_heads,
)
from relpose.forward_timing import reset_forward_timing, summarize_forward_timing


def _write_rgb(path, size, color=(32, 64, 96)):
    Image.new("RGB", size, color=color).save(path)
    return str(path)


def test_lingbot_released_long_pose_shapes_are_dataset_specific(tmp_path):
    wide = _write_rgb(tmp_path / "wide.png", (1241, 376))
    model = SimpleNamespace(
        eval_dataset_name="kitti-long", patch_size=14, img_size=518
    )
    assert _load_long_pose_official([wide], model, "cpu").shape == (1, 1, 3, 280, 504)

    model.eval_dataset_name = "vbr-long"
    assert _load_long_pose_official([wide], model, "cpu").shape == (1, 1, 3, 280, 504)

    oxford = _write_rgb(tmp_path / "oxford.png", (1440, 1080))
    model.eval_dataset_name = "oxford_spires_processed-long"
    assert _load_long_pose_official([oxford], model, "cpu").shape == (
        1,
        1,
        3,
        378,
        518,
    )


def test_lingbot_official_long_pose_shape_guard_is_dataset_specific():
    cfg = DictConfig({})
    model = SimpleNamespace(
        preprocess_mode="long_pose_official",
        eval_dataset_name="kitti-long",
    )
    _assert_expected_input_hw(torch.zeros(1, 2, 3, 280, 504), cfg, model)

    model.eval_dataset_name = "vbr-long"
    _assert_expected_input_hw(torch.zeros(1, 2, 3, 280, 504), cfg, model)

    model.eval_dataset_name = "oxford_spires_processed-long"
    _assert_expected_input_hw(torch.zeros(1, 2, 3, 378, 518), cfg, model)

    with pytest.raises(RuntimeError, match="profile=oxford"):
        _assert_expected_input_hw(torch.zeros(1, 2, 3, 280, 504), cfg, model)


def test_lingbot_released_long_pose_is_lazy_and_slice_equivalent(tmp_path):
    wide = _write_rgb(tmp_path / "wide.png", (1241, 376))
    model = SimpleNamespace(
        eval_dataset_name="kitti-long", patch_size=14, img_size=518
    )
    eager = _load_long_pose_official([wide, wide], model, "cpu")
    lazy = _load_long_pose_official([wide] * 65, model, "cpu")

    assert isinstance(lazy, LazyLingBotImages)
    assert lazy.shape == (1, 65, 3, 280, 504)
    assert lazy.device == torch.device("cpu")
    assert torch.equal(lazy[:, :2], eager)
    assert lazy[:, 64:65].shape == (1, 1, 3, 280, 504)


def test_lingbot_pose_adapter_requests_pose_only(monkeypatch):
    class FakeLingBot:
        def __init__(self):
            self.pose_only = None

        def __call__(self, images, dense_output_indices=None, pose_only=False):
            self.pose_only = pose_only
            return {"pose_enc": torch.zeros(1, images.shape[1], 9)}

    model = FakeLingBot()
    images = torch.zeros(1, 2, 3, 8, 12)
    monkeypatch.setattr(lingbot_interface, "_load_and_preprocess", lambda *a, **k: images)
    monkeypatch.setattr(lingbot_interface, "_assert_expected_input_hw", lambda *a, **k: None)
    monkeypatch.setattr(lingbot_interface, "_record_runtime", lambda *a, **k: None)
    monkeypatch.setattr(
        lingbot_interface,
        "pose_encoding_to_extri_intri",
        lambda pose, image_size_hw: (
            torch.eye(4).repeat(1, pose.shape[1], 1, 1)[:, :, :3],
            torch.eye(3).repeat(1, pose.shape[1], 1, 1),
        ),
    )

    poses, intrinsics = lingbot_interface.infer_cameras_c2w(
        ["a", "b"], model, DictConfig({"device": "cpu"})
    )
    assert model.pose_only is True
    assert poses.shape == (2, 3, 4)
    assert intrinsics.shape == (2, 3, 3)


def test_longstream_pose_adapter_requests_pose_only():
    class FakeLongStream:
        img_size = 518
        patch_size = 14

        def __init__(self):
            self.pose_only = None

        def image_loader(self, filelist, **kwargs):
            return [{"img": torch.zeros(1, 3, 8, 12)} for _ in filelist]

        def __call__(self, images, pose_only=False):
            self.pose_only = pose_only
            count = images.shape[1]
            w2c = torch.eye(4).repeat(count, 1, 1)[:, :3]
            return {
                "extrinsic_w2c": w2c,
                "intrinsic": torch.eye(3).repeat(count, 1, 1),
            }

    model = FakeLongStream()
    poses, intrinsics = infer_longstream_c2w(
        ["a", "b", "c"], model, DictConfig({"device": "cpu"})
    )
    assert model.pose_only is True
    assert poses.shape == (3, 3, 4)
    assert intrinsics.shape == (3, 3, 3)


def test_pose_only_dense_heads_are_disabled_and_restored():
    point_head = object()
    depth_head = object()
    module = SimpleNamespace(point_head=point_head, depth_head=depth_head)

    with _temporarily_disable_dense_heads(module):
        assert module.point_head is None
        assert module.depth_head is None

    assert module.point_head is point_head
    assert module.depth_head is depth_head


def test_pose_only_dense_heads_restore_after_failure():
    depth_head = object()
    module = SimpleNamespace(depth_head=depth_head)

    try:
        with _temporarily_disable_dense_heads(module):
            raise RuntimeError("expected")
    except RuntimeError:
        pass

    assert module.depth_head is depth_head


def test_horizon_pose_only_depth_readout_is_shape_compatible_and_restored():
    original = object()
    core = SimpleNamespace(dpt_decoder=original)
    wrapper = SimpleNamespace(horizonstream=core)
    images = torch.zeros(1, 3, 3, 8, 12)

    with _horizon_pose_only_depth_readout(wrapper):
        assert isinstance(core.dpt_decoder, _DiscardedHorizonDepthHead)
        depth, confidence = core.dpt_decoder([], images=images)
        assert depth.shape == (1, 3, 1, 1, 1)
        assert confidence.shape == (1, 3, 1, 1)

    assert core.dpt_decoder is original


def test_horizon_lazy_images_support_loop_window_advanced_indices(tmp_path):
    paths = [
        _write_rgb(tmp_path / f"horizon-{index}.png", (64, 48), color=(index, 4, 8))
        for index in range(6)
    ]
    images = LazyHorizonImages(paths, img_size=56, patch_size=14, crop=True, device="cpu")

    selected = images[:, [4, 1, 5]]

    assert selected.shape[0] == 1
    assert selected.shape[1] == 3
    assert selected.shape[-2:] == images.shape[-2:]


def test_horizon_loop_uses_cached_pose_and_emits_independent_variant(
    monkeypatch, tmp_path
):
    try:
        from horizonstream.loop import online_loop_reinfer
        from horizonstream.loop import runtime as loop_runtime
    except ImportError as error:
        pytest.skip(f"optional HorizonStream loop dependencies are unavailable: {error}")

    class FakeImages:
        shape = (1, 4, 3, 8, 12)

    candidate = loop_runtime.LoopCandidate(
        src_pos=3,
        dst_pos=0,
        src_frame=3,
        dst_frame=0,
        score=0.9,
        method="salad_online",
    )
    edge = loop_runtime.LoopEdge(
        src_pos=3,
        dst_pos=0,
        src_frame=3,
        dst_frame=0,
        score=0.9,
        inliers=8,
        method="salad_online_reinfer",
        transform_ji=np.eye(4),
    )

    monkeypatch.setattr(horizon_interface, "_load_and_preprocess", lambda *a, **k: FakeImages())
    monkeypatch.setattr(
        horizon_interface,
        "_retrieve_horizon_salad_candidates_lazy",
        lambda *a, **k: [candidate],
    )

    class FakeRefiner:
        def __init__(self, **kwargs):
            pass

        def refine(self, received):
            assert received is candidate
            return edge

    monkeypatch.setattr(online_loop_reinfer, "LoopReinferRefiner", FakeRefiner)
    monkeypatch.setattr(online_loop_reinfer, "_build_loop_cfg", lambda cfg: object())
    monkeypatch.setattr(
        loop_runtime, "build_keyframe_odometry_edges", lambda poses: ["odom"]
    )

    def fake_optimize(base, odom_edges, loop_edges, cfg):
        assert odom_edges == ["odom"]
        assert loop_edges == [edge]
        optimized = base.copy()
        optimized[:, 0, 3] += 2.0
        return optimized

    monkeypatch.setattr(loop_runtime, "optimize_keyframe_pose_graph", fake_optimize)
    monkeypatch.setattr(
        loop_runtime,
        "save_loop_edges_json",
        lambda path, edges: Path(path).write_text("[]", encoding="utf-8"),
    )

    base = torch.eye(4).repeat(4, 1, 1)
    result, metadata = run_horizon_loop_from_c2w(
        ["a", "b", "c", "d"],
        base,
        SimpleNamespace(model=object(), img_size=518, patch_size=14, crop=True),
        DictConfig({"device": "cpu"}),
        {
            "methods": ["salad"],
            "max_candidates": 5,
            "max_reinfer_candidates": 1,
            "min_frame_separation": 1,
        },
        str(tmp_path),
    )

    assert torch.equal(result[:, 0, 3], torch.full((4,), 2.0))
    assert metadata["num_candidates"] == 1
    assert metadata["num_loop_edges"] == 1
    assert (tmp_path / "candidates.json").is_file()
    assert (tmp_path / "loop_edges.json").is_file()


def test_abot_recon_pose_path_streams_one_preprocessed_frame_at_a_time(tmp_path):
    paths = [
        _write_rgb(tmp_path / f"{index}.png", (640, 360), color=(index, 2, 3))
        for index in range(4)
    ]

    class FakeABotRecon:
        local_window_override = None
        num_frames_cap = None
        fov_pad_rgb = [0.485, 0.456, 0.406]
        height = 280
        width = 504
        amp_dtype = "fp32"
        infer_mode = "stream"

        def __init__(self):
            self.model = SimpleNamespace(causal_global_attn=True)
            self.seen_shapes = []

        def eval(self):
            return self

        def inference_stream_iter(self, frames, num_frames, **kwargs):
            assert kwargs["output_keys"] == ("camera_poses",)
            for frame in frames:
                self.seen_shapes.append(tuple(frame.shape))
            assert len(self.seen_shapes) == num_frames
            return {"camera_poses": torch.eye(4).repeat(1, num_frames, 1, 1)}

    model = FakeABotRecon()
    output, _ = _run_stream_inference(
        paths, model, DictConfig({"device": "cpu"}), need_points=False
    )
    assert set(output) == {"camera_poses"}
    assert model.seen_shapes == [(1, 1, 3, 280, 504)] * len(paths)

def test_abot_recon_mvrecon_requests_direct_and_local_world_points(tmp_path):
    paths = [_write_rgb(tmp_path / "0.png", (640, 360))]

    class FakeABotRecon:
        local_window_override = None
        num_frames_cap = None
        fov_pad_rgb = [0.485, 0.456, 0.406]
        height = 280
        width = 504
        amp_dtype = "fp32"
        infer_mode = "stream"

        def __init__(self):
            self.model = SimpleNamespace(causal_global_attn=True)

        def eval(self):
            return self

        def inference_stream_iter(self, frames, num_frames, **kwargs):
            assert kwargs["output_keys"] == (
                "camera_poses",
                "points",
                "local_points",
            )
            list(frames)
            points = torch.zeros(1, num_frames, 280, 504, 3)
            return {
                "camera_poses": torch.eye(4).repeat(1, num_frames, 1, 1),
                "points": points,
                "local_points": points.clone(),
            }

    output, _ = _run_stream_inference(
        paths, FakeABotRecon(), DictConfig({"device": "cpu"}), need_points=True
    )
    assert set(output) == {"camera_poses", "points", "local_points"}


@pytest.mark.parametrize("loop_enabled", [False, True])
def test_abot_recon_pose_is_camera_only_before_optional_loop(monkeypatch, loop_enabled):
    base = torch.eye(4).repeat(3, 1, 1)
    looped = base.clone()
    looped[:, 0, 3] = 2.0
    calls = []

    def fake_forward(*args, **kwargs):
        assert kwargs["need_points"] is False
        return {"camera_poses": base.unsqueeze(0)}, None

    def fake_loop(paths, poses, model, cfg, **kwargs):
        calls.append((paths, poses.clone()))
        return looped

    monkeypatch.setattr(abot_recon_interface, "_run_stream_inference", fake_forward)
    monkeypatch.setattr(abot_recon_interface, "run_abot_recon_loop_from_c2w", fake_loop)
    result, intrinsics = abot_recon_interface.infer_cameras_c2w(
        ["a", "b", "c"], object(), DictConfig({"abot_recon_loop_enabled": loop_enabled})
    )
    assert intrinsics is None
    assert torch.equal(result, looped if loop_enabled else base)
    assert len(calls) == int(loop_enabled)

def test_ttt3r_dispatches_to_attention_gated_lighter_path():
    class FakeRecurrent:
        def __init__(self, update_type):
            self.config = SimpleNamespace(model_update_type=update_type)
            self.calls = []

        def forward_recurrent(self, views, device, ret_state):
            self.calls.append(("cut3r", list(views), device, ret_state))
            return ["cut-pred"], None

        def forward_recurrent_lighter(self, views, device, ret_state):
            self.calls.append(("ttt3r", list(views), device, ret_state))
            return ["ttt-pred"], None

    ttt = FakeRecurrent("ttt3r")
    assert _forward_recurrent_predictions(ttt, iter([1, 2]), "cuda") == ["ttt-pred"]
    assert ttt.calls == [("ttt3r", [1, 2], "cuda", False)]

    cut = FakeRecurrent("cut3r")
    assert _forward_recurrent_predictions(cut, iter([3]), "cuda") == ["cut-pred"]
    assert cut.calls == [("cut3r", [3], "cuda", False)]


def test_ttt3r_dispatch_rejects_unknown_update_rule():
    model = SimpleNamespace(config=SimpleNamespace(model_update_type="unknown"))
    try:
        _forward_recurrent_predictions(model, iter(()), "cpu")
    except ValueError as error:
        assert "model_update_type" in str(error)
    else:
        raise AssertionError("unknown state-update rule was silently accepted")


def test_cut3r_ttt3r_noreset_preserves_every_pose():
    predictions = [
        {"camera_pose": torch.eye(4).unsqueeze(0)} for _ in range(4)
    ]
    for index, prediction in enumerate(predictions):
        prediction["camera_pose"][0, 0, 3] = float(index)

    poses = _poses_from_predictions(
        predictions,
        reset_flags=[False] * 4,
        pose_decoder=lambda pose: pose,
    )

    assert poses.shape == (4, 4, 4)
    assert torch.equal(poses[:, 0, 3], torch.arange(4, dtype=torch.float32))


def test_cut3r_ttt3r_reset_inserts_overlap_after_each_boundary(monkeypatch):
    def fake_prepare_input(paths, size, device):
        assert len(paths) == 1
        return [{"img": torch.zeros(1, 3, 8, 12)}]

    monkeypatch.setattr(ttt3r_interface, "prepare_input", fake_prepare_input)
    model = SimpleNamespace(input_size=512)
    views = list(
        ttt3r_interface._iter_prepared_views(
            [f"frame-{index}.png" for index in range(5)],
            model,
            "cpu",
            reset_interval=2,
        )
    )

    assert [view["idx"] for view in views] == list(range(7))
    assert [bool(view["reset"].item()) for view in views] == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_cut3r_ttt3r_reset_drops_overlap_and_stitches_segments():
    # Original frames are 0,1,3,4. Frames 2 and 5 are overlap copies emitted
    # immediately after reset frames and must not appear in the trajectory.
    translations = [0.0, 1.0, 0.0, 1.0, 2.0, 0.0]
    predictions = []
    for translation in translations:
        pose = torch.eye(4).unsqueeze(0)
        pose[0, 0, 3] = translation
        predictions.append({"camera_pose": pose})

    poses = _poses_from_predictions(
        predictions,
        reset_flags=[False, True, False, False, True, False],
        pose_decoder=lambda pose: pose,
    )

    assert poses.shape == (4, 4, 4)
    assert torch.equal(poses[:, 0, 3], torch.arange(4, dtype=torch.float32))


def test_ttt3r_fps_path_uses_recurrent_rollout_after_preload(monkeypatch):
    wrapper = SimpleNamespace(model=object(), input_size=512)
    prepared = [
        {
            "img": torch.zeros(1, 3, 8, 12),
            "reset": torch.tensor([False]),
            "idx": index,
        }
        for index in range(3)
    ]
    seen = {}

    monkeypatch.setattr(ttt3r_interface, "pose_only_downstream_head", lambda _net: nullcontext())
    monkeypatch.setattr(ttt3r_interface, "prepare_input", lambda *a, **k: prepared)

    def recurrent(_net, views, device):
        seen["views"] = list(views)
        seen["device"] = device
        return [object()] * len(seen["views"])

    monkeypatch.setattr(ttt3r_interface, "_forward_recurrent_predictions", recurrent)
    monkeypatch.setattr(
        ttt3r_interface,
        "_poses_from_predictions",
        lambda predictions, reset_flags=None: torch.eye(4).repeat(
            len(predictions), 1, 1
        )[:, :3],
    )
    reset_forward_timing(wrapper)
    poses = ttt3r_interface._infer_pose_only_c2w(
        ["a", "b", "c"],
        wrapper,
        DictConfig({"device": "cpu", "measure_forward_fps": True}),
    )
    assert poses.shape == (3, 3, 4)
    assert seen == {"views": prepared, "device": "cpu"}
    timing = summarize_forward_timing(wrapper, 3)
    assert timing["num_frames"] == 3
    assert timing["forward_calls"] == 1
