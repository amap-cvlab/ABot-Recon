import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from relpose.forward_timing import (
    materialize_for_forward_timing,
    reset_forward_timing,
    save_forward_timing,
    summarize_forward_timing,
    time_forward,
)
from interfaces import official_streaming


class FakeModel:
    pass


def test_disabled_timer_records_nothing():
    model = FakeModel()
    reset_forward_timing(model)
    with time_forward(model, {"measure_forward_fps": False}, num_frames=4):
        pass
    assert summarize_forward_timing(model, 4) is None


def test_chunk_timings_are_frame_weighted(monkeypatch):
    model = FakeModel()
    reset_forward_timing(model)
    ticks = iter([10.0, 10.2, 20.0, 20.3])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = {"measure_forward_fps": True, "device": "cpu"}
    with time_forward(model, cfg, num_frames=2, label="chunk"):
        pass
    with time_forward(model, cfg, num_frames=3, label="chunk"):
        pass
    summary = summarize_forward_timing(model, 5)
    assert summary["forward_calls"] == 2
    assert summary["forward_seconds"] == pytest.approx(0.5)
    assert summary["avg_forward_ms_per_frame"] == pytest.approx(100.0)
    assert summary["forward_fps"] == pytest.approx(10.0)


def test_summary_rejects_wrong_frame_count():
    model = FakeModel()
    model._relpose_forward_timing_samples = [
        {"label": "forward", "num_frames": 3, "seconds": 1.0}
    ]
    with pytest.raises(ValueError, match="frame mismatch"):
        summarize_forward_timing(model, 4)


def test_materialize_limit_and_json(tmp_path):
    cfg = {
        "measure_forward_fps": True,
        "forward_timing_preload_limit": 3,
    }
    assert materialize_for_forward_timing(iter(range(3)), cfg, num_frames=3, label="x") == [0, 1, 2]
    with pytest.raises(ValueError, match="exceeds"):
        materialize_for_forward_timing(iter(range(4)), cfg, num_frames=4, label="x")
    output = tmp_path / "timing.json"
    save_forward_timing(output, {"forward_fps": 12.5})
    assert json.loads(output.read_text())["forward_fps"] == 12.5


def test_official_recurrent_pose_path_records_all_frames(monkeypatch):
    class Recurrent:
        def inference(self, frames, frame_writer, cache_results):
            for index, frame in enumerate(frames):
                frame_writer(index, frame, {"camera_pose": torch.zeros(1, 9)})

    def loader(paths, mode):
        return torch.zeros(len(paths), 3, 4, 6)

    def decoder(pose_enc, image_hw):
        count = pose_enc.shape[1]
        return torch.eye(4).repeat(1, count, 1, 1)[:, :, :3], None

    wrapper = SimpleNamespace(
        family="infinitevggt",
        model=Recurrent(),
        image_loader=loader,
        preprocess_mode="crop",
        pose_decoder=decoder,
    )
    cfg = SimpleNamespace(
        device="cpu",
        measure_forward_fps=True,
        forward_timing_preload_limit=8,
    )
    monkeypatch.setattr(official_streaming.torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(official_streaming.torch.amp, "autocast", lambda *a, **k: nullcontext())
    reset_forward_timing(wrapper)
    poses = official_streaming._infer_infinite_or_ovggt_poses(
        ["a", "b", "c"], wrapper, cfg
    )
    assert poses.shape[0] == 3
    summary = summarize_forward_timing(wrapper, 3)
    assert summary["num_frames"] == 3
    assert summary["forward_calls"] == 1


def test_stream3r_pose_chunks_record_exact_frame_count():
    class SessionModel:
        def __call__(self, images, **_kwargs):
            count = images.shape[0]
            return {
                "depth": torch.ones(1, count, 4, 6),
                "pose_enc": torch.zeros(1, count, 9),
                "aggregator_kv_cache_list": [],
                "camera_head_kv_cache_list": [],
            }

    class Session:
        def __init__(self):
            self.model = SessionModel()
            self.aggregator_kv_cache_list = []
            self.camera_head_kv_cache_list = []
            self.predictions = None

        def _update_cache(self, aggregator, camera):
            self.aggregator_kv_cache_list = aggregator
            self.camera_head_kv_cache_list = camera

    def decoder(pose_enc, image_hw):
        count = pose_enc.shape[1]
        return torch.eye(4).repeat(1, count, 1, 1)[:, :, :3], None

    wrapper = SimpleNamespace(
        family="stream3r",
        chunk_size=2,
        mode="window",
        preprocess_mode="crop",
        image_loader=lambda paths, mode: torch.zeros(len(paths), 3, 4, 6),
        make_session=Session,
        pose_decoder=decoder,
    )
    cfg = SimpleNamespace(device="cpu", measure_forward_fps=True)
    reset_forward_timing(wrapper)
    poses = official_streaming._infer_stream3r_poses(
        ["a", "b", "c", "d", "e"], wrapper, cfg
    )
    assert poses.shape[0] == 5
    summary = summarize_forward_timing(wrapper, 5)
    assert summary["num_frames"] == 5
    assert summary["forward_calls"] == 3
