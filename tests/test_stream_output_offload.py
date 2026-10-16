"""Dense stream outputs must leave the GPU per frame.

``select_frame_outputs`` is the single funnel used by both streaming paths
(``inference_stream_iter`` for paged and sdpa, plus ``_inference_stream_paged``).
Dense point/confidence maps are only concatenated after the whole stream has
run, so keeping them on the GPU accumulates O(num_frames) VRAM and OOMs long
streams; this module pins that behaviour down.
"""

from __future__ import annotations

import pytest
import torch

from abot_recon.modeling.streaming.network import (
    DENSE_OUTPUT_KEYS,
    select_frame_outputs,
)


def _dense(device: str = "cpu") -> torch.Tensor:
    return torch.zeros(1, 1, 2, 2, 3, device=device)


def test_dense_output_keys_cover_points_and_confidence():
    assert set(DENSE_OUTPUT_KEYS) == {"points", "local_points", "conf", "global_points"}


def test_allowed_and_skip_state_filtering():
    pred = {
        "camera_poses": torch.zeros(1, 1, 4, 4),
        "points": _dense(),
        "local_points": _dense(),
        "conf": torch.zeros(1, 1, 2, 2, 1),
        "past_key_values": object(),
        "camera_state": {"rel": torch.zeros(1)},
    }
    kept = select_frame_outputs(
        pred,
        skip_state=("past_key_values", "camera_state"),
        allowed={"camera_poses", "points"},
    )
    assert set(kept) == {"camera_poses", "points"}


def test_dense_indices_restrict_dense_outputs_to_dense_frames():
    pred = {"camera_poses": torch.zeros(1, 1, 4, 4), "points": _dense()}
    sparse_frame = select_frame_outputs(pred, dense_indices={0, 5}, frame_index=3)
    dense_frame = select_frame_outputs(pred, dense_indices={0, 5}, frame_index=5)
    assert set(sparse_frame) == {"camera_poses"}
    assert set(dense_frame) == {"camera_poses", "points"}


def test_without_dense_indices_every_frame_keeps_its_dense_output():
    pred = {"points": _dense()}
    assert set(select_frame_outputs(pred, frame_index=1234)) == {"points"}


def test_none_and_non_tensor_values_pass_through():
    kept = select_frame_outputs({"conf": None, "points": None, "camera_poses": "kept"})
    assert kept == {"conf": None, "points": None, "camera_poses": "kept"}


def test_dense_outputs_are_detached():
    source = torch.zeros(1, 1, 2, 2, 3, requires_grad=True) * 2.0
    kept = select_frame_outputs({"points": source})
    assert kept["points"].requires_grad is False
    assert kept["points"].grad_fn is None


def test_non_dense_tensors_keep_their_device():
    pred = {"camera_poses": torch.zeros(1, 1, 4, 4), "points": _dense()}
    kept = select_frame_outputs(pred)
    assert kept["camera_poses"] is pred["camera_poses"]
    assert kept["points"] is not pred["points"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to prove the offload")
def test_cuda_dense_outputs_move_to_host_while_poses_stay_on_gpu():
    pred = {
        "points": _dense("cuda"),
        "conf": torch.zeros(1, 1, 2, 2, 1, device="cuda"),
        "camera_poses": torch.zeros(1, 1, 4, 4, device="cuda"),
    }
    kept = select_frame_outputs(pred, skip_state=("past_key_values",))
    assert kept["points"].device.type == "cpu"
    assert kept["conf"].device.type == "cpu"
    assert kept["points"].dtype == pred["points"].dtype
    assert kept["camera_poses"].device.type == "cuda"
