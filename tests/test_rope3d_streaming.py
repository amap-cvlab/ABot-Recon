"""Regression tests for the streaming RoPE3D positional tables.

Covers two long-stream failures that used to surface as opaque tensor-shape
errors:

* the H/W coordinates are indexed by ``patch_start_idx + patch_grid``, so they
  can need more rows than ``max_seq_len`` (e.g. 5 + 36 = 41 rows for a 280x504
  input) -- the table must grow for those rows;
* a frame index at or beyond the configured stream length used to produce an
  empty temporal slice, silently dropping the temporal RoPE columns (width 22
  instead of 32) instead of failing.
"""

from __future__ import annotations

import pytest
import torch

from abot_recon.modeling.streaming.rope3d import RoPE3D


CPU = torch.device("cpu")
# 280x504 input -> 20x36 patch grid, plus 5 register tokens per frame.
PATCH_START_IDX = 5
PATCHES_H = 20
PATCHES_W = 36
TOKENS_PER_FRAME = PATCH_START_IDX + PATCHES_H * PATCHES_W
ROPE_WIDTH = 32  # (t_dim + h_dim + w_dim) / 2 for fhw_dim=(20, 22, 22)


def _rope(max_seq_len: int) -> RoPE3D:
    return RoPE3D(
        attention_head_dim=64,
        patch_size=(1, 14, 14),
        max_seq_len=max_seq_len,
        theta=10_000.0,
        fhw_dim=(20, 22, 22),
    )


def _one_frame(rope: RoPE3D, frame_idx: int) -> torch.Tensor:
    return rope(
        1,
        PATCHES_H,
        PATCHES_W,
        PATCH_START_IDX,
        CPU,
        f_start=frame_idx,
        f_end=frame_idx + 1,
    )


def test_temporal_coordinates_fit_the_configured_stream_length():
    rope = _rope(40)
    assert int(rope.freqs.shape[0]) == 40
    assert tuple(_one_frame(rope, 39).shape) == (1, 1, TOKENS_PER_FRAME, ROPE_WIDTH)


def test_spatial_coordinates_grow_the_table_beyond_max_seq_len():
    rope = _rope(40)
    assert tuple(_one_frame(rope, 0).shape) == (1, 1, TOKENS_PER_FRAME, ROPE_WIDTH)
    # Needs rows 5..41 for W, but the temporal limit stays at 40.
    assert int(rope.freqs.shape[0]) == PATCH_START_IDX + PATCHES_W == 41


def test_growth_does_not_change_existing_rows():
    reference = _rope(64)
    grown = _rope(40)
    _one_frame(grown, 0)
    rows = int(grown.freqs.shape[0])
    assert rows == 41
    assert torch.equal(grown.freqs[:rows], reference.freqs[:rows])
    for frame_idx in (0, 17, 39):
        assert torch.equal(_one_frame(grown, frame_idx), _one_frame(reference, frame_idx))


def test_small_spatial_grid_keeps_the_configured_table():
    rope = _rope(40)
    # patch_start_idx=0 needs only max(pph, ppw) = 36 rows.
    rope(1, PATCHES_H, PATCHES_W, 0, CPU, f_start=0, f_end=1)
    assert int(rope.freqs.shape[0]) == 40


def test_frame_at_stream_length_raises_instead_of_narrowing_freqs():
    rope = _rope(200)
    with pytest.raises(RuntimeError, match="out of range"):
        _one_frame(rope, 200)


def test_dense_range_beyond_stream_length_raises():
    rope = _rope(40)
    with pytest.raises(RuntimeError, match="stream length"):
        rope(41, PATCHES_H, PATCHES_W, PATCH_START_IDX, CPU, f_start=0, f_end=None)


def test_dense_range_at_the_limit_is_allowed():
    rope = _rope(40)
    freqs = rope(40, PATCHES_H, PATCHES_W, PATCH_START_IDX, CPU, f_start=0, f_end=None)
    assert tuple(freqs.shape) == (1, 1, 40 * TOKENS_PER_FRAME, ROPE_WIDTH)
