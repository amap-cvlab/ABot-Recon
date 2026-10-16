# Changelog

## Unreleased

### Fixed

- **Streaming dense outputs no longer accumulate in GPU memory**
  (`abot_recon/modeling/streaming/network.py`). Per-frame dense point and
  confidence maps used to be held on the GPU until the whole stream finished, so
  VRAM grew linearly (~4 MB/frame) with the number of frames. Long dense runs
  therefore ran out of memory: a 1403-frame run hit `CUDA out of memory` around
  frame 334 on an 8 GB GPU. `select_frame_outputs` now moves each frame's dense
  maps to host memory as soon as that frame is done, so VRAM stays flat and the
  final stacking concatenates on CPU. Both streaming funnels are covered
  (`inference_stream_iter` for the paged and sdpa paths, plus
  `_inference_stream_paged`).

- **RoPE3D positional tables are sized for the patch grid**
  (`abot_recon/modeling/streaming/rope3d.py`). Height and width coordinates are
  indexed by `patch_start_idx + patch_grid`, which can exceed `max_seq_len`
  (5 + 36 = 41 rows for a 280x504 input), so the table is now grown on demand.
  Small `max_frames` values used to fail with
  `shape '[1, 1, 36, -1]' is invalid for input of size 385`. Growing the table
  does not change the values of the rows that already existed.

- **Out-of-range temporal RoPE fails loudly instead of silently narrowing**
  (`abot_recon/modeling/streaming/rope3d.py`). A frame index at or beyond the
  configured stream length produced an empty temporal slice, which silently
  dropped the temporal RoPE columns (width 22 instead of 32) and only surfaced
  later as `Sizes of tensors must match except in dimension 2`. It now raises a
  `RuntimeError` naming the frame index and the configured stream length.

### Changed

- **Streams longer than `max_frames` are rejected before inference**
  (`abot_recon/model.py`). `max_frames` is a capacity (RoPE positional table and
  paged KV), not a frame cap, and never truncated the input: a longer stream used
  to fail mid-run after minutes of work. It now raises a `ValueError` up front.
  The `--max-frames` help text (CLI and `demo.py`) and the README tables state
  that it must cover the number of selected frames, and that fewer frames are
  selected with `--start`/`--end`/`--stride`.

### Tests

- `tests/test_rope3d_streaming.py`: table sizing for temporal and spatial
  coordinates, growth correctness, and the out-of-range guards.
- `tests/test_stream_output_offload.py`: dense-map offload, filtering and
  detection semantics of `select_frame_outputs`.
