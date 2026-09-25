"""LingBot-Map style foldback sampler for long-sequence streaming training.

Reference: "Geometric Context Transformer for Streaming 3D Reconstruction"
(arXiv:2604.14141), Section 4.3 - Foldback Video Sampler.

Quote from the paper:
    "The sampler starts at a random frame and advances with a random stride.
    Upon reaching a sequence boundary, it reverses direction and draws a new
    stride (distinct from the previous one) to avoid degenerate oscillation.
    This mechanism yields subsequences with naturally varying frame rates
    and no forward-time bias."

This implementation extends the paper in two practical ways:

1. `forward_only` flag (default False): when True, the sample is a single
   contiguous forward leg with NO direction reversal and NO jumps.
   Stride is drawn ONCE from the feasibility-clamped range
   [min_interval, min(max_interval, remaining // (num_views - 1))]
   so the entire walk fits without hitting the right boundary. This
   guarantees an uninterrupted video-style sequence, which is the
   correct behavior for datasets where time-reversed playback is
   unrealistic (e.g. driving data: cars don't drive backward; lighting
   / brake-light patterns become un-physical when time-reversed) AND
   for any setting where jump-to-fresh-start would create artificial
   discontinuities the model shouldn't see. The outer dataset loader is
   expected to filter `start_img_ids` so each start has at least
   `num_views - 1` frames of headroom (i.e. stride=1 is feasible).
   For synthetic worlds where reverse playback is fine, leave default
   False to enable multi-stride per-sample diversity via foldback.

2. `recent_stride_memory` (default 2): when drawing a stride at a fold,
   exclude the last K strides instead of just the immediately previous
   one. The paper's "distinct from previous one" rule prevents perfect
   2-frame oscillation but still allows alternating patterns like
   [4, 8, 4, 8, ...]. Setting K=2 also blocks alternating-2 patterns,
   which is cheap (we typically have 8-15 candidate strides) and slightly
   improves diversity. Set K=1 to match the paper exactly.

Why foldback over CUT3R-style fixed-stride for long-sequence (24~150 frame)
streaming training:
  1. Multi-stride per sample: a single training window contains several legs
     with different strides, exposing the model to varying motion magnitudes
     within one sequence (cf. real cameras that change speed mid-shot).
  2. Decouples max_interval from sequence length: fixed-stride sampling on a
     401-frame video at N=150 caps stride at 2; foldback uses the full
     [min, max] range by bouncing off the boundaries.
  3. "Distinct from recent K" stride at each fold prevents the trivial
     symmetric-oscillation shortcut a model could otherwise learn.

Returned positions preserve sampling order. Time is NOT monotonic across
folds in default mode; in `forward_only` mode time is locally monotonic
within each leg but jumps between legs.

Function name and file name use the `_long` suffix to signal that this is
the long-sequence variant; the original `get_seq_from_start_id` in
`base/base_multiview_dataset.py` is left untouched.
"""
from __future__ import annotations

import collections

import numpy as np


def _draw_log_uniform_stride_long(min_interval, max_interval, exclude, rng, uniform=True):
    """Draw a stride from [min_interval, max_interval] log-uniformly.

    Log-uniform (weights ~ 1/s) matches multiplicative perception of
    baselines: stride 1->2 and 8->16 represent comparable difficulty
    increments, so uniform sampling would over-sample large strides.

    `exclude` may be:
        - None: no exclusion.
        - A single int: that single stride is excluded.
        - An iterable of ints: all listed strides are excluded.

    If exclusion would empty the candidate set, we silently fall back to
    the full set rather than crashing. This implements the
    "distinct from recent K" rule with graceful degradation when the
    range is so small that exclusion runs out of candidates.
    """
    candidates = list(range(int(min_interval), int(max_interval) + 1))
    if not candidates:
        return max(1, int(min_interval))
    if exclude is not None:
        if isinstance(exclude, (int, np.integer)):
            ex_set = {int(exclude)}
        else:
            ex_set = {int(e) for e in exclude}
        filtered = [c for c in candidates if c not in ex_set]
        if filtered:
            candidates = filtered
    if uniform:
        weights = np.asarray([1.0 for s in candidates], dtype=np.float64) # uniform: weight ~ const
    else:
        weights = np.asarray([1.0 / s for s in candidates], dtype=np.float64) # log-uniform: weight ~ 1/stride
    
    weights = weights / weights.sum()
    return int(rng.choice(candidates, p=weights))


def foldback_from_start_long(
    num_views,
    id_ref,
    ids_all,
    rng,
    min_interval=1,
    max_interval=8,
    forward_only=False,
    recent_stride_memory=4,
    fix_interval_prob=0.5,
):
    """LingBot-Map foldback sampler.

    Two modes:

    * forward_only=False (default, foldback): walks the cursor; each leg
      uses a single stride drawn log-uniformly from [min_interval,
      max_interval]; on hitting a sequence boundary the cursor reverses
      direction and draws a new stride distinct from the last
      `recent_stride_memory` strides. Multi-stride per sample.

    * forward_only=True (CUT3R-style single forward leg, no breaks):
      monotone-in-time forward walk with NO reversal and NO jumps. Two
      sub-modes selected per sample by `fix_interval_prob`:
        - fixed-stride (with prob fix_interval_prob): one stride drawn
          log-uniformly from [min_interval, feasible_max] used for ALL
          pairs in the walk. Clean uniform-rate video.
        - variable-stride (with prob 1 - fix_interval_prob): each pair
          draws its own stride log-uniformly from [min_interval,
          min(max_interval, 2 * feasible_avg)]. Cumulative position can
          occasionally overflow; overflow positions are dropped and
          replaced by random valid frames from [id_ref, seq_len), then
          sorted to keep monotone time. This mirrors CUT3R's behavior
          and produces variable-rate but uninterrupted video.
      The outer layer must guarantee enough headroom from `id_ref`
      (>= num_views - 1 frames). `recent_stride_memory` is ignored.

    Args:
        num_views: number of indices to return.
        id_ref: starting global frame id (used as the initial cursor
            position; foldback walks from here).
        ids_all: list of all global frame ids in the scene.
        rng: numpy Generator for randomness.
        min_interval, max_interval: per-step stride bounds. Clamped to
            [1, len(ids_all)-1] internally. In forward_only mode,
            max_interval is further clamped to feasibility from id_ref.
        forward_only: if True, monotone forward leg with no folds/jumps.
            Default False (matches the LingBot-Map paper).
        recent_stride_memory: foldback-only. How many recent strides to
            exclude when drawing a new stride at a fold. Default 2.
            Ignored in forward_only mode.
        fix_interval_prob: forward_only-only. Probability of using a
            single fixed stride for the entire walk; otherwise per-pair
            random strides. Default 0.5. Ignored in foldback mode.

    Returns:
        (positions, is_video):
            - positions: list of indices INTO ids_all, length == num_views,
              preserved in sampling (foldback) order. Do NOT sort downstream
              (already sorted in forward_only mode).
            - is_video: always True. Both modes produce a temporally
              ordered sequence; setting True skips the base class's
              block-shuffle path.
    """
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    seq_len = len(ids_all)
    if seq_len == 0:
        raise ValueError("ids_all is empty")
    if seq_len == 1:
        return [0] * int(num_views), True

    # Sanitize bounds against the actual scene length.
    max_interval = max(1, min(int(max_interval), seq_len - 1))
    min_interval = max(1, min(int(min_interval), max_interval))

    if id_ref not in ids_all:
        raise ValueError("id_ref not in ids_all")
    pos_ref = ids_all.index(id_ref)

    # ------------------------------------------------------------------
    # forward_only: CUT3R-style monotone forward walk. Two sub-modes:
    #   1. fixed-stride (prob fix_interval_prob): one feasibility-clamped
    #      stride for the entire walk. Guaranteed in-bounds.
    #   2. variable-stride (prob 1 - fix_interval_prob): each pair's
    #      stride drawn independently. Cumulative position may overflow;
    #      overflowing positions are dropped and replaced by random
    #      valid frames from [pos_ref, seq_len), then sorted.
    # No reversal, no jumps in either sub-mode.
    # ------------------------------------------------------------------
    if forward_only:
        if num_views == 1:
            return [pos_ref], True
        remaining = seq_len - 1 - pos_ref
        # Per-pair feasibility for FIXED stride: pos_ref + (N-1)*s <= seq_len-1
        feasible_fixed_max = max(1, remaining // (num_views - 1))

        if rng.random() < float(fix_interval_prob):
            # Fixed-stride sub-mode: one stride for entire walk.
            eff_max = min(int(max_interval), feasible_fixed_max)
            eff_min = max(1, min(int(min_interval), eff_max))
            stride = _draw_log_uniform_stride_long(eff_min, eff_max, None, rng, uniform=True)
            positions = [pos_ref + i * stride for i in range(num_views)]
            positions = [min(p, seq_len - 1) for p in positions]
            return positions, True

        # Variable-stride sub-mode. CUT3R convention: cap per-pair stride
        # at 2x the average feasible stride so the AVERAGE walk fits.
        # Individual pairs may exceed; overflow is handled below.
        var_max = min(int(max_interval), max(1, 2 * feasible_fixed_max))
        var_min = max(1, min(int(min_interval), var_max))
        intervals = [
            _draw_log_uniform_stride_long(var_min, var_max, None, rng, uniform=True)
            for _ in range(num_views - 1)
        ]
        cum = pos_ref
        cumulative = [pos_ref]
        for s in intervals:
            cum = cum + s
            cumulative.append(cum)
        valid = [p for p in cumulative if p <= seq_len - 1]
        if len(valid) < num_views:
            # Overflow: fill missing slots with random valid frames in
            # [pos_ref, seq_len) that are not yet selected, then sort to
            # keep monotone time.
            n_missing = num_views - len(valid)
            valid_set = set(valid)
            available = [p for p in range(pos_ref, seq_len) if p not in valid_set]
            if len(available) >= n_missing:
                fills = rng.choice(available, n_missing, replace=False).tolist()
                fills = [int(f) for f in fills]
                positions = sorted(valid + fills)
            else:
                # Should not happen given outer-layer guarantees; pad
                # gracefully with the last valid position.
                positions = sorted(valid)
                while len(positions) < num_views:
                    positions.append(positions[-1] if positions else pos_ref)
        else:
            positions = sorted(valid)
        positions = [min(p, seq_len - 1) for p in positions[:num_views]]
        return positions, True

    # ------------------------------------------------------------------
    # Standard foldback (forward_only=False): cursor walks; reverse at
    # boundary; new stride distinct from the last K strides.
    # ------------------------------------------------------------------
    if pos_ref == 0:
        direction = 1
    elif pos_ref == seq_len - 1:
        direction = -1
    else:
        direction = 1  # walk forward by default

    recent_strides = collections.deque(maxlen=max(1, int(recent_stride_memory)))
    stride = _draw_log_uniform_stride_long(min_interval, max_interval, None, rng)
    recent_strides.append(stride)

    pos = pos_ref
    positions = [pos]
    # Hard safety cap; tens of folds is plenty for any reasonable num_views.
    safety = max(num_views * 50, 1000)

    while len(positions) < num_views and safety > 0:
        safety -= 1
        next_pos = pos + direction * stride
        if 0 <= next_pos < seq_len:
            pos = next_pos
            positions.append(pos)
            continue

        # Boundary: reverse direction, draw a new stride distinct from
        # the recent ones.
        direction = -direction
        stride = _draw_log_uniform_stride_long(
            min_interval, max_interval, list(recent_strides), rng
        )
        recent_strides.append(stride)
        next_pos = pos + direction * stride
        # Pathological case (very short sequence, large stride): clamp
        # the cursor inside the valid range so we still make progress.
        if not (0 <= next_pos < seq_len):
            next_pos = max(0, min(seq_len - 1, next_pos))
        pos = next_pos
        positions.append(pos)

    # Pad if safety ran out (should not happen on reasonable inputs).
    while len(positions) < num_views:
        positions.append(positions[-1])

    return positions[: int(num_views)], True
