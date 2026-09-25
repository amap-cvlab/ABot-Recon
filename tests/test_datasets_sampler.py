"""Public golden tests for the development sequence sampler's RNG contract.

Expected values were obtained independently by AST-extracting the actual W/X
methods on 2026-09-23 with NumPy 1.26.3, not by executing this release sampler.
Both reference method ASTs had SHA256
b76198bd519949536b644c3c00146c563d82eb1af95ed79a347c1ef307bb71ee.
An additional 21,600-case W/X/release sweep compared complete outputs and
post-call RNG states. No private checkout or dataset is needed for these tests.
"""

import numpy as np
import pytest

from abot_recon.training.datasets.base import MultiViewDataset


# seed, sequence length, output length, start position, min/max interval,
# video probability, fixed-interval probability, shuffle block size.
GOLDENS = [
    pytest.param(
        (3, 128, 12, 0, 1, 8, 1.0, 1.0, None),
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        True, 0.7345771514092145, id="long-fixed-video",
    ),
    pytest.param(
        (9, 128, 12, 3, 1, 8, 1.0, 0.0, None),
        [3, 7, 14, 22, 25, 26, 31, 37, 44, 50, 56, 64],
        True, 0.026587734467597213, id="long-variable-video",
    ),
    pytest.param(
        (11, 64, 10, 2, 1, 4, 0.0, 0.0, None),
        [8, 4, 2, 22, 20, 3, 10, 16, 19, 13],
        False, 0.5113900218032627, id="long-collection-full-shuffle",
    ),
    pytest.param(
        (17, 64, 10, 2, 1, 4, 0.0, 0.5, 4),
        [9, 5, 10, 2, 11, 16, 20, 13, 22, 23],
        False, 0.25404091379263927, id="long-collection-block-shuffle",
    ),
    pytest.param(
        (5, 32, 12, 20, 1, 4, 0.5, 0.5, 4),
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        True, 0.8050029237453802, id="exact-tail-no-rng-consumption",
    ),
    pytest.param(
        (12, 64, 16, 55, 1, 4, 1.0, 1.0, None),
        [34, 37, 40, 43, 46, 49, 52, 55, 34, 37, 40, 43, 46, 49, 52, 55],
        True, 0.8963093737046804, id="short-fixed-video-revisit",
    ),
    pytest.param(
        (12, 64, 16, 55, 1, 4, 1.0, 0.0, None),
        [34, 36, 40, 44, 45, 46, 47, 48, 34, 36, 40, 44, 45, 46, 47, 48],
        True, 0.11507938212344748, id="short-variable-video-revisit",
    ),
    pytest.param(
        (0, 18, 20, 8, 1, 4, 0.0, 0.5, 4),
        [8, 9, 7, 10, 11, 13, 14, 12, 15, 8, 9, 7, 10, 11, 13, 14, 12, 15, 8, 9],
        False, 0.5436249914654229, id="short-collection-block-revisit",
    ),
    pytest.param(
        (4, 18, 20, 8, 1, 4, 0.0, 0.5, 4),
        [14, 10, 11, 8, 9, 11, 13, 11, 7, 12, 13, 7, 10, 9, 14, 6, 10, 14, 9, 7],
        False, 0.9841529999311214, id="short-collection-random-repeats",
    ),
    pytest.param(
        (1, 18, 20, 8, 1, 4, 0.0, 0.5, 4),
        [4, 4, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11, 11, 11, 11, 11],
        False, 0.303194829291645, id="short-collection-ordered-repeats",
    ),
]


@pytest.mark.parametrize("case,expected,is_video,next_random", GOLDENS)
def test_sequence_sampler_matches_development_goldens(case, expected, is_video, next_random):
    seed, count, num_views, start, minimum, maximum, video, fixed, block = case
    # The method only needs the shuffle helper; avoid filesystem initialization.
    sampler = object.__new__(MultiViewDataset)
    rng = np.random.default_rng(seed)
    ids = list(range(700, 700 + count))  # Nonzero IDs catch position/ID confusion.
    positions, actual_video = sampler.get_seq_from_start_id(
        num_views, ids[start], ids, rng,
        min_interval=minimum, max_interval=maximum,
        video_prob=video, fix_interval_prob=fixed, block_shuffle=block,
    )
    assert list(positions) == expected
    assert bool(actual_video) is is_video
    assert rng.random() == next_random


def test_single_frame_sampler_does_not_consume_rng():
    # Explicit release guard for the reference method's N=1 division-by-zero.
    sampler = object.__new__(MultiViewDataset)
    rng = np.random.default_rng(17)
    untouched = np.random.default_rng(17)
    assert sampler.get_seq_from_start_id(1, 703, list(range(700, 720)), rng) == ([3], True)
    assert rng.random() == untouched.random()
