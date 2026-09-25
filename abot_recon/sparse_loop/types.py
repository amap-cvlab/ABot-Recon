from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoopCandidate:
    src_pos: int
    dst_pos: int
    src_frame: int
    dst_frame: int
    score: float
    method: str = "salad"


@dataclass(frozen=True)
class LoopEdge:
    src_pos: int
    dst_pos: int
    src_frame: int
    dst_frame: int
    score: float
    inliers: int
    method: str
    transform_ji: np.ndarray
