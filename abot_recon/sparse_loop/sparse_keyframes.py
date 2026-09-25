"""Sparse pose-graph nodes with smooth full-trajectory correction."""

# Copyright (c) 2026 ABot-Recon Authors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def select_keyframes(num_frames: int, loop_edges: Sequence, stride: int) -> np.ndarray:
    if num_frames <= 0 or stride <= 0:
        raise ValueError("num_frames and stride must be positive")
    selected = set(range(0, num_frames, stride))
    selected.update((0, num_frames - 1))
    for edge in loop_edges:
        selected.update((int(edge.src_pos), int(edge.dst_pos)))
    if min(selected) < 0 or max(selected) >= num_frames:
        raise IndexError("loop endpoint is outside the trajectory")
    return np.asarray(sorted(selected), dtype=np.int64)


def prepare_sparse_graph(poses: np.ndarray, loop_edges: Sequence, stride: int):
    keyframes = select_keyframes(len(poses), loop_edges, stride)
    node_for_frame = {int(frame): node for node, frame in enumerate(keyframes)}
    remapped = [
        replace(
            edge,
            src_pos=node_for_frame[int(edge.src_pos)],
            dst_pos=node_for_frame[int(edge.dst_pos)],
        )
        for edge in loop_edges
    ]
    return keyframes, poses[keyframes].copy(), remapped


def _project_rotations(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for index, matrix in enumerate(values):
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = u @ vt
        output[index] = rotation
    return output


def interpolate_corrections(
    base: np.ndarray,
    keyframe_ids: np.ndarray,
    optimized_keyframes: np.ndarray,
) -> np.ndarray:
    """Interpolate left-multiplicative corrections over all frame indices."""
    raw_keyframes = base[keyframe_ids]
    corrections = optimized_keyframes @ np.linalg.inv(raw_keyframes)
    rotations = _project_rotations(corrections[:, :3, :3])
    translations = corrections[:, :3, 3]
    frame_ids = np.arange(len(base), dtype=np.float64)
    interpolated_rotations = Slerp(
        keyframe_ids.astype(np.float64), Rotation.from_matrix(rotations)
    )(frame_ids).as_matrix()
    interpolated_translations = np.column_stack(
        [np.interp(frame_ids, keyframe_ids, translations[:, axis]) for axis in range(3)]
    )
    per_frame = np.repeat(np.eye(4)[None], len(base), axis=0)
    per_frame[:, :3, :3] = interpolated_rotations
    per_frame[:, :3, 3] = interpolated_translations
    output = per_frame @ base
    output[:, :3, :3] = _project_rotations(output[:, :3, :3])
    output[:, 3] = (0, 0, 0, 1)
    output[keyframe_ids] = optimized_keyframes
    return output
