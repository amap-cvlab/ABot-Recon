from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .gpu_pgo import optimize_keyframe_pose_graph_gpu_sparse
from .types import LoopEdge


@dataclass(frozen=True)
class PoseGraphConfig:
    pose_graph_model: str = "se3"
    pose_graph_update_mode: str = "all"
    pose_graph_loop_weight: float = 0.01
    pose_graph_trans_weight: float = 1.0
    pose_graph_rot_weight: float = 1.0
    pose_graph_scale_weight: float = 1.0
    pose_graph_max_iterations: int = 30
    pose_graph_lambda_init: float = 1.0e-6
    pose_graph_solver_verbose: bool = False
    rigid_min_inliers: int = 24
    gpu_pgo_device: str = "cuda"
    gpu_pgo_pcg_max_iterations: int = 256
    gpu_pgo_pcg_tolerance: float = 1.0e-5
    gpu_pgo_pcg_check_interval: int = 8
    gpu_pgo_coarse_group_size: int = 64
    gpu_pgo_solve_dtype: str = "float64"
    gpu_pgo_outer_relative_tolerance: float = 0.0


def relative_measurement(source_c2w: np.ndarray, destination_c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(destination_c2w) @ source_c2w


def build_odometry_edges(camera_poses: np.ndarray) -> list[LoopEdge]:
    return [
        LoopEdge(
            src_pos=index - 1,
            dst_pos=index,
            src_frame=index - 1,
            dst_frame=index,
            score=1.0,
            inliers=0,
            method="odometry",
            transform_ji=relative_measurement(camera_poses[index - 1], camera_poses[index]),
        )
        for index in range(1, len(camera_poses))
    ]


def optimize_pose_graph(
    camera_poses: np.ndarray,
    odometry_edges: Sequence[LoopEdge],
    loop_edges: Sequence[LoopEdge],
    config: PoseGraphConfig,
    *,
    runtime_stats: dict | None = None,
) -> np.ndarray:
    return optimize_keyframe_pose_graph_gpu_sparse(
        keyframe_c2w_init=camera_poses,
        odom_edges=odometry_edges,
        loop_edges=loop_edges,
        cfg=config,
        device=config.gpu_pgo_device,
        pcg_max_iterations=config.gpu_pgo_pcg_max_iterations,
        pcg_tolerance=config.gpu_pgo_pcg_tolerance,
        pcg_check_interval=config.gpu_pgo_pcg_check_interval,
        coarse_group_size=config.gpu_pgo_coarse_group_size,
        solve_dtype=config.gpu_pgo_solve_dtype,
        outer_relative_tolerance=config.gpu_pgo_outer_relative_tolerance,
        runtime_stats=runtime_stats,
    )


def save_edges(path: str | Path, edges: Sequence[LoopEdge]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for edge in edges:
        item = asdict(edge)
        item["transform_ji"] = edge.transform_ji.tolist()
        payload.append(item)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
