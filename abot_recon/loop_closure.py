from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .sparse_loop.assets import ensure_loop_assets
from .sparse_loop.graph import (
    LoopEdge,
    PoseGraphConfig,
    build_odometry_edges,
    optimize_pose_graph,
    relative_measurement,
    save_edges,
)
from .sparse_loop.retrieval import (
    RetrievalConfig,
    SaladDescriptorWorker,
    retrieve_candidates,
    retrieve_candidates_from_descriptors,
)
from .sparse_loop.sparse_keyframes import interpolate_corrections, prepare_sparse_graph


DEFAULT_SALAD_CHECKPOINT = "checkpoints/loop/dino_salad.ckpt"
DEFAULT_DINO_CHECKPOINT = "checkpoints/loop/dinov2_vitb14_pretrain.pth"


def _optimize_pose_graph_with_grad(
    graph_poses: np.ndarray,
    graph_loop_edges: Sequence[LoopEdge],
    graph_config: PoseGraphConfig,
    runtime_stats: dict[str, Any],
) -> np.ndarray:
    """Run autograd-based PGO safely from the public inference-mode API."""
    with torch.inference_mode(False), torch.enable_grad():
        return optimize_pose_graph(
            graph_poses,
            build_odometry_edges(graph_poses),
            graph_loop_edges,
            graph_config,
            runtime_stats=runtime_stats,
        )


@dataclass(frozen=True)
class LoopClosureConfig:
    enabled: bool = False
    auto_download: bool = False
    salad_checkpoint: str = DEFAULT_SALAD_CHECKPOINT
    dino_checkpoint: str = DEFAULT_DINO_CHECKPOINT
    salad_backbone: str = "dinov2_vitb14"
    salad_image_size: tuple[int, int] = (336, 336)
    salad_batch_size: int = 32
    descriptor_cache: bool = True
    descriptor_queue_size: int = 64
    salad_score_threshold: float = 0.85
    retrieval_top_k: int = 5
    min_frame_separation: int = 30
    nms_radius: int = 25
    keyframe_stride: int = 1
    loop_chunk_size: int = 10
    max_candidates: int = 1000
    max_reinfer_candidates: int = 50
    default_inliers: int = 128
    pose_graph_loop_weight: float = 0.01
    pose_graph_node_mode: str = "sparse_keyframes"
    pose_graph_keyframe_stride: int = 50
    pose_graph_trans_weight: float = 1.0
    pose_graph_rot_weight: float = 1.0
    pose_graph_max_iterations: int = 30
    pose_graph_lambda_init: float = 1.0e-6
    gpu_pgo_pcg_max_iterations: int = 256
    gpu_pgo_pcg_tolerance: float = 1.0e-5
    gpu_pgo_pcg_check_interval: int = 8
    gpu_pgo_coarse_group_size: int = 64
    gpu_pgo_solve_dtype: str = "float64"
    verbose: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "LoopClosureConfig":
        if values is None:
            return cls()
        data = dict(values)
        aliases = {
            "salad_ckpt_path": "salad_checkpoint",
            "salad_dino_weights_path": "dino_checkpoint",
            "salad_score_thresh": "salad_score_threshold",
        }
        for old_name, new_name in aliases.items():
            if old_name in data and new_name not in data:
                data[new_name] = data.pop(old_name)
        allowed = cls.__dataclass_fields__
        return cls(**{name: value for name, value in data.items() if name in allowed})


def _centered_range(center: int, size: int, total: int) -> tuple[int, int]:
    size = max(1, min(size, total))
    start = center - size // 2
    end = start + size
    if start < 0:
        return 0, size
    if end > total:
        return total - size, total
    return start, end


def _as_4x4(poses: torch.Tensor) -> torch.Tensor:
    if poses.ndim != 3:
        raise ValueError(f"expected poses [N,3/4,4], got {tuple(poses.shape)}")
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] != (3, 4):
        raise ValueError(f"expected poses [N,3/4,4], got {tuple(poses.shape)}")
    bottom = torch.zeros((*poses.shape[:-2], 1, 4), dtype=poses.dtype, device=poses.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat((poses, bottom), dim=-2)


def _load_rgb(paths: Sequence[str | Path], image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), Image.Resampling.BICUBIC)
            frames.append(np.asarray(rgb, dtype=np.uint8))
    if not frames:
        raise ValueError("loop closure received an empty image sequence")
    return np.stack(frames)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def apply_loop_closure(
    model: torch.nn.Module,
    image_paths: Sequence[str | Path],
    base_poses: torch.Tensor,
    *,
    device: torch.device,
    loop_cfg: LoopClosureConfig | Mapping[str, Any] | None = None,
    save_dir: str | Path | None = None,
    descriptors: np.ndarray | None = None,
    descriptor_stats: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    config = (
        loop_cfg
        if isinstance(loop_cfg, LoopClosureConfig)
        else LoopClosureConfig.from_mapping(loop_cfg)
    )
    base_poses = _as_4x4(base_poses.detach().float().cpu())
    if not config.enabled:
        return base_poses, {"enabled": False}
    if len(image_paths) != len(base_poses):
        raise ValueError(
            f"loop closure path/pose length mismatch: {len(image_paths)} vs {len(base_poses)}"
        )

    started = time.perf_counter()
    salad_checkpoint, dino_checkpoint = ensure_loop_assets(
        config.salad_checkpoint,
        config.dino_checkpoint,
        auto_download=config.auto_download,
    )
    total = len(image_paths)
    keyframe_stride = max(config.keyframe_stride, 1)
    keyframe_indices = list(range(0, total, keyframe_stride))
    if keyframe_indices[-1] != total - 1:
        keyframe_indices.append(total - 1)
    keyframe_paths = [image_paths[index] for index in keyframe_indices]

    retrieval_started = time.perf_counter()
    retrieval_config = RetrievalConfig(
        salad_checkpoint=salad_checkpoint,
        dino_checkpoint=dino_checkpoint,
        backbone=config.salad_backbone,
        image_size=config.salad_image_size,
        batch_size=config.salad_batch_size,
        score_threshold=config.salad_score_threshold,
        top_k=config.retrieval_top_k,
        min_frame_separation=max(1, int(np.ceil(config.min_frame_separation / keyframe_stride))),
        nms_radius=max(0, int(np.ceil(config.nms_radius / keyframe_stride))),
        max_candidates=config.max_candidates,
        verbose=config.verbose,
    )
    faiss_stats: dict[str, Any] = {}
    if descriptors is None:
        keyframe_candidates = retrieve_candidates(
            _load_rgb(keyframe_paths, config.salad_image_size), retrieval_config, device
        )
    else:
        keyframe_candidates = retrieve_candidates_from_descriptors(
            descriptors[keyframe_indices], retrieval_config, device, runtime_stats=faiss_stats
        )
    keyframe_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    retrieval_seconds = time.perf_counter() - retrieval_started

    loop_edges: list[LoopEdge] = []
    edge_debug = []
    reinference_seconds = 0.0
    for rank, candidate in enumerate(
        keyframe_candidates[: max(config.max_reinfer_candidates, 0)], start=1
    ):
        source = keyframe_indices[candidate.src_frame]
        destination = keyframe_indices[candidate.dst_frame]
        if abs(source - destination) < config.min_frame_separation:
            continue
        source_start, source_end = _centered_range(source, config.loop_chunk_size, total)
        destination_start, destination_end = _centered_range(
            destination, config.loop_chunk_size, total
        )
        frame_ids = sorted(
            set(range(source_start, source_end)) | set(range(destination_start, destination_end))
        )
        local_paths = [Path(image_paths[index]) for index in frame_ids]
        reinference_started = time.perf_counter()
        local_poses = model.infer_paths(
            local_paths,
            output_local_points=False,
            output_world_points=False,
            output_confidence=False,
        )[
            "camera_poses"
        ]
        reinference_seconds += time.perf_counter() - reinference_started
        local = _as_4x4(local_poses.detach().float().cpu()).numpy().astype(np.float64)
        transform = relative_measurement(
            local[frame_ids.index(source)], local[frame_ids.index(destination)]
        )
        loop_edges.append(
            LoopEdge(
                src_pos=source,
                dst_pos=destination,
                src_frame=source,
                dst_frame=destination,
                score=candidate.score,
                inliers=max(config.default_inliers, 1),
                method="salad_online_abot_reinfer",
                transform_ji=transform,
            )
        )
        edge_debug.append(
            {
                "candidate_rank": rank,
                "src_frame": source,
                "dst_frame": destination,
                "score": candidate.score,
                "loop_frame_ids": frame_ids,
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pgo_seconds = 0.0
    pgo_stats: dict[str, Any] = {}
    if loop_edges:
        pgo_started = time.perf_counter()
        base_numpy = base_poses.numpy().astype(np.float64)
        graph_poses = base_numpy
        graph_loop_edges = loop_edges
        keyframes = None
        if config.pose_graph_node_mode == "sparse_keyframes":
            keyframes, graph_poses, graph_loop_edges = prepare_sparse_graph(
                base_numpy, loop_edges, config.pose_graph_keyframe_stride
            )
        graph_config = PoseGraphConfig(
            pose_graph_loop_weight=config.pose_graph_loop_weight,
            pose_graph_trans_weight=config.pose_graph_trans_weight,
            pose_graph_rot_weight=config.pose_graph_rot_weight,
            pose_graph_max_iterations=config.pose_graph_max_iterations,
            pose_graph_lambda_init=config.pose_graph_lambda_init,
            gpu_pgo_device=str(device),
            gpu_pgo_pcg_max_iterations=config.gpu_pgo_pcg_max_iterations,
            gpu_pgo_pcg_tolerance=config.gpu_pgo_pcg_tolerance,
            gpu_pgo_pcg_check_interval=config.gpu_pgo_pcg_check_interval,
            gpu_pgo_coarse_group_size=config.gpu_pgo_coarse_group_size,
            gpu_pgo_solve_dtype=config.gpu_pgo_solve_dtype,
        )
        optimized_graph = _optimize_pose_graph_with_grad(
            graph_poses, graph_loop_edges, graph_config, pgo_stats
        )
        optimized = (
            optimized_graph
            if keyframes is None
            else interpolate_corrections(base_numpy, keyframes, optimized_graph)
        )
        output = torch.from_numpy(optimized).to(dtype=base_poses.dtype)
        pgo_seconds = time.perf_counter() - pgo_started
    else:
        output = base_poses

    info = {
        "enabled": True,
        "num_frames": total,
        "keyframe_stride": keyframe_stride,
        "num_keyframes": len(keyframe_indices),
        "num_candidates": len(keyframe_candidates),
        "num_loop_edges": len(loop_edges),
        "retrieval_sec": retrieval_seconds,
        "reinfer_sec": reinference_seconds,
        "pgo_sec": pgo_seconds,
        "total_sec": time.perf_counter() - started,
        **faiss_stats,
        **dict(descriptor_stats or {}),
        **pgo_stats,
    }
    if save_dir is not None:
        output_dir = Path(save_dir)
        _save_json(output_dir / "loop_info.json", info)
        _save_json(
            output_dir / "loop_candidates_keyframes.json",
            [asdict(candidate) for candidate in keyframe_candidates],
        )
        _save_json(output_dir / "loop_edges_debug.json", edge_debug)
        save_edges(output_dir / "loop_edges.json", loop_edges)
    if config.verbose:
        print(
            "[loop_closure] "
            f"frames={total} candidates={len(keyframe_candidates)} "
            f"edges={len(loop_edges)} total={info['total_sec']:.2f}s",
            flush=True,
        )
    return output, info


def start_descriptor_worker(
    *,
    salad_checkpoint: str | Path,
    dino_checkpoint: str | Path,
    device: str | torch.device,
    batch_size: int = 32,
    queue_size: int = 64,
):
    salad, dino = ensure_loop_assets(
        salad_checkpoint,
        dino_checkpoint,
        auto_download=False,
    )
    retrieval = RetrievalConfig(
        salad_checkpoint=salad,
        dino_checkpoint=dino,
        batch_size=batch_size,
        faiss_use_gpu=True,
    )
    return SaladDescriptorWorker(retrieval, torch.device(device), queue_size=queue_size).start()


def refine_trajectory(
    image_paths,
    camera_poses,
    model,
    config,
    *,
    descriptors=None,
    descriptor_stats=None,
):
    loop_config = LoopClosureConfig(
        enabled=True,
        salad_checkpoint=str(config.loop_salad_checkpoint),
        dino_checkpoint=str(config.loop_dino_checkpoint),
    )
    refined, _ = apply_loop_closure(
        model,
        image_paths,
        camera_poses,
        device=torch.device(config.device),
        loop_cfg=loop_config,
        save_dir=config.loop_output_dir,
        descriptors=descriptors,
        descriptor_stats=descriptor_stats,
    )
    return refined
