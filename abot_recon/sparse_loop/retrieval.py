# Copyright (c) 2026 ABot-Recon Authors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import io
import math
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from .types import LoopCandidate


@dataclass(frozen=True)
class RetrievalConfig:
    salad_checkpoint: Path
    dino_checkpoint: Path
    backbone: str = "dinov2_vitb14"
    image_size: tuple[int, int] = (336, 336)
    batch_size: int = 32
    score_threshold: float = 0.85
    top_k: int = 5
    min_frame_separation: int = 30
    nms_radius: int = 25
    max_candidates: int = 1000
    faiss_use_gpu: bool = True
    faiss_gpu_id: int = -1
    faiss_query_batch_size: int = 512
    faiss_require_gpu: bool = False
    verbose: bool = True


def _load_dinov2(model_name: str) -> torch.nn.Module:
    cache = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
        if (cache / "hubconf.py").is_file():
            return torch.hub.load(str(cache), model_name, pretrained=False, source="local")
        return torch.hub.load(
            "facebookresearch/dinov2:main",
            model_name,
            pretrained=False,
            trust_repo=True,
            skip_validation=True,
        )


def _sinkhorn_log_transport(
    log_rows: torch.Tensor,
    log_columns: torch.Tensor,
    scores: torch.Tensor,
    *,
    iterations: int = 3,
) -> torch.Tensor:
    scores = scores / 1.0
    row_update = torch.zeros_like(log_rows)
    column_update = torch.zeros_like(log_columns)
    for _ in range(iterations):
        row_update = log_rows - torch.logsumexp(scores + column_update.unsqueeze(1), dim=2)
        column_update = log_columns - torch.logsumexp(scores + row_update.unsqueeze(2), dim=1)
    return scores + row_update.unsqueeze(2) + column_update.unsqueeze(1)


def _cluster_probabilities(scores: torch.Tensor, dustbin: torch.Tensor) -> torch.Tensor:
    batch, clusters, patches = scores.shape
    augmented = torch.empty(batch, clusters + 1, patches, dtype=scores.dtype, device=scores.device)
    augmented[:, :clusters] = scores
    augmented[:, clusters] = dustbin
    normalizer = -torch.tensor(math.log(patches + clusters), device=scores.device)
    log_rows = normalizer.expand(clusters + 1).contiguous()
    log_columns = normalizer.expand(patches).contiguous()
    log_rows[-1] += math.log(patches - clusters)
    transport = _sinkhorn_log_transport(
        log_rows.expand(batch, -1), log_columns.expand(batch, -1), augmented
    )
    return torch.exp(transport - normalizer)[:, :-1]


class SaladAggregator(torch.nn.Module):
    """SALAD descriptor aggregation compatible with the released checkpoint."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.num_channels = channels
        self.num_clusters = 64
        self.cluster_dim = 128
        self.token_dim = 256
        self.token_features = torch.nn.Sequential(
            torch.nn.Linear(channels, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.token_dim),
        )
        self.cluster_features = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 512, 1),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 512, 1),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        feature_map, class_token = inputs
        local_features = self.cluster_features(feature_map).flatten(2)
        assignment = _cluster_probabilities(self.score(feature_map).flatten(2), self.dust_bin)
        assignment = assignment.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        local_features = local_features.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        pooled = torch.nn.functional.normalize(
            (local_features * assignment).sum(dim=-1), p=2, dim=1
        ).flatten(1)
        token = torch.nn.functional.normalize(self.token_features(class_token), p=2, dim=-1)
        return torch.nn.functional.normalize(torch.cat((token, pooled), dim=-1), p=2, dim=-1)


class DinoBackbone(torch.nn.Module):
    CHANNELS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str, checkpoint: Path) -> None:
        super().__init__()
        if model_name not in self.CHANNELS:
            raise ValueError(f"unsupported DINOv2 backbone: {model_name}")
        self.num_channels = self.CHANNELS[model_name]
        self.model = _load_dinov2(model_name)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        class_token = tokens[:, 0]
        feature_map = (
            tokens[:, 1:]
            .reshape(batch, height // 14, width // 14, self.num_channels)
            .permute(0, 3, 1, 2)
        )
        return feature_map, class_token


class SaladDescriptor(torch.nn.Module):
    def __init__(self, backbone_name: str, dino_checkpoint: Path) -> None:
        super().__init__()
        self.backbone = DinoBackbone(backbone_name, dino_checkpoint)
        self.aggregator = SaladAggregator(self.backbone.num_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(images))


def _load_salad_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("model.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    missing, _ = model.load_state_dict(cleaned, strict=False)
    missing_aggregator = [name for name in missing if name.startswith("aggregator.")]
    if missing_aggregator:
        raise RuntimeError(
            "SALAD checkpoint is missing aggregator parameters: "
            + ", ".join(missing_aggregator[:5])
        )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1.0e-8, None)


def _prepare_images(batch: np.ndarray, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    height, width = size
    resized = np.stack(
        [cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for image in batch]
    )
    tensor = torch.from_numpy(resized).float().permute(0, 3, 1, 2).to(device) / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def suppress_nearby_candidates(
    candidates: Sequence[LoopCandidate], radius: int, limit: int
) -> list[LoopCandidate]:
    if not candidates or radius <= 0:
        return list(candidates)[:limit]
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    max_position = max((max(c.src_pos, c.dst_pos) for c in ranked), default=-1)
    suppressed: set[int] = set()
    kept: list[LoopCandidate] = []
    for candidate in ranked:
        low, high = sorted((candidate.src_pos, candidate.dst_pos))
        if low in suppressed or high in suppressed:
            continue
        kept.append(candidate)
        suppressed.update(range(max(0, low - radius), min(low + radius + 1, high)))
        suppressed.update(
            range(max(low + 1, high - radius), min(high + radius + 1, max_position + 1))
        )
        if len(kept) >= limit:
            break
    return kept


def compute_descriptors(
    rgb: np.ndarray, config: RetrievalConfig, device: torch.device
) -> np.ndarray:
    model = SaladDescriptor(config.backbone, config.dino_checkpoint)
    _load_salad_checkpoint(model, config.salad_checkpoint)
    model = model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rgb), config.batch_size):
            images = _prepare_images(
                rgb[start : start + config.batch_size], config.image_size, device
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                descriptor = model(images)
            chunks.append(descriptor.float().cpu().numpy().astype(np.float32))
    descriptors = np.ascontiguousarray(
        _normalize_rows(np.concatenate(chunks).astype(np.float32))
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return descriptors


def retrieve_candidates_from_descriptors(
    descriptors: np.ndarray,
    config: RetrievalConfig,
    device: torch.device,
    *,
    runtime_stats: dict | None = None,
) -> list[LoopCandidate]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("loop closure requires faiss-cpu or faiss-gpu") from exc
    descriptors = np.ascontiguousarray(
        _normalize_rows(np.asarray(descriptors, dtype=np.float32))
    )
    build_started = time.perf_counter()
    cpu_index = faiss.IndexFlatIP(descriptors.shape[1])
    index = cpu_index
    resources = None
    if config.faiss_use_gpu and device.type == "cuda":
        try:
            gpu_id = int(device.index or 0) if config.faiss_gpu_id < 0 else config.faiss_gpu_id
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("installed FAISS package has no GPU API")
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        except Exception as exc:
            if config.faiss_require_gpu:
                raise RuntimeError("GPU FAISS was requested but is unavailable") from exc
    elif config.faiss_require_gpu:
        raise RuntimeError("GPU FAISS requires CUDA inference")
    index.add(descriptors)
    build_seconds = time.perf_counter() - build_started

    scores_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    search_k = min(len(descriptors), config.top_k + 1)
    search_started = time.perf_counter()
    for start in range(0, len(descriptors), config.faiss_query_batch_size):
        scores, indices = index.search(
            descriptors[start : start + config.faiss_query_batch_size], search_k
        )
        scores_parts.append(scores)
        indices_parts.append(indices)
    scores = np.concatenate(scores_parts)
    indices = np.concatenate(indices_parts)
    if runtime_stats is not None:
        runtime_stats.update(
            faiss_backend="gpu" if index is not cpu_index else "cpu",
            faiss_index_build_sec=build_seconds,
            faiss_search_sec=time.perf_counter() - search_started,
            faiss_query_batch_size=config.faiss_query_batch_size,
            faiss_query_batches=len(scores_parts),
        )

    unique: dict[tuple[int, int], LoopCandidate] = {}
    for source in range(len(descriptors)):
        for rank in range(1, search_k):
            destination = int(indices[source, rank])
            score = float(scores[source, rank])
            if destination < 0 or destination == source or score <= config.score_threshold:
                continue
            if abs(source - destination) <= config.min_frame_separation:
                continue
            high, low = max(source, destination), min(source, destination)
            candidate = LoopCandidate(high, low, high, low, score, "salad_online")
            if (high, low) not in unique or unique[(high, low)].score < score:
                unique[(high, low)] = candidate
    del index, cpu_index, resources
    return suppress_nearby_candidates(
        list(unique.values()), config.nms_radius, config.max_candidates
    )


def retrieve_candidates(
    rgb: np.ndarray, config: RetrievalConfig, device: torch.device
) -> list[LoopCandidate]:
    return retrieve_candidates_from_descriptors(
        compute_descriptors(rgb, config, device), config, device
    )


class SaladDescriptorWorker:
    """Build descriptors concurrently from tensors decoded by the base stream."""

    def __init__(self, config: RetrievalConfig, device: torch.device, queue_size: int = 64):
        self.config = config
        self.device = device
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self.stop_token = object()
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._run, name="salad-descriptor", daemon=True)
        self.error: BaseException | None = None
        self.chunks: list[np.ndarray] = []
        self.started = 0.0
        self.stats = {
            "salad_cache_frames": 0,
            "salad_cache_batches": 0,
            "salad_cache_model_init_sec": 0.0,
            "salad_cache_preprocess_sec": 0.0,
            "salad_cache_forward_sec": 0.0,
        }

    def start(self):
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def submit(self, image: torch.Tensor) -> None:
        image = image.detach().cpu()
        if image.ndim != 5 or image.shape[:2] != (1, 1):
            raise ValueError(f"descriptor worker expects [1,1,3,H,W], got {image.shape}")
        self.queue.put(image)

    def _process(self, model: torch.nn.Module, tensors: list[torch.Tensor]) -> None:
        started = time.perf_counter()
        rgb = torch.cat(tensors, dim=1).squeeze(0).permute(0, 2, 3, 1)
        rgb = rgb.clamp(0, 1).mul(255).add(.5).to(torch.uint8).numpy()
        images = _prepare_images(rgb, self.config.image_size, self.device)
        self.stats["salad_cache_preprocess_sec"] += time.perf_counter() - started
        started = time.perf_counter()
        start_event = None
        end_event = None
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            descriptors = model(images)
        if end_event is not None and start_event is not None:
            end_event.record()
            end_event.synchronize()
            self.stats["salad_cache_forward_sec"] += start_event.elapsed_time(end_event) / 1000
        else:
            self.stats["salad_cache_forward_sec"] += time.perf_counter() - started
        values = descriptors.float().cpu().numpy().astype(np.float32, copy=False)
        self.chunks.append(values)
        self.stats["salad_cache_frames"] += len(values)
        self.stats["salad_cache_batches"] += 1

    def _run(self) -> None:
        model = None
        try:
            started = time.perf_counter()
            model = SaladDescriptor(self.config.backbone, self.config.dino_checkpoint)
            _load_salad_checkpoint(model, self.config.salad_checkpoint)
            model = model.to(self.device).eval()
            self.stats["salad_cache_model_init_sec"] = time.perf_counter() - started
            pending = []
            while True:
                item = self.queue.get()
                if item is self.stop_token:
                    if pending and not self.cancelled.is_set():
                        self._process(model, pending)
                    break
                if not self.cancelled.is_set():
                    pending.append(item)
                    if len(pending) >= self.config.batch_size:
                        self._process(model, pending)
                        pending = []
        except BaseException as exc:
            self.error = exc
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _stop(self) -> None:
        while self.thread.is_alive():
            try:
                self.queue.put(self.stop_token, timeout=.5)
                return
            except queue.Full:
                if self.error:
                    return

    def finish(self) -> tuple[np.ndarray, dict]:
        started = time.perf_counter()
        self._stop()
        self.thread.join()
        self.stats["salad_cache_finish_wait_sec"] = time.perf_counter() - started
        self.stats["salad_cache_total_sec"] = time.perf_counter() - self.started
        if self.error:
            raise RuntimeError("SALAD descriptor worker failed") from self.error
        values = np.ascontiguousarray(_normalize_rows(np.concatenate(self.chunks)))
        return values, dict(self.stats)

    def cancel(self) -> None:
        self.cancelled.set()
        self._stop()
        self.thread.join(timeout=30)
