from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .checkpoint import (
    DEFAULT_MODEL_ID,
    RELEASE_CHECKPOINT,
    resolve_pretrained_checkpoint,
)
from .config import InferenceConfig
from .geometry import relative_from_c2w, transform_local_points
from .types import ReconstructionResult


class ABotRecon:
    """User-facing streaming reconstruction model."""

    def __init__(self, model: torch.nn.Module, config: InferenceConfig):
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = DEFAULT_MODEL_ID,
        *,
        filename: str = RELEASE_CHECKPOINT,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        **overrides,
    ) -> "ABotRecon":
        """Load ABot-Recon from a local checkpoint or Hugging Face repository."""
        config = InferenceConfig()
        overrides["checkpoint"] = resolve_pretrained_checkpoint(
            pretrained_model_name_or_path,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        config = config.override(**overrides)
        from .model import build_model

        return cls(build_model(config), config)

    @torch.inference_mode()
    def infer(
        self,
        image_paths: Iterable[str | Path],
        *,
        output_points: bool | None = None,
        output_confidence: bool | None = None,
        loop_closure: bool | None = None,
        dense_output_indices: Iterable[int] | None = None,
    ) -> ReconstructionResult:
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths must contain at least one frame")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        points = self.config.output_points if output_points is None else output_points
        confidence = (
            self.config.output_confidence if output_confidence is None else output_confidence
        )
        use_loop = self.config.loop_closure if loop_closure is None else loop_closure
        dense_indices = None
        if dense_output_indices is not None:
            dense_indices = [int(index) for index in dense_output_indices]
            if not points and not confidence:
                raise ValueError("dense_output_indices requires a dense output")
            if any(index < 0 or index >= len(paths) for index in dense_indices):
                raise ValueError("dense_output_indices contains an out-of-range frame")
            if dense_indices != sorted(set(dense_indices)):
                raise ValueError("dense_output_indices must be unique and increasing")

        descriptor_worker = None
        if use_loop and hasattr(self.model, "_frames"):
            from .loop_closure import start_descriptor_worker

            descriptor_worker = start_descriptor_worker(
                salad_checkpoint=self.config.loop_salad_checkpoint,
                dino_checkpoint=self.config.loop_dino_checkpoint,
                device=self.config.device,
            )
        try:
            inference_kwargs = {
                "output_points": bool(points),
                "output_confidence": bool(confidence),
                "dense_output_indices": dense_indices,
            }
            if descriptor_worker is not None:
                inference_kwargs["image_observer"] = descriptor_worker.submit
            output = self.model.infer_paths(paths, **inference_kwargs)
        except BaseException:
            if descriptor_worker is not None:
                descriptor_worker.cancel()
            raise

        descriptors = None
        descriptor_stats = None
        if descriptor_worker is not None:
            descriptors, descriptor_stats = descriptor_worker.finish()
        camera_poses_noloop = output["camera_poses"]
        relative_poses_noloop = relative_from_c2w(camera_poses_noloop)
        camera_poses_loop = None
        relative_poses_loop = None
        camera_poses = camera_poses_noloop
        if use_loop:
            from .loop_closure import refine_trajectory

            if descriptor_worker is None:
                camera_poses_loop = refine_trajectory(
                    paths, camera_poses_noloop, self.model, self.config
                )
            else:
                camera_poses_loop = refine_trajectory(
                    paths,
                    camera_poses_noloop,
                    self.model,
                    self.config,
                    descriptors=descriptors,
                    descriptor_stats=descriptor_stats,
                )
            relative_poses_loop = relative_from_c2w(camera_poses_loop)
            camera_poses = camera_poses_loop

        relative_poses = relative_from_c2w(camera_poses)
        local_points = output.get("local_points")
        world_points = output.get("world_points")
        if local_points is not None and (world_points is None or use_loop):
            dense_poses = camera_poses if dense_indices is None else camera_poses[dense_indices]
            world_points = transform_local_points(local_points, dense_poses)
        conf = output.get("confidence")
        return ReconstructionResult(
            camera_poses=camera_poses,
            relative_poses=relative_poses,
            camera_poses_noloop=camera_poses_noloop,
            relative_poses_noloop=relative_poses_noloop,
            camera_poses_loop=camera_poses_loop,
            relative_poses_loop=relative_poses_loop,
            local_points=local_points,
            world_points=world_points,
            confidence=conf,
            metadata={
                "frames": len(paths),
                "loop_closure": bool(use_loop),
                "attention_backend": output.get("attention_backend", "unknown"),
                "dense_output_indices": dense_indices,
                "pose_outputs": ["noloop", "loop"] if use_loop else ["noloop"],
            },
        )

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if reset is not None:
            reset()
