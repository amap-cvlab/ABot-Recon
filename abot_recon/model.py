from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Iterable

import torch

from .checkpoint import checkpoint_has_prefix, load_model_checkpoint
from .config import InferenceConfig
from .preprocessing import iter_preprocessed


RELATIVE_CAMERA_HEAD = {
    "head_type": "token_pair",
    "rotation_format": "quat",
    "hidden_dim": 512,
    "pair_hidden_dim": 512,
    "num_pose_tokens": 5,
    "init_std": 1.0e-4,
    "rot_correction_mode": "temporal_rotation_refinement",
    "rot_correction_kernel": 10,
    "rot_correction_max_deg": 2.0,
    "rot_correction_use_age_embed": True,
    "translation_param": "vector",
}


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def flashinfer_available() -> bool:
    try:
        from .modeling.streaming.paged_kv import (
            flashinfer_available as available,
        )

        return bool(available())
    except (ImportError, RuntimeError):
        return False


def resolve_attention_backend(requested: str) -> str:
    """Resolve auto to paged when possible; explicit paged never falls back."""
    if requested == "sdpa":
        return "sdpa"
    available = flashinfer_available()
    if requested == "paged" and not available:
        raise RuntimeError("attention_backend='paged' requires a working FlashInfer installation")
    return "paged" if available else "sdpa"


def disable_unavailable_packaged_flash_attention(model: torch.nn.Module) -> int:
    """Select SDPA once instead of entering exception-driven fallback per layer."""
    disabled = 0
    for module in model.modules():
        if not getattr(module, "use_packaged_flash_attn", False):
            continue
        owner = importlib.import_module(module.__class__.__module__)
        if getattr(owner, "_flash_attn_func_bhld_optional", None) is None:
            module.use_packaged_flash_attn = False
            disabled += 1
    return disabled


class ReleasedABotReconModel(torch.nn.Module):
    """Thin runtime wrapper around the checkpoint-exact inference network."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        from .modeling.streaming.network import ABotReconNetwork

        confidence = checkpoint_has_prefix(config.checkpoint, ("conf_decoder.", "conf_head."))
        self.attention_backend = resolve_attention_backend(config.attention_backend)
        device = torch.device(config.device)
        with device, torch.no_grad():
            self.network = ABotReconNetwork(
                pos_type="rope100",
                decoder_size="large",
                load_vggt=False,
                freeze_encoder=False,
                freeze_prediction_heads=False,
                use_global_points=False,
                train_conf=False,
                enable_confidence=confidence,
                confidence_only_train=False,
                init_conf_decoder_from_point=False,
                num_dec_blk_not_to_checkpoint=4,
                causal_global_attn=True,
                use_packaged_flash_attn=False,
                camera_pose_mode="relative_adjacent",
                relative_camera_head_cfg=RELATIVE_CAMERA_HEAD,
                point_z_log_max=10.0,
                ckpt=None,
                use_paged_kv=self.attention_backend == "paged",
                paged_max_total_frames=config.max_frames,
                paged_force_fp32=False,
                global_pos_encoding="rope3d",
                rope3d_config={
                    "theta": 10_000.0,
                    "max_seq_len": config.max_frames,
                    "fhw_dim": [20, 22, 22],
                },
                local_window_frames=config.local_window_frames,
                infer_mode="stream",
                gate_layers=list(range(36)),
            )
        self.disabled_packaged_flash_modules = disable_unavailable_packaged_flash_attention(
            self.network
        )
        self.config = config
        self.device_name = config.device
        self.compute_dtype = _torch_dtype(config.amp_dtype)
        # The network is constructed directly on its execution device above.
        load_model_checkpoint(self.network, config.checkpoint)
        self.eval()

    def reset(self) -> None:
        manager = getattr(self.network, "_paged_manager", None)
        if manager is not None:
            manager.reset()

    def _frames(self, paths: Iterable[Path]):
        for tensor, _ in iter_preprocessed(
            paths, height=self.config.height, width=self.config.width
        ):
            observer = getattr(self, "_image_observer", None)
            if observer is not None:
                # The loop descriptor worker consumes the exact CPU tensor that
                # is sent to the streaming model, avoiding a second decode path.
                observer(tensor.unsqueeze(0).unsqueeze(0))
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            yield tensor.to(
                device=self.device_name,
                dtype=self.compute_dtype if self.device_name.startswith("cuda") else torch.float32,
                non_blocking=True,
            )

    @torch.inference_mode()
    def infer_paths(
        self,
        paths: list[Path],
        *,
        output_local_points: bool,
        output_world_points: bool,
        output_confidence: bool,
        dense_output_indices: list[int] | None = None,
        image_observer: Callable[[torch.Tensor], None] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.reset()
        self._image_observer = image_observer
        output_keys = ["camera_poses"]
        if output_local_points or output_world_points:
            output_keys.append("local_points")
        if output_world_points:
            output_keys.append("points")
        if output_confidence:
            output_keys.append("conf")
        autocast_enabled = (
            self.device_name.startswith("cuda") and self.compute_dtype != torch.float32
        )
        device_type = "cuda" if self.device_name.startswith("cuda") else "cpu"
        try:
            with torch.autocast(
                device_type=device_type,
                dtype=self.compute_dtype,
                enabled=autocast_enabled,
            ):
                output = self.network.inference_stream_iter(
                    self._frames(paths),
                    num_frames=len(paths),
                    output_keys=output_keys,
                    dense_output_indices=dense_output_indices,
                )
        finally:
            self._image_observer = None

        def frames_only(value):
            if value is None:
                return None
            if value.ndim >= 1 and value.shape[0] == 1:
                value = value[0]
            return value.detach().float().cpu()

        result = {"camera_poses": frames_only(output["camera_poses"])}
        result["attention_backend"] = self.attention_backend
        if output_local_points or output_world_points:
            result["local_points"] = frames_only(output.get("local_points"))
        if output_world_points:
            result["world_points"] = frames_only(output.get("points"))
        if output_confidence:
            logits = frames_only(output.get("conf"))
            if logits is None:
                raise RuntimeError("The selected checkpoint has no confidence output")
            if logits.ndim == 4 and logits.shape[-1] == 1:
                logits = logits[..., 0]
            result["confidence"] = torch.sigmoid(logits)
        return result


def build_model(config: InferenceConfig) -> ReleasedABotReconModel:
    return ReleasedABotReconModel(config)
