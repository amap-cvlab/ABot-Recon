from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def merge_ema_state_dict(
    full_state: Mapping[str, torch.Tensor],
    ema_state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Overlay a full model state with a full- or trainable-only EMA state."""
    shadow = ema_state.get("model", ema_state.get("shadow"))
    if not isinstance(shadow, Mapping):
        raise ValueError("EMA checkpoint has no model/shadow state")
    unexpected = sorted(name for name in shadow if name not in full_state)
    if unexpected:
        raise RuntimeError(f"EMA contains unknown model keys: {unexpected[:8]}")
    merged = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in full_state.items()
    }
    for name, tensor in shadow.items():
        merged[name] = tensor.detach().to(
            device="cpu",
            dtype=merged[name].dtype,
        ).contiguous()
    return merged


class ModelEMA:
    """FP32 EMA of trainable parameters, resident on the model device."""

    format_version = 1

    def __init__(self, model: nn.Module, *, decay: float, trainable_only: bool = True) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay!r}")
        self.decay = float(decay)
        self.trainable_only = bool(trainable_only)
        self.num_updates = 0
        self.param_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad or not self.trainable_only
        ]
        if not self.param_names:
            raise ValueError("EMA selected no model parameters")
        parameters = dict(model.named_parameters())
        self.shadow = {
            name: parameters[name].detach().to(dtype=torch.float32).clone()
            for name in self.param_names
        }

    def _parameters(self, model: nn.Module) -> dict[str, nn.Parameter]:
        parameters = dict(model.named_parameters())
        current = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad or not self.trainable_only
        ]
        if current != self.param_names:
            raise ValueError(
                "EMA parameter structure no longer matches the model: "
                f"ema={len(self.param_names)}, model={len(current)}"
            )
        return parameters

    @property
    def storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.shadow.values())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = self._parameters(model)
        one_minus_decay = 1.0 - self.decay
        for name in self.param_names:
            source = parameters[name].detach().to(
                device=self.shadow[name].device,
                dtype=self.shadow[name].dtype,
            )
            self.shadow[name].mul_(self.decay).add_(source, alpha=one_minus_decay)
        self.num_updates += 1

    @contextmanager
    def apply(self, model: nn.Module):
        parameters = self._parameters(model)
        pairs = [(parameters[name], self.shadow[name]) for name in self.param_names]
        for parameter, shadow in pairs:
            if parameter.device != shadow.device:
                raise RuntimeError(
                    f"EMA and model must share a device, got {shadow.device} and {parameter.device}"
                )

        def swap() -> None:
            for parameter, shadow in pairs:
                online = parameter.data
                parameter.data = shadow.data
                shadow.data = online

        with torch.no_grad():
            swap()
        try:
            yield
        finally:
            with torch.no_grad():
                swap()

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "trainable_only": self.trainable_only,
            "param_names": list(self.param_names),
            "decay": self.decay,
            "num_updates": self.num_updates,
            "model": self.shadow,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        model_state = state.get("model", state.get("shadow"))
        if not isinstance(model_state, Mapping):
            raise ValueError("EMA checkpoint has no model/shadow state")
        missing = [name for name in self.param_names if name not in model_state]
        unexpected = [name for name in model_state if name not in self.shadow]
        if missing or unexpected:
            raise RuntimeError(
                f"EMA state mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        for name in self.param_names:
            self.shadow[name].copy_(
                model_state[name].to(
                    device=self.shadow[name].device,
                    dtype=self.shadow[name].dtype,
                )
            )
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
