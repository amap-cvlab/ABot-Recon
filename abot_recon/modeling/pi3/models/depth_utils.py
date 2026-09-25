from __future__ import annotations

from typing import Optional

import torch


def depth_from_log_z(
    log_z: torch.Tensor,
    log_z_max: Optional[float] = None,
) -> torch.Tensor:
    """Convert point-head log-depth to depth with an optional overflow guard."""
    if log_z_max is not None:
        log_z = log_z.clamp(max=float(log_z_max))
    return torch.exp(log_z)
