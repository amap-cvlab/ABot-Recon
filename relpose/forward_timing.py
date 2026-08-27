"""Forward-only timing helpers for long-pose evaluation adapters."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch


_SAMPLES_ATTR = "_relpose_forward_timing_samples"


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def forward_timing_enabled(cfg: Any) -> bool:
    return bool(_cfg_get(cfg, "measure_forward_fps", False))


def reset_forward_timing(model: Any) -> None:
    setattr(model, _SAMPLES_ATTR, [])


def _synchronize(device: Any) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


@contextmanager
def time_forward(
    model: Any,
    cfg: Any,
    *,
    num_frames: int,
    label: str = "forward",
) -> Iterator[None]:
    """Time one actual model call, excluding adapter preprocessing/postprocessing."""
    if not forward_timing_enabled(cfg):
        yield
        return

    device = _cfg_get(cfg, "device", "cuda")
    _synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        _synchronize(device)
        elapsed = time.perf_counter() - start
        samples = getattr(model, _SAMPLES_ATTR, None)
        if samples is None:
            samples = []
            setattr(model, _SAMPLES_ATTR, samples)
        samples.append(
            {
                "label": str(label),
                "num_frames": int(num_frames),
                "seconds": float(elapsed),
            }
        )


def materialize_for_forward_timing(
    values: Sequence | Iterator,
    cfg: Any,
    *,
    num_frames: int,
    label: str,
):
    """Preload lazy GPU inputs before timing an iterator-consuming model call."""
    if not forward_timing_enabled(cfg):
        return values
    limit = int(_cfg_get(cfg, "forward_timing_preload_limit", 256) or 0)
    if limit > 0 and int(num_frames) > limit:
        raise ValueError(
            f"{label} forward-only timing needs inputs preloaded outside the timer, "
            f"but num_frames={num_frames} exceeds forward_timing_preload_limit={limit}. "
            "Use --fps-frames at or below this limit."
        )
    return list(values)


def summarize_forward_timing(model: Any, expected_frames: int) -> dict | None:
    samples = list(getattr(model, _SAMPLES_ATTR, []) or [])
    if not samples:
        return None
    total_frames = sum(int(sample["num_frames"]) for sample in samples)
    total_seconds = sum(float(sample["seconds"]) for sample in samples)
    if total_frames != int(expected_frames):
        raise ValueError(
            f"Forward timing frame mismatch: timed={total_frames}, expected={expected_frames}"
        )
    if total_seconds <= 0:
        raise ValueError(f"Invalid forward timing duration: {total_seconds}")
    return {
        "num_frames": total_frames,
        "forward_calls": len(samples),
        "forward_seconds": total_seconds,
        "avg_forward_ms_per_frame": 1000.0 * total_seconds / total_frames,
        "forward_fps": total_frames / total_seconds,
        "samples": samples,
    }


def save_forward_timing(path: str | Path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
