"""Runtime audit records shared by pose and point-cloud evaluation."""

from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

_ATTR = "_eval3r_runtime_manifest"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def clear_model_runtime(model) -> None:
    if hasattr(model, _ATTR):
        delattr(model, _ATTR)


def record_model_runtime(
    model,
    *,
    input_hw: Sequence[int],
    input_storage_dtype: str,
    forward_compute_dtype: str,
    preprocess: str,
    online_state: str,
    forward_frames: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    height, width = (int(input_hw[0]), int(input_hw[1]))
    record: Dict[str, Any] = {
        "input_h": height,
        "input_w": width,
        "input_storage_dtype": str(input_storage_dtype),
        "forward_compute_dtype": str(forward_compute_dtype),
        "preprocess": str(preprocess),
        "online_state": str(online_state),
        "forward_frames": int(forward_frames),
    }
    if extra:
        record.update(_jsonable(extra))
    setattr(model, _ATTR, record)


def require_model_runtime(model) -> Dict[str, Any]:
    record = getattr(model, _ATTR, None)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"{type(model).__name__} did not report actual runtime shape/dtype/state"
        )
    required = {
        "input_h",
        "input_w",
        "input_storage_dtype",
        "forward_compute_dtype",
        "preprocess",
        "online_state",
        "forward_frames",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError(f"Incomplete runtime manifest from adapter: missing={missing}")
    return dict(record)


def original_rgb_hw(path: str) -> Tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def cuda_runtime_name(device: str) -> str:
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return str(device_obj)
    index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
    return torch.cuda.get_device_name(index)


def _frame_ids(values: Optional[Iterable[Any]], fallback_count: int) -> list:
    if values is None:
        return list(range(int(fallback_count)))
    array = np.asarray(list(values)).reshape(-1)
    return [_jsonable(value) for value in array]


def write_runtime_manifest(
    *,
    output_root: str,
    model_name: str,
    dataset_name: str,
    sequence_name: str,
    task: str,
    filelist: Sequence[str],
    runtime: Dict[str, Any],
    metric_frame_ids: Optional[Iterable[Any]],
    metric_frame_count: int,
    checkpoint: str,
    device: str,
    protocol: Dict[str, Any],
) -> Tuple[str, str]:
    if not filelist:
        raise ValueError("Cannot write runtime manifest for an empty sequence")
    protocol = dict(protocol)
    source_ids = _frame_ids(protocol.pop("source_frame_ids", None), len(filelist))
    metric_ids = _frame_ids(metric_frame_ids, metric_frame_count)
    source_h, source_w = original_rgb_hw(filelist[0])
    payload: Dict[str, Any] = {
        "model": str(model_name),
        "dataset": str(dataset_name),
        "sequence": str(sequence_name),
        "task": str(task),
        "checkpoint": str(checkpoint),
        "source_rgb_h": source_h,
        "source_rgb_w": source_w,
        "source_frame_count": len(filelist),
        "source_frame_ids": source_ids,
        "metric_frame_count": int(metric_frame_count),
        "metric_frame_ids": metric_ids,
        "gpu": cuda_runtime_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        **_jsonable(runtime),
        **_jsonable(protocol),
    }
    if int(payload["forward_frames"]) != len(filelist):
        raise RuntimeError(
            f"Runtime forward-frame mismatch for {sequence_name}: "
            f"adapter={payload['forward_frames']}, input={len(filelist)}"
        )

    safe = str(sequence_name).replace("/", "_").replace(" ", "_")
    manifest_dir = Path(output_root) / "runtime_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / f"{safe}.json"
    temporary = json_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, json_path)

    # Rebuild from the per-sequence records so resume/re-evaluation is an
    # idempotent upsert rather than an append that silently duplicates rows.
    records = []
    for path in sorted(manifest_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        for key, value in list(item.items()):
            if isinstance(value, (dict, list)):
                item[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        records.append(item)
    fieldnames = sorted({key for item in records for key in item})
    csv_path_obj = Path(output_root) / "runtime_manifest.csv"
    csv_tmp = csv_path_obj.with_suffix(f".csv.tmp-{os.getpid()}")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(csv_tmp, csv_path_obj)
    return str(csv_path_obj), str(json_path)
