#!/usr/bin/env python3
"""Collect the formal seven-model pose benchmark into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "cut3r",
    "ttt3r",
    "longstream",
    "infinitevggt",
    "ovggt",
    "stream3r_window5",
    "horizon",
)
DATASETS = ("kitti", "oxford", "vbr")
DATASET_KEYS = {
    "kitti": "kitti-long",
    "oxford": "oxford_spires_processed-long",
    "vbr": "vbr-long",
}


def read_latest_sequence_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest[row["seq"]] = row
    return list(latest.values())


def find_one(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name)) if root.is_dir() else []
    if len(matches) > 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {matches}")
    return matches[0] if matches else None


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    status_root = root / "status"
    output_rows = []

    fps_by_method: dict[str, float] = {}
    for method in METHODS:
        timing_path = find_one(root / "fps" / method, "forward_timing_summary.json")
        if timing_path:
            fps_by_method[method] = float(json.loads(timing_path.read_text())["forward_fps"])

    for method in METHODS:
        for dataset in DATASETS:
            task = f"{method}__{dataset}"
            run_root = root / "metrics" / f"{method}_{dataset}_s1_sim3"
            seq_path = find_one(run_root, "seq_metrics.csv")
            rows = read_latest_sequence_rows(seq_path) if seq_path else []
            if (status_root / f"{task}.done").is_file():
                status = "complete"
            elif (status_root / f"{task}.failed").is_file():
                status = "failed"
            elif rows:
                status = "partial"
            else:
                status = "pending"
            output_rows.append({
                "model": method,
                "dataset": DATASET_KEYS[dataset],
                "status": status,
                "sequences": len(rows),
                "ATE": mean(rows, "ATE"),
                "RPE-r_deg": mean(rows, "RPE rot"),
                "RPE-t": mean(rows, "RPE trans"),
                "forward_FPS_32frame_VBR": fps_by_method.get(method),
            })

    output = root / "pose_metrics_fps_comparison.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    print(output)
    for row in output_rows:
        print(
            f"{row['model']:18s} {row['dataset']:34s} {row['status']:8s} "
            f"n={row['sequences']:2d} ATE={row['ATE']} "
            f"RPE-r={row['RPE-r_deg']} RPE-t={row['RPE-t']} "
            f"FPS={row['forward_FPS_32frame_VBR']}"
        )


if __name__ == "__main__":
    main()
