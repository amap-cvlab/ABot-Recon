"""
Build per-scene continuous-3840-frame windows for Oxford Spires.

Scans ``{DATA_OXFORD_SPIRES}/{scene}/raw/{CAM}/*.jpg`` and
``{scene}/processed/trajectory/gt-tum.txt`` (TUM format with scalar-last
quaternion: ``ts tx ty tz qx qy qz qw``). For each scene:

1. Sorts images by their filename timestamp.
2. For each image timestamp, finds the nearest GT timestamp; the image is
   considered "has GT" if the absolute time gap <= ``--max_gt_gap`` seconds.
3. Locates the longest run of consecutive "has GT" images and selects the
   first window of length ``--window_len`` (default 3840) inside that run.
   Scenes with no such run are recorded with ``selected=false``.
4. Writes the result to a JSON file with per-scene image paths, matched GT
   row indices, and timestamp diagnostics.

Usage:
    python eval/build_oxford_spires_json.py \
        --root data/oxford_spires/sequences \
        --out  eval/oxford_spires_sequences.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_ROOT = "data/oxford_spires/sequences"


def _image_timestamp(path: str) -> float:
    name = osp.splitext(osp.basename(path))[0]
    return float(name)


def _load_gt_timestamps(gt_file: str) -> np.ndarray:
    arr = np.loadtxt(gt_file, usecols=(0,))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _nearest_indices(query: np.ndarray, ref_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each value in ``query``, return the nearest index in ``ref_sorted``
    plus the absolute time gap."""
    ins = np.searchsorted(ref_sorted, query)
    ins = np.clip(ins, 1, len(ref_sorted) - 1)
    left = ref_sorted[ins - 1]
    right = ref_sorted[ins]
    pick_right = (np.abs(query - right) < np.abs(query - left))
    nearest_idx = np.where(pick_right, ins, ins - 1)
    nearest_gap = np.abs(query - ref_sorted[nearest_idx])
    return nearest_idx, nearest_gap


def _longest_true_run(mask: np.ndarray) -> Tuple[int, int]:
    """Return ``(start, length)`` of the longest contiguous True run. Empty
    masks return ``(-1, 0)``."""
    if mask.size == 0:
        return -1, 0
    best_start, best_len = -1, 0
    cur_start, cur_len = 0, 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_start, best_len


def _scan_scene(
    scene_dir: str, cam: str, window_len: int, max_gt_gap: float
) -> Dict[str, object]:
    cam_dir = osp.join(scene_dir, "raw", cam)
    gt_file = osp.join(scene_dir, "processed", "trajectory", "gt-tum.txt")
    out: Dict[str, object] = {
        "img_dir": cam_dir,
        "gt_file": gt_file,
        "selected": False,
        "reason": "",
    }
    if not osp.isdir(cam_dir):
        out["reason"] = f"missing cam dir: {cam_dir}"
        return out
    if not osp.isfile(gt_file):
        out["reason"] = f"missing gt: {gt_file}"
        return out

    paths = sorted(glob.glob(osp.join(cam_dir, "*.jpg")))
    if not paths:
        out["reason"] = "no jpg images"
        return out

    img_ts = np.array([_image_timestamp(p) for p in paths], dtype=np.float64)
    gt_ts = _load_gt_timestamps(gt_file)
    if gt_ts.size == 0:
        out["reason"] = "empty gt"
        return out

    sort_order = np.argsort(gt_ts)
    gt_ts_sorted = gt_ts[sort_order]
    nearest_sorted_idx, gaps = _nearest_indices(img_ts, gt_ts_sorted)
    nearest_gt_idx = sort_order[nearest_sorted_idx]
    has_gt = gaps <= max_gt_gap

    n_imgs = len(paths)
    n_with_gt = int(has_gt.sum())
    out.update(
        {
            "n_images_total": n_imgs,
            "n_gt_rows": int(gt_ts.size),
            "n_images_with_gt": n_with_gt,
            "gap_stats": {
                "median_s": float(np.median(gaps)),
                "p95_s": float(np.quantile(gaps, 0.95)),
                "max_s": float(np.max(gaps)),
            },
            "max_gt_gap_s": max_gt_gap,
            "window_len": window_len,
        }
    )

    run_start, run_len = _longest_true_run(has_gt)
    if run_len < window_len:
        out["reason"] = (
            f"longest consecutive GT-matched run = {run_len} < {window_len}"
        )
        out["longest_run"] = {"start": int(run_start), "length": int(run_len)}
        return out

    sel_start = run_start
    sel_end = run_start + window_len  # exclusive
    sel_paths = paths[sel_start:sel_end]
    sel_img_ts = img_ts[sel_start:sel_end]
    sel_gt_idx = nearest_gt_idx[sel_start:sel_end]
    sel_gaps = gaps[sel_start:sel_end]

    out.update(
        {
            "selected": True,
            "start_index": int(sel_start),
            "end_index": int(sel_end),  # exclusive
            "image_files": [osp.basename(p) for p in sel_paths],
            "image_timestamps": sel_img_ts.tolist(),
            "gt_row_indices": sel_gt_idx.astype(int).tolist(),
            "selected_gap_stats": {
                "median_s": float(np.median(sel_gaps)),
                "p95_s": float(np.quantile(sel_gaps, 0.95)),
                "max_s": float(np.max(sel_gaps)),
            },
            "longest_run": {"start": int(run_start), "length": int(run_len)},
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("build_oxford_spires_json")
    p.add_argument("--root", default=DEFAULT_ROOT, type=str,
                   help="Path to oxford_spires/sequences directory")
    p.add_argument("--out", required=True, type=str,
                   help="Output JSON path")
    p.add_argument("--cam", default="cam0", type=str,
                   help="Camera subdir under raw/ (default cam0)")
    p.add_argument("--window_len", default=3840, type=int,
                   help="Required number of consecutive GT-matched frames")
    p.add_argument("--max_gt_gap", default=0.05, type=float,
                   help="Max |img_ts - gt_ts| (seconds) for a frame to count as having GT")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Optional explicit scene whitelist (default: auto-discover)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not osp.isdir(args.root):
        raise SystemExit(f"--root not a directory: {args.root}")

    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = sorted(
            d for d in os.listdir(args.root) if osp.isdir(osp.join(args.root, d))
        )

    print(f"[scan] root={args.root} cam={args.cam} window_len={args.window_len} "
          f"max_gt_gap={args.max_gt_gap}s scenes={len(scenes)}")

    results: Dict[str, Dict[str, object]] = {}
    n_selected = 0
    for scene in scenes:
        scene_dir = osp.join(args.root, scene)
        info = _scan_scene(scene_dir, args.cam, args.window_len, args.max_gt_gap)
        results[scene] = info
        tag = "[ok]" if info.get("selected") else "[skip]"
        n_imgs = info.get("n_images_total", "?")
        n_with_gt = info.get("n_images_with_gt", "?")
        extra = ""
        if info.get("selected"):
            n_selected += 1
            gap = info.get("selected_gap_stats", {}).get("max_s", -1)
            extra = (
                f" start={info['start_index']} end={info['end_index']} "
                f"max_gap={gap:.4f}s"
            )
        else:
            extra = f" reason={info.get('reason', '?')}"
        print(f"  {tag} {scene}: imgs={n_imgs} gt_matched={n_with_gt}{extra}")

    summary = {
        "root": args.root,
        "cam": args.cam,
        "window_len": args.window_len,
        "max_gt_gap_s": args.max_gt_gap,
        "n_scenes_total": len(scenes),
        "n_scenes_selected": n_selected,
        "scenes": results,
    }
    os.makedirs(osp.dirname(osp.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}: {n_selected}/{len(scenes)} scenes selected")


if __name__ == "__main__":
    main()
