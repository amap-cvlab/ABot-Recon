"""TUM RGB-D dynamic benchmark adapters for point-cloud evaluation.

The existing ``rgb_90`` subset was selected using RGB/pose association only.
This adapter independently associates those RGB timestamps with registered
depth timestamps.  All 90 RGB frames are forwarded to the model, while frames
without a depth match inside ``depth_max_dt`` receive an all-false GT mask and
therefore do not contribute to point-cloud alignment or metrics.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from datasets.utils.cropping import resize_image, resize_image_depth_and_intrinsic
from models.vggt.utils.geometry import unproject_depth_map_to_point_map


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()


def _read_timestamp_file(path: str) -> List[Tuple[float, str]]:
    rows: List[Tuple[float, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed timestamp row in {path}: {line!r}")
            rows.append((float(fields[0]), fields[1]))
    return rows


def _associate_one_to_one(
    first_timestamps: Sequence[float],
    second_timestamps: Sequence[float],
    max_difference: float,
) -> Dict[int, Tuple[int, float]]:
    """Globally greedy, one-to-one timestamp association (TUM devkit style)."""
    second = np.asarray(second_timestamps, dtype=np.float64)
    if second.ndim != 1:
        raise ValueError(f"second_timestamps must be 1D, got shape {second.shape}")
    order = np.argsort(second, kind="stable")
    sorted_second = second[order]
    candidates = []
    for i, value in enumerate(first_timestamps):
        value = float(value)
        lo = int(np.searchsorted(sorted_second, value - max_difference, side="right"))
        hi = int(np.searchsorted(sorted_second, value + max_difference, side="left"))
        for sorted_j in range(lo, hi):
            j = int(order[sorted_j])
            candidates.append((abs(value - float(second[j])), i, j))
    candidates.sort()
    available_first = set(range(len(first_timestamps)))
    available_second = set(range(len(second_timestamps)))
    matches: Dict[int, Tuple[int, float]] = {}
    for difference, i, j in candidates:
        if i not in available_first or j not in available_second:
            continue
        available_first.remove(i)
        available_second.remove(j)
        matches[i] = (j, float(difference))
    return matches


def _c2w_from_tum_row(row: np.ndarray) -> np.ndarray:
    tx, ty, tz, qx, qy, qz, qw = row[1:8]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    c2w[:3, 3] = [tx, ty, tz]
    return c2w


class TUMDynamics90(Dataset):
    """Eight TUM dynamic sequences using the prepared 90-frame RGB subset."""

    def __init__(
        self,
        TUM_DIR: str,
        load_img_size: int = 518,
        depth_max_dt: float = 0.005,
        pose_max_dt: float = 0.02,
        depth_min: float = 1e-3,
        depth_max: float = 5.0,
        expected_num_frames: Optional[int] = 90,
        min_depth_matches: int = 80,
        min_depth_match_ratio: float = 0.0,
        frame_source: str = "rgb_90",
        groundtruth_filename: str = "groundtruth_90.txt",
        protocol_name: str = "TUM-Dynamics-90",
    ):
        self.TUM_DIR = osp.abspath(TUM_DIR)
        self.load_img_size = load_img_size
        self.depth_max_dt = float(depth_max_dt)
        self.pose_max_dt = float(pose_max_dt)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.expected_num_frames = (
            None if expected_num_frames is None else int(expected_num_frames)
        )
        self.min_depth_matches = int(min_depth_matches)
        self.min_depth_match_ratio = float(min_depth_match_ratio)
        self.frame_source = str(frame_source)
        self.groundtruth_filename = str(groundtruth_filename)
        self.protocol_name = str(protocol_name)
        if self.frame_source not in {"rgb_90", "rgb_index"}:
            raise ValueError(f"Unsupported TUM frame_source: {self.frame_source!r}")
        if not 0.0 <= self.min_depth_match_ratio <= 1.0:
            raise ValueError(
                f"min_depth_match_ratio must be in [0, 1], got {self.min_depth_match_ratio}"
            )

        if not osp.isdir(self.TUM_DIR):
            raise FileNotFoundError(f"TUM root not found: {self.TUM_DIR}")
        self.sequence_list = sorted(
            name
            for name in os.listdir(self.TUM_DIR)
            if name.startswith("rgbd_dataset_freiburg3_")
            and osp.isdir(osp.join(self.TUM_DIR, name))
        )
        if not self.sequence_list:
            raise RuntimeError(f"No Freiburg3 sequences found under {self.TUM_DIR}")

        self.metadata: Dict[str, Dict] = {}
        for sequence_name in self.sequence_list:
            self.metadata[sequence_name] = self._build_sequence_metadata(sequence_name)

        counts = [item["num_depth_matches"] for item in self.metadata.values()]
        print(
            f"[{self.protocol_name}] {len(self.sequence_list)} sequences, "
            f"RGB={sum(len(item['rgb_paths']) for item in self.metadata.values())} total, "
            f"strict depth matches="
            f"{min(counts)}..{max(counts)}/seq (|dt|<{self.depth_max_dt:.6f}s)"
        )

    def _build_sequence_metadata(self, sequence_name: str) -> Dict:
        sequence_dir = osp.join(self.TUM_DIR, sequence_name)
        rgb_index = _read_timestamp_file(osp.join(sequence_dir, "rgb.txt"))
        if self.frame_source == "rgb_90":
            rgb_dir = osp.join(sequence_dir, "rgb_90")
            rgb_paths = sorted(
                [
                    osp.join(rgb_dir, name)
                    for name in os.listdir(rgb_dir)
                    if name.endswith(".png")
                ],
                key=lambda path: float(osp.splitext(osp.basename(path))[0]),
            )
            rgb_timestamps = np.asarray(
                [float(osp.splitext(osp.basename(path))[0]) for path in rgb_paths],
                dtype=np.float64,
            )
        else:
            rgb_timestamps = np.asarray([row[0] for row in rgb_index], dtype=np.float64)
            rgb_paths = [osp.join(sequence_dir, row[1]) for row in rgb_index]

        if self.expected_num_frames is not None and len(rgb_paths) != self.expected_num_frames:
            raise ValueError(
                f"{sequence_name}: expected {self.expected_num_frames} rgb_90 frames, "
                f"found {len(rgb_paths)}"
            )
        # Ensure every selected file is an original RGB observation, not an
        # accidentally renamed/copy-shifted frame.
        if self.frame_source == "rgb_90":
            indexed_rgb_timestamps = np.asarray(
                [row[0] for row in rgb_index], dtype=np.float64
            )
            for timestamp, path in zip(rgb_timestamps, rgb_paths):
                nearest = int(np.argmin(np.abs(indexed_rgb_timestamps - timestamp)))
                if abs(float(indexed_rgb_timestamps[nearest] - timestamp)) > 1e-7:
                    raise ValueError(
                        f"{sequence_name}: rgb_90 timestamp {timestamp} absent from rgb.txt"
                    )
                expected_name = osp.basename(rgb_index[nearest][1])
                if osp.basename(path) != expected_name:
                    raise ValueError(
                        f"{sequence_name}: rgb_90 file {osp.basename(path)} != "
                        f"rgb.txt {expected_name}"
                    )

        gt_rows_all = np.loadtxt(
            osp.join(sequence_dir, self.groundtruth_filename), dtype=np.float64
        )
        gt_rows_all = np.atleast_2d(gt_rows_all)
        if self.frame_source == "rgb_90":
            gt_rows = gt_rows_all
            num_raw_rgb = len(rgb_paths)
            num_pose_dropped = 0
        else:
            pose_matches = _associate_one_to_one(
                rgb_timestamps, gt_rows_all[:, 0], self.pose_max_dt
            )
            num_raw_rgb = len(rgb_paths)
            valid_rgb_ids = sorted(pose_matches)
            num_pose_dropped = num_raw_rgb - len(valid_rgb_ids)
            if not valid_rgb_ids:
                raise ValueError(
                    f"{sequence_name}: no RGB frame has a unique pose within "
                    f"{self.pose_max_dt:.6f}s"
                )
            gt_rows = np.stack(
                [gt_rows_all[pose_matches[i][0]] for i in valid_rgb_ids], axis=0
            )
            rgb_paths = [rgb_paths[i] for i in valid_rgb_ids]
            rgb_timestamps = rgb_timestamps[valid_rgb_ids]
        gt_rows = np.atleast_2d(gt_rows)
        if gt_rows.shape != (len(rgb_paths), 8):
            raise ValueError(
                f"{sequence_name}: {self.groundtruth_filename} selection shape "
                f"{gt_rows.shape}, expected ({len(rgb_paths)}, 8)"
            )
        pose_dt = np.abs(rgb_timestamps - gt_rows[:, 0])
        if np.any(pose_dt >= self.pose_max_dt):
            bad = int(np.argmax(pose_dt))
            raise ValueError(
                f"{sequence_name}: RGB/pose dt={pose_dt[bad]:.6f}s at frame {bad}, "
                f"limit={self.pose_max_dt:.6f}s"
            )
        extrinsics = np.stack(
            [np.linalg.inv(_c2w_from_tum_row(row)) for row in gt_rows], axis=0
        ).astype(np.float32)

        depth_index = _read_timestamp_file(osp.join(sequence_dir, "depth.txt"))
        depth_timestamps = np.asarray([row[0] for row in depth_index], dtype=np.float64)
        matches = _associate_one_to_one(
            rgb_timestamps, depth_timestamps, self.depth_max_dt
        )
        depth_paths: List[Optional[str]] = [None] * len(rgb_paths)
        depth_match_dt: List[Optional[float]] = [None] * len(rgb_paths)
        nearest_depth_dt: List[float] = []
        for i, timestamp in enumerate(rgb_timestamps):
            nearest_depth_dt.append(float(np.min(np.abs(depth_timestamps - timestamp))))
            if i not in matches:
                continue
            depth_index_id, difference = matches[i]
            depth_path = osp.join(sequence_dir, depth_index[depth_index_id][1])
            depth_paths[i] = depth_path
            depth_match_dt[i] = difference

        num_depth_matches = sum(path is not None for path in depth_paths)
        required_depth_matches = max(
            self.min_depth_matches,
            int(np.ceil(self.min_depth_match_ratio * len(rgb_paths))),
        )
        if num_depth_matches < required_depth_matches:
            raise ValueError(
                f"{sequence_name}: only {num_depth_matches}/{len(rgb_paths)} "
                f"depth matches within {self.depth_max_dt:.6f}s; "
                f"minimum={required_depth_matches}"
            )
        return {
            "rgb_paths": rgb_paths,
            "rgb_timestamps": rgb_timestamps,
            "gt_rows": gt_rows,
            "pose_dt": pose_dt,
            "extrinsics": extrinsics,
            "depth_paths": depth_paths,
            "depth_match_dt": depth_match_dt,
            "nearest_depth_dt": nearest_depth_dt,
            "num_depth_matches": num_depth_matches,
            "num_raw_rgb": num_raw_rgb,
            "num_pose_dropped": num_pose_dropped,
        }

    def __len__(self) -> int:
        return len(self.sequence_list)

    def get_seq_framenum(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
    ) -> int:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name]["rgb_paths"])

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        ids = np.random.choice(
            self.get_seq_framenum(sequence_name=sequence_name), n_per_seq, replace=False
        )
        return self.get_data(sequence_name=sequence_name, ids=ids)

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[List[int], np.ndarray, None] = None,
    ) -> Dict:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        item = self.metadata[sequence_name]
        seq_len = len(item["rgb_paths"])
        if ids is None:
            ids = list(range(seq_len))
        elif isinstance(ids, np.ndarray):
            if ids.ndim != 1:
                raise ValueError(f"ids must be 1D, got {ids.ndim}D")
            ids = ids.tolist()
        ids = [int(i) for i in ids]
        if any(i < 0 or i >= seq_len for i in ids):
            raise IndexError(f"{sequence_name}: ids outside [0, {seq_len}): {ids}")

        # TUM Freiburg3 registered RGB-D calibration, native 640x480.
        base_intrinsic = np.asarray(
            [[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        intrinsics = np.repeat(base_intrinsic[None], len(ids), axis=0)
        extrinsics = item["extrinsics"][ids]
        image_paths: List[str] = []
        images: List[torch.Tensor] = []
        depths: List[np.ndarray] = []
        has_depth: List[bool] = []

        for out_index, frame_id in enumerate(ids):
            image_path = item["rgb_paths"][frame_id]
            image = Image.open(image_path).convert("RGB")
            depth_path = item["depth_paths"][frame_id]
            if depth_path is None:
                depth = np.zeros((image.height, image.width), dtype=np.float32)
                matched = False
            else:
                raw_depth = imageio.imread(depth_path)
                if raw_depth.ndim != 2:
                    raise ValueError(f"TUM depth must be 2D, got {raw_depth.shape}: {depth_path}")
                depth = np.nan_to_num(raw_depth.astype(np.float32) / 5000.0, nan=0.0)
                depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
                matched = True
            if depth.shape != (image.height, image.width):
                image = resize_image(image, (depth.shape[1], depth.shape[0]))

            if self.load_img_size is not None and int(self.load_img_size) > 0:
                image, depth, intrinsics[out_index] = resize_image_depth_and_intrinsic(
                    image=image,
                    depth_map=depth,
                    intrinsic=intrinsics[out_index],
                    output_width=int(self.load_img_size),
                )
            image_paths.append(image_path)
            images.append(to_tensor(image))
            depths.append(depth)
            has_depth.append(matched)

        depths_array = np.stack(depths, axis=0)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths_array[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics[:, :3, :],
        )
        batch = {
            "seq_id": sequence_name,
            "seq_len": seq_len,
            "ind": torch.tensor(ids, dtype=torch.long),
            "image_paths": image_paths,
            "images": torch.stack(images, dim=0),
            "pointclouds": pointclouds,
            "valid_mask": depths_array > self.depth_min,
            "extrs": torch.from_numpy(extrinsics),
            "intrs": torch.from_numpy(intrinsics).float(),
            "has_depth": torch.tensor(has_depth, dtype=torch.bool),
            "depth_match_dt": [item["depth_match_dt"][i] for i in ids],
        }
        return batch

    def write_depth_association_audit(self, output_path: str) -> None:
        report = {
            "protocol": {
                "rgb_subset": (
                    "rgb_90 (all frames forwarded)"
                    if self.frame_source == "rgb_90"
                    else "all rgb.txt frames (no temporal subsampling)"
                ),
                "association": "global greedy one-to-one timestamp matching",
                "depth_max_dt_seconds": self.depth_max_dt,
                "pose_max_dt_seconds": self.pose_max_dt,
                "depth_scale": 5000.0,
                "depth_range_m": [self.depth_min, self.depth_max],
                "intrinsics": [535.4, 539.2, 320.1, 247.6],
                "unmatched_depth_policy": "forward RGB; all-false pointcloud GT mask",
            },
            "sequences": {},
        }
        for sequence_name in self.sequence_list:
            item = self.metadata[sequence_name]
            frames = []
            for frame_id, rgb_path in enumerate(item["rgb_paths"]):
                frames.append(
                    {
                        "frame_id": frame_id,
                        "rgb": osp.relpath(rgb_path, self.TUM_DIR),
                        "rgb_timestamp": float(item["rgb_timestamps"][frame_id]),
                        "pose_timestamp": float(item["gt_rows"][frame_id, 0]),
                        "pose_dt_seconds": float(item["pose_dt"][frame_id]),
                        "depth": (
                            None
                            if item["depth_paths"][frame_id] is None
                            else osp.relpath(item["depth_paths"][frame_id], self.TUM_DIR)
                        ),
                        "depth_dt_seconds": item["depth_match_dt"][frame_id],
                        "nearest_depth_dt_seconds": item["nearest_depth_dt"][frame_id],
                    }
                )
            matched_dt = [dt for dt in item["depth_match_dt"] if dt is not None]
            report["sequences"][sequence_name] = {
                "num_raw_rgb": item["num_raw_rgb"],
                "num_rgb": len(item["rgb_paths"]),
                "num_pose_dropped": item["num_pose_dropped"],
                "num_depth_matches": item["num_depth_matches"],
                "max_matched_depth_dt_seconds": max(matched_dt),
                "max_pose_dt_seconds": float(np.max(item["pose_dt"])),
                "frames": frames,
            }
        os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


class TUMDynamicsFull(TUMDynamics90):
    """All original RGB frames from the eight Freiburg3 dynamic sequences."""

    def __init__(self, TUM_DIR: str, *args, **kwargs):
        defaults = {
            "expected_num_frames": None,
            "min_depth_matches": 0,
            "min_depth_match_ratio": 0.8,
            "frame_source": "rgb_index",
            "groundtruth_filename": "groundtruth.txt",
            "protocol_name": "TUM-Dynamics-Full",
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(TUM_DIR, *args, **kwargs)
