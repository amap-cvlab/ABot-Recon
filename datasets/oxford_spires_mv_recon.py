"""Oxford Spires multi-view reconstruction dataset.

All rectified RGB frames are forwarded in temporal order. Dense TLS-derived
depth is loaded only for the sampled metric frames listed in ``frame_ids.txt``.
This keeps Stream models at stride 1 while evaluating geometry every N frames.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.vggt.utils.geometry import unproject_depth_map_to_point_map


def _load_matrix_rows(path: str, columns: int) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got {rows.shape}")
    return rows


def resize_sparse_depth_and_intrinsic(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-resize sparse depth and scale K with pixel-centre geometry."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    src_h, src_w = depth.shape
    if output_width <= 0 or output_width == src_w:
        return np.asarray(depth, dtype=np.float32), np.asarray(intrinsic, dtype=np.float32)
    dst_w = int(output_width)
    dst_h = max(1, int(round(src_h * dst_w / float(src_w))))
    resized = cv2.resize(depth, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    K = np.asarray(intrinsic, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return resized.astype(np.float32, copy=False), K.astype(np.float32)


class OxfordSpiresMVRecon(Dataset):
    """Rectified Oxford RGB plus TLS-projected sparse depth for mv_recon."""

    def __init__(
        self,
        image_root: str,
        gt_root: str,
        load_img_size: int = 518,
        expected_interval: int = 10,
        depth_min: float = 0.1,
        depth_max: float = 200.0,
        alignment_depth_max: float = 80.0,
    ):
        self.image_root = osp.abspath(image_root)
        self.gt_root = osp.abspath(gt_root)
        self.load_img_size = int(load_img_size)
        self.expected_interval = int(expected_interval)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        requested_alignment_depth_max = float(alignment_depth_max)
        if requested_alignment_depth_max <= self.depth_min:
            raise ValueError(
                "alignment_depth_max must exceed depth_min, got "
                f"{requested_alignment_depth_max} <= {self.depth_min}"
            )
        self.alignment_depth_max = min(requested_alignment_depth_max, self.depth_max)
        if not osp.isdir(self.image_root) or not osp.isdir(self.gt_root):
            raise FileNotFoundError(
                f"Oxford roots missing: image_root={self.image_root}, gt_root={self.gt_root}"
            )

        image_sequences = {
            name
            for name in os.listdir(self.image_root)
            if osp.isdir(osp.join(self.image_root, name, "images"))
        }
        gt_sequences = {
            name
            for name in os.listdir(self.gt_root)
            if osp.isfile(osp.join(self.gt_root, name, "DONE.json"))
        }
        self.sequence_list = sorted(image_sequences & gt_sequences)
        if not self.sequence_list:
            raise RuntimeError("No Oxford sequence has both processed RGB and point-cloud GT")

        self.metadata: Dict[str, Dict] = {}
        for sequence_name in self.sequence_list:
            self.metadata[sequence_name] = self._load_sequence_metadata(sequence_name)

    def _load_sequence_metadata(self, sequence_name: str) -> Dict:
        image_dir = osp.join(self.image_root, sequence_name, "images")
        image_paths = sorted(
            osp.join(image_dir, name)
            for name in os.listdir(image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        processed_dir = osp.join(self.image_root, sequence_name)
        poses_c2w = _load_matrix_rows(osp.join(processed_dir, "poses_c2w.txt"), 16).reshape(-1, 4, 4)
        intr_values = np.loadtxt(osp.join(processed_dir, "intrinsics.txt"), dtype=np.float64).reshape(-1)
        if len(intr_values) != 6:
            raise ValueError(f"{sequence_name}: intrinsics.txt must contain fx fy cx cy W H")
        fx, fy, cx, cy, width, height = intr_values
        if len(image_paths) != len(poses_c2w):
            raise ValueError(
                f"{sequence_name}: {len(image_paths)} RGB frames vs {len(poses_c2w)} poses"
            )

        gt_dir = osp.join(self.gt_root, sequence_name)
        with open(osp.join(gt_dir, "DONE.json"), "r", encoding="utf-8") as handle:
            done = json.load(handle)
        frame_ids = np.loadtxt(osp.join(gt_dir, "frame_ids.txt"), dtype=np.int64)
        frame_ids = np.atleast_1d(frame_ids).astype(np.int64)
        sampled_poses = _load_matrix_rows(osp.join(gt_dir, "poses_c2w.txt"), 16).reshape(-1, 4, 4)
        if len(frame_ids) != len(sampled_poses):
            raise ValueError(f"{sequence_name}: frame_ids and sampled poses disagree")
        if np.any(frame_ids < 0) or np.any(frame_ids >= len(image_paths)):
            raise ValueError(f"{sequence_name}: GT frame IDs outside RGB sequence")
        if not np.all(np.diff(frame_ids) > 0):
            raise ValueError(f"{sequence_name}: frame_ids.txt must be strictly increasing")
        if self.expected_interval > 0 and len(frame_ids) > 1:
            if not np.all(np.diff(frame_ids) == self.expected_interval):
                raise ValueError(
                    f"{sequence_name}: expected GT interval {self.expected_interval}, "
                    f"got {np.unique(np.diff(frame_ids)).tolist()}"
                )
        np.testing.assert_allclose(
            sampled_poses,
            poses_c2w[frame_ids],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"{sequence_name}: sampled GT poses do not match processed poses",
        )
        if int(done["source_frame_count"]) != len(image_paths):
            raise ValueError(f"{sequence_name}: DONE source_frame_count is stale")
        if (int(width), int(height)) != (int(done["width"]), int(done["height"])):
            raise ValueError(f"{sequence_name}: processed and GT image sizes disagree")

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return {
            "image_paths": image_paths,
            "poses_c2w": poses_c2w,
            "intrinsic": K,
            "native_hw": (int(height), int(width)),
            "gt_dir": gt_dir,
            "gt_frame_ids": frame_ids,
            "gt_frame_set": set(frame_ids.tolist()),
            "done": done,
        }

    def __len__(self) -> int:
        return len(self.sequence_list)

    def get_seq_framenum(
        self, index: Optional[int] = None, sequence_name: Optional[str] = None
    ) -> int:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name]["image_paths"])

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[List[int], np.ndarray, None] = None,
    ) -> Dict:
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify index or sequence_name")
            sequence_name = self.sequence_list[index]
        item = self.metadata[sequence_name]
        seq_len = len(item["image_paths"])
        if ids is None:
            ids = list(range(seq_len))
        ids = [int(value) for value in np.asarray(ids).reshape(-1).tolist()]
        if not ids or any(value < 0 or value >= seq_len for value in ids):
            raise IndexError(f"{sequence_name}: invalid frame IDs")
        if any(right <= left for left, right in zip(ids, ids[1:])):
            raise ValueError("Oxford streaming IDs must be unique and chronological")

        metric_positions = [i for i, frame_id in enumerate(ids) if frame_id in item["gt_frame_set"]]
        metric_frame_ids = [ids[i] for i in metric_positions]
        if not metric_positions:
            raise ValueError(
                f"{sequence_name}: requested range contains no TLS depth frame; "
                f"GT interval={self.expected_interval}"
            )

        depths: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for frame_id in metric_frame_ids:
            depth_path = osp.join(item["gt_dir"], "depth", f"{frame_id:06d}.npy")
            if not osp.isfile(depth_path):
                raise FileNotFoundError(depth_path)
            depth = np.load(depth_path).astype(np.float32, copy=False)
            if depth.shape != item["native_hw"]:
                raise ValueError(
                    f"{sequence_name} frame {frame_id}: depth {depth.shape} != {item['native_hw']}"
                )
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0
            depth, K = resize_sparse_depth_and_intrinsic(
                depth, item["intrinsic"], self.load_img_size
            )
            depths.append(depth)
            intrinsics.append(K)

        depths_array = np.stack(depths, axis=0)
        intrinsics_array = np.stack(intrinsics, axis=0)
        metric_c2w = item["poses_c2w"][metric_frame_ids]
        metric_w2c = np.linalg.inv(metric_c2w).astype(np.float32)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths_array[..., None],
            intrinsics_cam=intrinsics_array,
            extrinsics_cam=metric_w2c[:, :3, :],
        ).astype(np.float32, copy=False)

        full_c2w = item["poses_c2w"][ids]
        full_w2c = np.linalg.inv(full_c2w).astype(np.float32)
        return {
            "seq_id": sequence_name,
            "seq_len": seq_len,
            "ind": torch.tensor(ids, dtype=torch.long),
            "image_paths": [item["image_paths"][i] for i in ids],
            "image_hw": tuple(depths_array.shape[-2:]),
            "images": None,
            "pointclouds": pointclouds,
            "valid_mask": depths_array > self.depth_min,
            "alignment_gt_mask": (
                (depths_array > self.depth_min)
                & (depths_array <= self.alignment_depth_max)
            ),
            "metric_indices": np.asarray(metric_positions, dtype=np.int64),
            "metric_frame_ids": np.asarray(metric_frame_ids, dtype=np.int64),
            "metric_image_paths": [item["image_paths"][i] for i in metric_frame_ids],
            "extrs": torch.from_numpy(full_w2c),
            "metric_extrs": torch.from_numpy(metric_w2c),
            "intrs": torch.from_numpy(intrinsics_array),
        }

    def write_depth_association_audit(self, output_path: str) -> None:
        report = {
            "protocol": {
                "inference": "all rectified RGB frames in chronological order",
                "geometry": "TLS depth projected into processed undistorted pinhole images",
                "metric_frames": f"frame_ids.txt (expected interval={self.expected_interval})",
                "depth_resize": "nearest; K scaled per axis with pixel-centre convention",
                "depth_range_m": [self.depth_min, self.depth_max],
                "alignment_gt_depth_max_m": self.alignment_depth_max,
            },
            "sequences": {},
        }
        for name, item in self.metadata.items():
            report["sequences"][name] = {
                "num_inference_frames": len(item["image_paths"]),
                "num_metric_frames": len(item["gt_frame_ids"]),
                "native_hw": list(item["native_hw"]),
                "metric_frame_first": int(item["gt_frame_ids"][0]),
                "metric_frame_last": int(item["gt_frame_ids"][-1]),
            }
        os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
