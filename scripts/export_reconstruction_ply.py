#!/usr/bin/env python3
"""Export ABot-Recon poses and dense point maps to a colored binary PLY."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export camera frustums and/or ABot-Recon point maps to one PLY file."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .ply path")
    parser.add_argument("--poses", type=Path, help="camera_poses*.npy with shape [N,4,4]")
    parser.add_argument("--points", type=Path, help="local/world points .pt or an RGB .ply")
    parser.add_argument("--colors", type=Path, help="Optional colors.pt with shape [N,H,W,3]")
    parser.add_argument(
        "--points-frame",
        choices=("auto", "local", "world"),
        default="auto",
        help="Coordinate frame of --points; auto infers it from the filename",
    )
    parser.add_argument("--metadata", type=Path, help="metadata.json with dense_output_indices")
    parser.add_argument("--confidence", type=Path, help="Optional confidence.pt")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--point-stride", type=int, default=4, help="Sample every N pixels")
    parser.add_argument("--frame-stride", type=int, default=1, help="Sample every N dense frames")
    parser.add_argument("--max-points", type=int, default=5_000_000)
    parser.add_argument("--pose-stride", type=int, default=1, help="Draw every Nth camera")
    parser.add_argument("--frustum-scale", type=float, default=0.15)
    parser.add_argument("--bev-output", type=Path, help="Optional top-down trajectory PNG")
    parser.add_argument("--bev-size", type=int, default=1600, help="BEV image width and height")
    parser.add_argument(
        "--bev-plane",
        choices=("auto", "xy", "xz", "yz"),
        default="auto",
        help="World-coordinate plane used for BEV; auto selects the two largest spans",
    )
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("points", "local_points", "world_points", "confidence", "tensor"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def load_poses(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape [N,4,4], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("Pose array contains NaN or Inf")
    return poses


def time_colors(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 1:
        alpha = np.zeros(len(indices), dtype=np.float32)
    else:
        alpha = indices.astype(np.float32) / float(count - 1)
    red = np.round(255.0 * (1.0 - alpha))
    blue = np.round(255.0 * alpha)
    green = np.round(70.0 + 80.0 * (1.0 - np.abs(2.0 * alpha - 1.0)))
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def load_rgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply_types = {
        "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
        "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
        "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    }
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        current_element = None
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = raw.decode("ascii").strip()
            fields = line.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                pass
            elif fields and fields[0] == "format":
                raise ValueError("Only binary_little_endian PLY input is supported")
            elif fields[:2] == ["element", "vertex"]:
                current_element = "vertex"
                vertex_count = int(fields[2])
            elif fields and fields[0] == "element":
                current_element = fields[1]
            elif fields and fields[0] == "property" and current_element == "vertex":
                if fields[1] == "list":
                    raise ValueError("List properties are not supported on PLY vertices")
                if fields[1] not in ply_types:
                    raise ValueError(f"Unsupported PLY property type: {fields[1]}")
                vertex_properties.append((fields[2], ply_types[fields[1]]))
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        dtype = np.dtype(vertex_properties)
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)

    required = {"x", "y", "z", "red", "green", "blue"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError(f"PLY must contain XYZ and RGB vertex properties: {path}")
    points = np.stack((data["x"], data["y"], data["z"]), axis=1).astype(np.float32)
    colors = np.stack((data["red"], data["green"], data["blue"]), axis=1).astype(np.uint8)
    valid = np.isfinite(points).all(axis=1)
    return points[valid], colors[valid]


def dense_pose_indices(metadata: Path | None, dense_count: int, pose_count: int) -> np.ndarray:
    if metadata is not None:
        payload = json.loads(metadata.read_text())
        indices = payload.get("dense_output_indices")
        if indices is not None:
            result = np.asarray(indices, dtype=np.int64)
            if len(result) != dense_count:
                raise ValueError("metadata dense_output_indices does not match point frame count")
            return result
    if dense_count != pose_count:
        raise ValueError(
            f"Point maps have {dense_count} frames but poses have {pose_count}; "
            "pass --metadata containing dense_output_indices"
        )
    return np.arange(dense_count, dtype=np.int64)


def prepare_points(
    args: argparse.Namespace, poses: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if args.points is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    if args.points.suffix.lower() == ".ply":
        if args.colors is not None:
            raise ValueError("--colors is not used when --points is already an RGB PLY")
        points, colors = load_rgb_ply(args.points)
        if args.max_points > 0 and len(points) > args.max_points:
            keep = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
            points, colors = points[keep], colors[keep]
        return points, colors

    point_maps = load_tensor(args.points)
    if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
        raise ValueError(f"Expected point maps with shape [N,H,W,3], got {point_maps.shape}")
    dense_count = len(point_maps)
    coordinate_frame = args.points_frame
    if coordinate_frame == "auto":
        name = args.points.name.lower()
        if "world" in name:
            coordinate_frame = "world"
        elif "local" in name:
            coordinate_frame = "local"
        else:
            raise ValueError("Cannot infer point frame; pass --points-frame local or world")

    if coordinate_frame == "local" and poses is None:
        raise ValueError("Local point maps require --poses to transform them into world coordinates")

    if coordinate_frame == "local":
        pose_indices = dense_pose_indices(args.metadata, dense_count, len(poses))
        if np.any((pose_indices < 0) | (pose_indices >= len(poses))):
            raise ValueError("dense_output_indices contains an out-of-range pose index")
    else:
        pose_indices = None

    confidence = None
    if args.confidence is not None:
        confidence = load_tensor(args.confidence)
        if confidence.shape != point_maps.shape[:-1]:
            raise ValueError(
                f"Confidence shape {tuple(confidence.shape)} does not match "
                f"point maps {tuple(point_maps.shape[:-1])}"
            )

    rgb_maps = None
    if args.colors is not None:
        rgb_maps = load_tensor(args.colors)
        if rgb_maps.shape != point_maps.shape:
            raise ValueError(
                f"Color shape {tuple(rgb_maps.shape)} does not match "
                f"point maps {tuple(point_maps.shape)}"
            )
        if rgb_maps.dtype != torch.uint8:
            rgb_maps = rgb_maps.float()
            if rgb_maps.numel() and float(rgb_maps.max()) <= 1.5:
                rgb_maps = rgb_maps * 255.0
            rgb_maps = rgb_maps.round().clamp(0, 255).to(torch.uint8)

    chunks: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    frame_ids = np.arange(0, dense_count, args.frame_stride, dtype=np.int64)
    for dense_index in frame_ids:
        sampled = point_maps[dense_index, :: args.point_stride, :: args.point_stride].numpy()
        flat = sampled.reshape(-1, 3).astype(np.float64, copy=False)
        valid = np.isfinite(flat).all(axis=1)
        if confidence is not None and args.confidence_threshold is not None:
            conf = confidence[dense_index, :: args.point_stride, :: args.point_stride]
            valid &= conf.numpy().reshape(-1) >= args.confidence_threshold
        flat = flat[valid]
        if coordinate_frame == "local" and len(flat):
            pose = poses[pose_indices[dense_index]]
            flat = flat @ pose[:3, :3].T + pose[:3, 3]
        chunks.append(flat.astype(np.float32, copy=False))
        if rgb_maps is not None:
            sampled_rgb = rgb_maps[
                dense_index, :: args.point_stride, :: args.point_stride
            ].numpy().reshape(-1, 3)
            colors.append(sampled_rgb[valid])
        else:
            colors.append(np.full((len(flat), 3), 180, dtype=np.uint8))

    points = np.concatenate(chunks) if chunks else np.empty((0, 3), np.float32)
    rgb = np.concatenate(colors) if colors else np.empty((0, 3), np.uint8)
    if args.max_points > 0 and len(points) > args.max_points:
        keep = np.linspace(0, len(points) - 1, args.max_points, dtype=np.int64)
        points, rgb = points[keep], rgb[keep]
    return points, rgb


def prepare_frustums(
    poses: np.ndarray | None, stride: int, scale: float, vertex_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if poses is None:
        empty_v = np.empty((0, 3), np.float32)
        empty_c = np.empty((0, 3), np.uint8)
        return empty_v, empty_c, np.empty((0, 2), np.int32), empty_c

    template = scale * np.array(
        [[0, 0, 0], [-0.8, -0.5, 1], [0.8, -0.5, 1], [0.8, 0.5, 1], [-0.8, 0.5, 1]],
        dtype=np.float64,
    )
    local_edges = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
        dtype=np.int32,
    )
    selected = np.arange(0, len(poses), stride, dtype=np.int64)
    vertices: list[np.ndarray] = []
    vertex_colors: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    edge_colors: list[np.ndarray] = []
    for camera_number, pose_index in enumerate(selected):
        pose = poses[pose_index]
        vertices.append((template @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
        color = time_colors(np.array([pose_index]), len(poses))[0]
        vertex_colors.append(np.broadcast_to(color, (len(template), 3)).copy())
        base = vertex_offset + camera_number * len(template)
        edges.append(local_edges + base)
        edge_colors.append(np.broadcast_to(color, (len(local_edges), 3)).copy())

    if len(selected) > 1:
        centers = vertex_offset + np.arange(len(selected), dtype=np.int32) * len(template)
        trajectory_edges = np.stack((centers[:-1], centers[1:]), axis=1)
        trajectory_colors = time_colors(selected[:-1], len(poses))
        edges.append(trajectory_edges)
        edge_colors.append(trajectory_colors)

    return (
        np.concatenate(vertices),
        np.concatenate(vertex_colors),
        np.concatenate(edges),
        np.concatenate(edge_colors),
    )


def write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    edges: np.ndarray,
    edge_colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated by ABot-Recon export_reconstruction_ply.py\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element edge {len(edges)}\n"
        "property int vertex1\nproperty int vertex2\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    edge_dtype = np.dtype(
        [("v1", "<i4"), ("v2", "<i4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices.T
    vertex_data["r"], vertex_data["g"], vertex_data["b"] = colors.T
    edge_data = np.empty(len(edges), dtype=edge_dtype)
    if len(edges):
        edge_data["v1"], edge_data["v2"] = edges.T
        edge_data["r"], edge_data["g"], edge_data["b"] = edge_colors.T
    with path.open("wb") as handle:
        handle.write(header)
        vertex_data.tofile(handle)
        edge_data.tofile(handle)


def write_bev(path: Path, poses: np.ndarray, size: int, plane: str) -> str:
    if size < 256:
        raise ValueError("--bev-size must be at least 256")
    centers = poses[:, :3, 3]
    if plane == "auto":
        spans = np.ptp(centers, axis=0)
        axes = tuple(sorted(np.argsort(spans)[-2:].tolist()))
        plane = "".join("xyz"[axis] for axis in axes)
    else:
        axes = tuple("xyz".index(axis) for axis in plane)

    trajectory = centers[:, list(axes)].astype(np.float64)
    lower = trajectory.min(axis=0)
    upper = trajectory.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    margin = max(48, size // 20)
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])
    canvas = np.empty((len(trajectory), 2), dtype=np.float64)
    canvas[:, 0] = margin + (trajectory[:, 0] - lower[0]) * scale
    canvas[:, 1] = size - margin - (trajectory[:, 1] - lower[1]) * scale

    image = Image.new("RGB", (size, size), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    grid_color = (222, 226, 232)
    for fraction in np.linspace(0.0, 1.0, 11):
        coordinate = int(round(margin + fraction * (size - 2 * margin)))
        draw.line((coordinate, margin, coordinate, size - margin), fill=grid_color, width=1)
        draw.line((margin, coordinate, size - margin, coordinate), fill=grid_color, width=1)
    draw.rectangle((margin, margin, size - margin, size - margin), outline=(150, 158, 170), width=2)

    colors = time_colors(np.arange(len(canvas)), len(canvas))
    line_width = max(3, size // 400)
    for index in range(len(canvas) - 1):
        draw.line(
            (*canvas[index], *canvas[index + 1]),
            fill=tuple(int(value) for value in colors[index]),
            width=line_width,
        )

    marker_radius = max(7, size // 120)
    for point, fill, label, label_y in (
        (canvas[0], (230, 45, 45), "START", -2 * marker_radius),
        (canvas[-1], (35, 85, 230), "END", marker_radius + 4),
    ):
        x, y = point
        draw.ellipse(
            (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
            fill=fill,
            outline=(255, 255, 255),
            width=2,
        )
        draw.text((x + marker_radius + 5, y + label_y), label, fill=(35, 40, 48))

    heading_step = max(1, len(poses) // 40)
    heading_scale = max(size * 0.012, marker_radius * 2.0)
    for pose_index in range(0, len(poses), heading_step):
        forward_3d = poses[pose_index, :3, 2]
        direction = np.array([forward_3d[axes[0]], -forward_3d[axes[1]]], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        origin = canvas[pose_index]
        tip = origin + direction / norm * heading_scale
        draw.line((*origin, *tip), fill=(55, 60, 70), width=max(1, line_width // 2))

    draw.text((margin, 14), f"BEV trajectory ({plane.upper()} plane)", fill=(25, 30, 38))
    draw.text(
        (margin, size - margin + 12),
        f"{len(poses):,} poses | span: {span[0]:.2f} x {span[1]:.2f}",
        fill=(75, 82, 94),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return plane


def main() -> None:
    args = parse_args()
    if args.poses is None and args.points is None:
        raise SystemExit("At least one of --poses or --points is required")
    if args.point_stride < 1 or args.frame_stride < 1 or args.pose_stride < 1:
        raise SystemExit("All stride values must be >= 1")
    if args.frustum_scale <= 0:
        raise SystemExit("--frustum-scale must be positive")
    if args.bev_output is not None and args.poses is None:
        raise SystemExit("--bev-output requires --poses")

    poses = load_poses(args.poses)
    points, point_colors = prepare_points(args, poses)
    empty_edges = np.empty((0, 2), dtype=np.int32)
    empty_edge_colors = np.empty((0, 3), dtype=np.uint8)
    write_binary_ply(args.output, points, point_colors, empty_edges, empty_edge_colors)
    print(f"Wrote {args.output}: {len(points):,} RGB points")
    if args.bev_output is not None:
        plane = write_bev(args.bev_output, poses, args.bev_size, args.bev_plane)
        print(f"Wrote {args.bev_output}: {len(poses):,} poses on the {plane.upper()} plane")


if __name__ == "__main__":
    main()
