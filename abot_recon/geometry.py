from __future__ import annotations

import torch


def relative_from_c2w(camera_poses: torch.Tensor) -> torch.Tensor:
    """Return adjacent transforms mapping frame t coordinates into frame t+1."""
    if camera_poses.ndim != 3 or camera_poses.shape[-2:] != (4, 4):
        raise ValueError("camera_poses must have shape [N,4,4]")
    if len(camera_poses) < 2:
        return camera_poses.new_empty((0, 4, 4))
    return torch.linalg.solve(camera_poses[1:], camera_poses[:-1])


def transform_local_points(local_points: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
    if local_points.ndim != 4 or local_points.shape[-1] != 3:
        raise ValueError("local_points must have shape [N,H,W,3]")
    if camera_poses.shape != (len(local_points), 4, 4):
        raise ValueError("camera_poses and local_points frame counts must match")
    rotation = camera_poses[:, :3, :3]
    translation = camera_poses[:, :3, 3]
    return torch.einsum("nij,nhwj->nhwi", rotation, local_points) + translation[:, None, None]

