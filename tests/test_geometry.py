import torch

from abot_recon.geometry import relative_from_c2w, transform_local_points


def test_relative_pose_and_world_points():
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 2.0
    relative = relative_from_c2w(poses)
    assert torch.allclose(relative[0, :3, 3], torch.tensor([-2.0, 0.0, 0.0]))
    local = torch.zeros(2, 1, 1, 3)
    world = transform_local_points(local, poses)
    assert torch.allclose(world[:, 0, 0, 0], torch.tensor([0.0, 2.0]))

