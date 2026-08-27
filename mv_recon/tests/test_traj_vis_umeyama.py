import numpy as np

from mv_recon.traj_vis import apply_sim3_to_c2w, trajectory_umeyama_transform


def _poses(points):
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(points), axis=0)
    poses[:, :3, 3] = points
    return poses


def test_trajectory_umeyama_recovers_known_sim3():
    pred = _poses(
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [0.0, 2.0, 1.0]]
        )
    )
    angle = np.deg2rad(35.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    scale = 2.7
    translation = np.asarray([4.0, -3.0, 1.2])
    gt = pred.copy()
    gt[:, :3, :3] = rotation @ pred[:, :3, :3]
    gt[:, :3, 3] = (scale * (rotation @ pred[:, :3, 3].T)).T + translation

    transform = trajectory_umeyama_transform(pred, gt, with_scale=True)
    aligned = apply_sim3_to_c2w(pred, transform)

    np.testing.assert_allclose(aligned[:, :3, 3], gt[:, :3, 3], atol=1e-10)
    np.testing.assert_allclose(aligned[:, :3, :3], gt[:, :3, :3], atol=1e-10)
    np.testing.assert_allclose(
        np.mean(np.linalg.norm(transform[:3, :3], axis=0)), scale, atol=1e-10
    )


def test_trajectory_se3_keeps_scale():
    pred = _poses(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    gt = pred.copy()
    gt[:, :3, 3] = 3.0 * pred[:, :3, 3] + np.asarray([2.0, -1.0, 0.5])

    transform = trajectory_umeyama_transform(pred, gt, with_scale=False)

    np.testing.assert_allclose(
        np.mean(np.linalg.norm(transform[:3, :3], axis=0)), 1.0, atol=1e-12
    )
