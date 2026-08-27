import sys
import types

import numpy as np
import pytest
import torch

from interfaces.ttt3r import (
    _forward_recurrent_predictions,
    _poses_from_predictions,
    _validate_reset_interval,
    assert_views_never_reset,
    matrix_cumprod,
    pose_only_downstream_head,
    prepare_pointcloud_output_chunked,
)


@pytest.mark.parametrize(
    ("update_type", "expected"),
    [("cut3r", "full"), ("ttt3r", "lighter")],
)
def test_pointcloud_recurrent_dispatch_uses_official_update_rule(update_type, expected):
    calls = []

    class FakeNet:
        config = types.SimpleNamespace(model_update_type=update_type)

        def forward_recurrent(self, views, device, ret_state):
            calls.append("full")
            return ["cut"], None

        def forward_recurrent_lighter(self, views, device, ret_state):
            calls.append("lighter")
            return ["ttt"], None

    output = _forward_recurrent_predictions(FakeNet(), iter(()), "cuda")
    assert calls == [expected]
    assert output == (["cut"] if expected == "full" else ["ttt"])


def test_reset_interval_validation():
    assert _validate_reset_interval(None) is None
    assert _validate_reset_interval(0) is None
    assert _validate_reset_interval(100) == 100


def test_pose_reset_stitching_matches_official_overlap_protocol():
    def pose_at_x(value):
        pose = torch.eye(4)[None]
        pose[:, 0, 3] = value
        return {"camera_pose": pose}

    # Frames 0,1 form chunk one. The duplicate of frame 1 is dropped; frames
    # 2,3 are local to that duplicate and get left-multiplied by chunk-one base.
    predictions = [
        pose_at_x(0.0),
        pose_at_x(1.0),
        pose_at_x(0.0),
        pose_at_x(1.0),
        pose_at_x(2.0),
    ]
    poses = _poses_from_predictions(
        predictions,
        reset_flags=[False, True, False, False, True],
        pose_decoder=lambda value: value,
    )
    assert poses[:, 0, 3].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_matrix_cumprod_composes_transforms_in_sequence_order():
    first = torch.eye(4)
    first[0, 3] = 2.0
    second = torch.eye(4)
    second[1, 3] = 3.0
    products = matrix_cumprod(torch.stack([first, second]))

    torch.testing.assert_close(products[0], first)
    torch.testing.assert_close(products[1], first @ second)


def test_prepare_pointcloud_output_chunked_matches_pose_and_nearest_resize(monkeypatch):
    camera_module = types.ModuleType("dust3r.utils.camera")
    camera_module.pose_encoding_to_camera = lambda value: value
    geometry_module = types.ModuleType("dust3r.utils.geometry")

    def geotrf(pose, points):
        return points @ pose[..., :3, :3].transpose(-1, -2) + pose[..., None, :3, 3]

    geometry_module.geotrf = geotrf
    monkeypatch.setitem(sys.modules, "dust3r.utils.camera", camera_module)
    monkeypatch.setitem(sys.modules, "dust3r.utils.geometry", geometry_module)

    pose0 = torch.eye(4)[None]
    pose1 = torch.eye(4)[None]
    pose1[:, 0, 3] = 10.0
    local0 = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, -1.0]]]])
    local1 = local0 + torch.tensor([0.0, 1.0, 0.0])
    outputs = {
        "pred": [
            {"camera_pose": pose0, "pts3d_in_self_view": local0},
            {"camera_pose": pose1, "pts3d_in_self_view": local1},
        ],
        "views": [
            {"reset": torch.tensor([False])},
            {"reset": torch.tensor([False])},
        ],
    }

    points, poses, mask = prepare_pointcloud_output_chunked(
        outputs, target_hw=(2, 4), chunk_size=1
    )

    assert points.shape == (2, 2, 4, 3)
    assert mask.shape == (2, 2, 4)
    np.testing.assert_allclose(poses, torch.cat([pose0, pose1]).numpy())
    np.testing.assert_allclose(points[0, 0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(points[1, 0, 0], [11.0, 3.0, 3.0])
    assert mask[0, 0].tolist() == [True, True, False, False]
    assert outputs["pred"] == [None, None]


def test_long_pose_views_must_never_reset():
    assert_views_never_reset([{"reset": torch.tensor([False])} for _ in range(8)])
    with pytest.raises(ValueError, match="unexpectedly resets"):
        assert_views_never_reset(
            [{"reset": torch.tensor([False])}, {"reset": torch.tensor([True])}]
        )


def test_prepare_pointcloud_output_selects_dense_frames_but_keeps_all_poses(monkeypatch):
    camera_module = types.ModuleType("dust3r.utils.camera")
    camera_module.pose_encoding_to_camera = lambda value: value
    geometry_module = types.ModuleType("dust3r.utils.geometry")
    geometry_module.geotrf = lambda pose, points: (
        points @ pose[..., :3, :3].transpose(-1, -2)
        + pose[..., None, :3, 3]
    )
    monkeypatch.setitem(sys.modules, "dust3r.utils.camera", camera_module)
    monkeypatch.setitem(sys.modules, "dust3r.utils.geometry", geometry_module)

    poses = []
    predictions = []
    views = []
    for index in range(5):
        pose = torch.eye(4)[None]
        pose[:, 0, 3] = index
        poses.append(pose)
        predictions.append(
            {
                "camera_pose": pose,
                "pts3d_in_self_view": torch.ones(1, 1, 2, 3),
            }
        )
        views.append({"reset": torch.tensor([False])})
    outputs = {"pred": predictions, "views": views}
    points, full_poses, mask = prepare_pointcloud_output_chunked(
        outputs,
        target_hw=(1, 2),
        output_indices=[0, 2, 4],
    )
    assert points.shape[0] == mask.shape[0] == 3
    assert full_poses.shape[0] == 5
    assert points[:, 0, 0, 0].tolist() == [1.0, 3.0, 5.0]


def test_pose_only_head_uses_official_pose_branch_and_restores(monkeypatch):
    postprocess_module = types.ModuleType("dust3r.heads.postprocess")
    postprocess_module.postprocess_pose = lambda value, mode: value + 3.0
    monkeypatch.setitem(sys.modules, "dust3r.heads.postprocess", postprocess_module)

    class FakeHead(torch.nn.Module):
        has_pose = True
        pose_mode = "fake"

        def __init__(self):
            super().__init__()
            self.pose_head = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.eye_(self.pose_head.weight)

        def forward(self, decout, img_shape, **kwargs):
            return {"dense": torch.ones(1, 100, 100, 3)}

    downstream_head = FakeHead()
    net = types.SimpleNamespace(
        downstream_head=downstream_head,
        head=lambda decout, img_shape, **kwargs: downstream_head(
            decout, img_shape, **kwargs
        ),
    )
    original_forward = downstream_head.forward.__func__
    decout = [torch.zeros(1, 2, 2), torch.tensor([[[1.0, 2.0], [9.0, 9.0]]])]
    with pose_only_downstream_head(net):
        output = net.head(decout, (16, 16))
        assert set(output) == {"camera_pose"}
        torch.testing.assert_close(output["camera_pose"], torch.tensor([[4.0, 5.0]]))
    assert downstream_head.forward.__func__ is original_forward
