from pathlib import Path

import numpy as np
import torch

import abot_recon.loop_closure as loop_module
from abot_recon.config import InferenceConfig
from abot_recon.sparse_loop.retrieval import suppress_nearby_candidates
from abot_recon.sparse_loop.sparse_keyframes import (
    interpolate_corrections,
    prepare_sparse_graph,
)
from abot_recon.sparse_loop.types import LoopCandidate


def test_disabled_loop_is_identity():
    poses = torch.eye(4).repeat(3, 1, 1)
    output, info = loop_module.apply_loop_closure(
        object(),
        ["0.jpg", "1.jpg", "2.jpg"],
        poses,
        device=torch.device("cpu"),
        loop_cfg=loop_module.LoopClosureConfig(enabled=False),
    )
    assert torch.equal(output, poses)
    assert info == {"enabled": False}


def test_public_loop_adapter_uses_overridable_assets(monkeypatch, tmp_path):
    captured = {}

    def fake_apply(model, paths, poses, **kwargs):
        captured.update(kwargs)
        return poses, {"enabled": True}

    monkeypatch.setattr(loop_module, "apply_loop_closure", fake_apply)
    config = InferenceConfig(
        device="cpu",
        amp_dtype="fp32",
        loop_salad_checkpoint=tmp_path / "salad.ckpt",
        loop_dino_checkpoint=tmp_path / "dino.pth",
        loop_output_dir=tmp_path / "output",
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    output = loop_module.refine_trajectory([Path("0.jpg"), Path("1.jpg")], poses, object(), config)
    assert torch.equal(output, poses)
    loop_cfg = captured["loop_cfg"]
    assert loop_cfg.salad_checkpoint == str(tmp_path / "salad.ckpt")
    assert loop_cfg.dino_checkpoint == str(tmp_path / "dino.pth")
    assert captured["save_dir"] == tmp_path / "output"


def test_pose_graph_optimization_reenables_grad_inside_inference_mode(monkeypatch):
    captured = {}
    sentinel = np.repeat(np.eye(4)[None], 2, axis=0)

    def fake_optimize(poses, odometry, loop_edges, config, runtime_stats):
        captured["grad_enabled"] = torch.is_grad_enabled()
        captured["inference_mode"] = torch.is_inference_mode_enabled()
        captured["odometry_count"] = len(odometry)
        return sentinel

    monkeypatch.setattr(loop_module, "optimize_pose_graph", fake_optimize)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    with torch.inference_mode():
        output = loop_module._optimize_pose_graph_with_grad(
            poses, [], loop_module.PoseGraphConfig(), {}
        )

    assert output is sentinel
    assert captured == {
        "grad_enabled": True,
        "inference_mode": False,
        "odometry_count": 1,
    }


def test_candidate_suppression_matches_sparse_loop_policy():
    candidates = [
        LoopCandidate(100, 10, 100, 10, 0.95),
        LoopCandidate(105, 12, 105, 12, 0.94),
        LoopCandidate(200, 60, 200, 60, 0.93),
    ]
    kept = suppress_nearby_candidates(candidates, radius=10, limit=10)
    assert [(item.src_pos, item.dst_pos) for item in kept] == [(100, 10), (200, 60)]


def test_relative_measurement_convention():
    source = np.eye(4)
    destination = np.eye(4)
    destination[0, 3] = 2.0
    measured = loop_module.relative_measurement(source, destination)
    np.testing.assert_array_equal(measured, np.linalg.inv(destination) @ source)


def test_sparse_graph_retains_regular_nodes_and_loop_endpoints():
    poses = np.repeat(np.eye(4)[None], 101, axis=0)
    edge = loop_module.LoopEdge(
        src_pos=73,
        dst_pos=17,
        src_frame=73,
        dst_frame=17,
        score=0.9,
        inliers=128,
        method="test",
        transform_ji=np.eye(4),
    )
    ids, graph_poses, edges = prepare_sparse_graph(poses, [edge], stride=50)
    np.testing.assert_array_equal(ids, [0, 17, 50, 73, 100])
    assert graph_poses.shape == (5, 4, 4)
    assert (edges[0].src_pos, edges[0].dst_pos) == (3, 1)


def test_sparse_corrections_are_smooth_and_exact_at_keyframes():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = np.arange(5)
    keyframes = np.asarray([0, 4])
    optimized = poses[keyframes].copy()
    optimized[1, 1, 3] = 2.0
    output = interpolate_corrections(poses, keyframes, optimized)
    np.testing.assert_allclose(output[keyframes], optimized)
    np.testing.assert_allclose(output[:, 1, 3], np.linspace(0, 2, 5))
