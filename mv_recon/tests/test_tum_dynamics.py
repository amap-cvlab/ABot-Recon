import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from datasets.tum_dynamics import TUMDynamics90, TUMDynamicsFull, _associate_one_to_one
from mv_recon.lingbot_protocol import get_dataset_eval_options, resolve_pc_gt_load_img_size


def _write_index(path: Path, title: str, rows):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n")
        for timestamp, relative_path in rows:
            handle.write(f"{timestamp:.6f} {relative_path}\n")


def _make_fake_sequence(root: Path) -> str:
    sequence_name = "rgbd_dataset_freiburg3_walking_fake"
    sequence = root / sequence_name
    (sequence / "rgb").mkdir(parents=True)
    (sequence / "rgb_90").mkdir()
    (sequence / "depth").mkdir()

    rgb_timestamps = [1.0, 2.0, 3.0, 4.0]
    for i, timestamp in enumerate(rgb_timestamps):
        rgb = np.full((48, 64, 3), 20 + i, dtype=np.uint8)
        name = f"{timestamp:.6f}.png"
        Image.fromarray(rgb).save(sequence / "rgb" / name)
        Image.fromarray(rgb).save(sequence / "rgb_90" / name)
    _write_index(
        sequence / "rgb.txt",
        "rgb",
        [(ts, f"rgb/{ts:.6f}.png") for ts in rgb_timestamps],
    )

    depth_rows = [(1.0001, "depth/1.000100.png"), (2.010, "depth/2.010000.png"), (3.0002, "depth/3.000200.png")]
    for timestamp, relative_path in depth_rows:
        raw = np.full((48, 64), 10000, dtype=np.uint16)  # 2 metres
        imageio.imwrite(sequence / relative_path, raw)
    _write_index(sequence / "depth.txt", "depth", depth_rows)

    with (sequence / "groundtruth_90.txt").open("w", encoding="utf-8") as handle:
        for i, timestamp in enumerate(rgb_timestamps):
            # c2w translation x=i; identity quaternion.
            handle.write(f"{timestamp + 0.001:.6f} {i} 0 0 0 0 0 1\n")
    with (sequence / "groundtruth.txt").open("w", encoding="utf-8") as handle:
        handle.write("# full ground truth\n")
        for i, timestamp in enumerate(rgb_timestamps):
            handle.write(f"{timestamp + 0.001:.6f} {i} 0 0 0 0 0 1\n")
    return sequence_name


def test_one_to_one_association_does_not_reuse_depth():
    matches = _associate_one_to_one([1.000, 1.004], [1.002], max_difference=0.005)
    assert len(matches) == 1
    assert list(matches.values())[0][0] == 0


def test_tum_dataset_strict_depth_mask_pose_and_audit(tmp_path):
    sequence_name = _make_fake_sequence(tmp_path)
    dataset = TUMDynamics90(
        TUM_DIR=str(tmp_path),
        load_img_size=0,
        expected_num_frames=4,
        min_depth_matches=2,
        depth_max_dt=0.005,
    )
    assert dataset.sequence_list == [sequence_name]
    assert dataset.metadata[sequence_name]["num_depth_matches"] == 2

    data = dataset.get_data(sequence_name=sequence_name, ids=[0, 1, 2, 3])
    assert tuple(data["images"].shape) == (4, 3, 48, 64)
    assert data["has_depth"].tolist() == [True, False, True, False]
    assert data["valid_mask"][0].all()
    assert not data["valid_mask"][1].any()
    assert data["valid_mask"][2].all()
    assert not data["valid_mask"][3].any()
    np.testing.assert_allclose(data["intrs"][0].numpy(), [[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    # Ground-truth rows are c2w, returned extrinsics are w2c.
    np.testing.assert_allclose(data["extrs"][3, :3, 3].numpy(), [-3.0, 0.0, 0.0])

    audit_path = tmp_path / "audit.json"
    dataset.write_depth_association_audit(str(audit_path))
    audit = json.loads(audit_path.read_text())
    assert audit["sequences"][sequence_name]["num_depth_matches"] == 2
    assert audit["sequences"][sequence_name]["frames"][1]["depth"] is None


def test_tum_protocol_is_explicit():
    options = get_dataset_eval_options("TUM-Dynamics-90")
    assert options == {
        "icp_threshold": 0.1,
        "voxel_size": 4.0 / 512.0,
        "eval_threshold": 0.25,
    }
    assert resolve_pc_gt_load_img_size("TUM-Dynamics-90", True, 518) == 0


def test_tum_full_uses_rgb_index_and_timestamp_associated_poses(tmp_path):
    sequence_name = _make_fake_sequence(tmp_path)
    dataset = TUMDynamicsFull(
        TUM_DIR=str(tmp_path),
        load_img_size=0,
        min_depth_match_ratio=0.5,
        depth_max_dt=0.005,
    )
    assert dataset.get_seq_framenum(sequence_name=sequence_name) == 4
    assert [Path(path).parent.name for path in dataset.metadata[sequence_name]["rgb_paths"]] == [
        "rgb", "rgb", "rgb", "rgb"
    ]
    data = dataset.get_data(sequence_name=sequence_name, ids=[0, 3])
    assert data["ind"].tolist() == [0, 3]
    np.testing.assert_allclose(data["extrs"][1, :3, 3].numpy(), [-3.0, 0.0, 0.0])
    assert get_dataset_eval_options("TUM-Dynamics-Full") == get_dataset_eval_options(
        "TUM-Dynamics-90"
    )
    assert resolve_pc_gt_load_img_size("TUM-Dynamics-Full", True, 518) == 0
