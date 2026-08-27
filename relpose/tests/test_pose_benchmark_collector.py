import csv
import json
import subprocess
import sys
from pathlib import Path


def test_collector_deduplicates_resumed_sequence_rows(tmp_path):
    metrics = tmp_path / "metrics/cut3r_kitti_s1_sim3/x/kitti-long"
    metrics.mkdir(parents=True)
    with (metrics / "seq_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "dataset", "seq", "ATE", "RPE trans", "RPE rot"]
        )
        writer.writeheader()
        writer.writerow({"model": "cut3r", "dataset": "kitti-long", "seq": "00", "ATE": 9, "RPE trans": 3, "RPE rot": 2})
        writer.writerow({"model": "cut3r", "dataset": "kitti-long", "seq": "00", "ATE": 1, "RPE trans": 0.3, "RPE rot": 0.2})
    fps = tmp_path / "fps/cut3r/x"
    fps.mkdir(parents=True)
    (fps / "forward_timing_summary.json").write_text(json.dumps({"forward_fps": 12.5}))
    status = tmp_path / "status"
    status.mkdir()
    (status / "cut3r__kitti.done").touch()

    script = Path(__file__).parents[2] / "scripts/collect_pose_benchmark.py"
    subprocess.run([sys.executable, str(script), str(tmp_path)], check=True)
    rows = list(csv.DictReader((tmp_path / "pose_metrics_fps_comparison.csv").open()))
    row = next(row for row in rows if row["model"] == "cut3r" and row["dataset"] == "kitti-long")
    assert row["status"] == "complete"
    assert row["sequences"] == "1"
    assert float(row["ATE"]) == 1.0
    assert float(row["RPE-r_deg"]) == 0.2
    assert float(row["RPE-t"]) == 0.3
    assert float(row["forward_FPS_32frame_VBR"]) == 12.5
