import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).parents[1]
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}


def test_release_text_has_no_machine_paths_or_training_source_dependency():
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"], text=True
    ).splitlines()
    forbidden = (
        "/" + "mnt/",
        "baseline_" + "rot_" + "corr_v2.0",
        "FRAME_" + "EVIDENCE_ROOT",
    )
    offenders = []
    for relative in tracked:
        if relative == "tests/test_eval_release_layout.py":
            continue
        path = ROOT / relative
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(relative)
    assert offenders == []


def test_eval_defaults_use_release_runtime_and_protocol_loop_policy():
    model = OmegaConf.load(ROOT / "configs/model/default.yaml")
    ours = model.abot_recon.cfg
    assert ours.ckpt == "checkpoints/abot_recon.safetensors"
    assert ours.attention_backend == "auto"
    assert "source_root" not in ours

    evaluation = OmegaConf.load(ROOT / "configs/eval.yaml")
    assert evaluation.abot_recon_loop_enabled is False
    assert "abot_recon_loop_horizon_root" not in evaluation

    adapter = (ROOT / "interfaces/abot_recon.py").read_text()
    dense_function = adapter.split("def infer_custom_reconstruction", 1)[1].split(
        "def infer_cameras_w2c", 1
    )[0]
    assert "run_abot_recon_loop_from_c2w" not in dense_function
    assert "horizonstream_root" not in adapter

    pose_launcher = (ROOT / "scripts/run_long_pose_protocol.sh").read_text()
    assert 'ABOT_RECON_LOOP_MODE="${ABOT_RECON_LOOP_MODE:-auto}"' in pose_launcher
    assert 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-paged}"' in pose_launcher
    assert 'ROPE2D_BACKEND="${ABOT_RECON_ROPE2D_BACKEND:-cuda}"' in pose_launcher
    assert "abot_recon_loop_horizon_root" not in pose_launcher

    data = OmegaConf.load(ROOT / "configs/data/mv_recon.yaml")
    assert data["Oxford-Spires-S1-I10"].cfg.expected_interval == 10
    oxford_eval = OmegaConf.load(ROOT / "configs/evaluation/mv_recon_oxford_stride1.yaml")
    assert oxford_eval.pc_eval_threshold == 4.0
    reconstruction_launcher = (ROOT / "scripts/run_mv_recon_protocol.sh").read_text()
    suite_launcher = (ROOT / "scripts/run_mv_recon_stride1_suite.sh").read_text()
    assert (
        'ATTENTION_BACKEND="${ATTENTION_BACKEND:-paged}"' in reconstruction_launcher
    )
    assert 'ROPE2D_BACKEND="${ABOT_RECON_ROPE2D_BACKEND:-cuda}"' in reconstruction_launcher
    for launcher in (reconstruction_launcher, suite_launcher):
        assert 'OXFORD_METRIC_INTERVAL="${OXFORD_METRIC_INTERVAL:-10}"' in launcher
        assert "formal Oxford protocol requires --oxford-metric-interval 10" in launcher
    assert (
        '"data.$DATA_KEY.cfg.expected_interval=$OXFORD_METRIC_INTERVAL"'
        in reconstruction_launcher
    )


def test_release_shell_launchers_parse():
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_formal_metadata_runs_without_editable_install(tmp_path):
    script = ROOT / "scripts/formal_run_metadata.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_oxford_metric_interval_is_scoped_to_oxford(tmp_path):
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"test")
    runner = ROOT / "scripts/run_mv_recon_protocol.sh"

    def dry_run(dataset):
        return subprocess.run(
            [
                "bash",
                str(runner),
                "--method",
                "abot_recon",
                "--dataset",
                dataset,
                "--out-dir",
                str(tmp_path / dataset),
                "--ckpt",
                str(checkpoint),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PY": sys.executable},
        ).stdout

    assert "expected_interval" not in dry_run("7scenes")
    assert "expected_interval" not in dry_run("tum")
    oxford_command = dry_run("oxford")
    assert "expected_interval=10" in oxford_command
    assert "pc_eval_threshold=4.0" in oxford_command
