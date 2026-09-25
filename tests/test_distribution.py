"""Build both release artifacts without network access and check their contract."""
from pathlib import Path
from email.parser import Parser
import importlib.util
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY = (
    "configs/finetune_tartanground_skynet.yaml",
    "tests/integration/check_train_validation.py",
)


def test_source_and_wheel_distribution_contents(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("setuptools", "wheel")):
        pytest.skip("distribution checks need the pyproject build-system requirements")
    source = tmp_path / "source"
    shutil.copytree(
        ROOT, source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "__pycache__", "*.pyc", ".pytest_cache",
            ".ruff_cache", "*.egg-info", "build", "dist", "outputs", "checkpoints",
        ),
    )
    # Ensure exclusions are tested even when the local-only originals are absent.
    for name in LOCAL_ONLY:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local-only test sentinel\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_sdist, build_wheel; "
         "import sys; output = sys.argv[1]; build_sdist(output); build_wheel(output)", str(dist)],
        cwd=source, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next(dist.glob("*.tar.gz")), "r:gz") as archive:
        source_names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as archive:
        wheel_names = set(archive.namelist())
        metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    requirements = [Requirement(item) for item in metadata.get_all("Requires-Dist", [])]
    opencv = [item for item in requirements if item.name.startswith("opencv-")]
    assert len(opencv) == 2  # train and loop use the same EXR-capable distribution.
    for item in opencv:
        assert item.name == "opencv-python-headless"
        assert "4.8" in item.specifier and "5.0" not in item.specifier

    required_source = {
        ".gitignore", "README.md", "README_ZH.md", "MODEL_LICENSE.md", "NOTICE",
        "demo.py", "benchmark_comparison_transparent.png",
        "configs/finetune.yaml", "configs/finetune_smoke.yaml",
        "configs/finetune_custom.yaml", "configs/finetune_cut3r.yaml",
        "scripts/make_training_sample.py", "scripts/merge_ema_checkpoint.py",
        "scripts/download_loop_assets.py", "docs/training_datasets.md",
        "preprocess/README.md", "preprocess/requirements.txt",
        "preprocess/preprocess_scannet_seq.py", "tests/test_training.py",
        "tests/integration/test_real_training.py", "tests/integration/check_scheduler.py",
        "tests/integration/test_ddp_training.py", "tests/integration/test_real_checkpoint.py",
    }
    assert required_source <= source_names, sorted(required_source - source_names)
    resources = {
        "abot_recon/modeling/pi3/models/curope/setup.py",
        "abot_recon/modeling/pi3/models/curope/curope.cpp",
        "abot_recon/modeling/pi3/models/curope/kernels.cu",
    }
    resources.update(path.relative_to(ROOT).as_posix() for path in
                     (ROOT / "abot_recon/training/datasets/metadata").glob("*.txt"))
    assert resources <= source_names, sorted(resources - source_names)
    assert resources <= wheel_names, sorted(resources - wheel_names)
    for name in LOCAL_ONLY:
        assert name not in source_names
        assert not any(path.endswith(name) for path in wheel_names)
    for name in ("LICENSE", "NOTICE", "MODEL_LICENSE.md", "THIRD_PARTY_NOTICES.md"):
        assert any(path.endswith("/" + name) for path in wheel_names), name
    for path in (ROOT / "licenses").glob("*.txt"):
        assert "licenses/" + path.name in source_names
        assert any(name.endswith("/licenses/" + path.name) for name in wheel_names)
    assert not any(name.startswith(("configs/", "scripts/", "tests/", "preprocess/"))
                   for name in wheel_names)
