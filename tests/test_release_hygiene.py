from fnmatch import fnmatchcase
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


LOCAL_ONLY_FILES = (
    "configs/finetune_tartanground_skynet.yaml",
    "tests/integration/check_train_validation.py",
)


def _is_ignored(relative, rules):
    """Match this repository's simple ignore rules in exported source archives."""
    ignored = False
    for rule in rules:
        rule = rule.strip()
        if not rule or rule.startswith("#"):
            continue
        negate = rule.startswith("!")
        pattern = rule.lstrip("!").rstrip("/")
        anchored = pattern.startswith("/")
        pattern = pattern.lstrip("/")
        parts = relative.parts
        if anchored or "/" in pattern:
            matched = fnmatchcase(relative.as_posix(), pattern)
            matched |= any(fnmatchcase("/".join(parts[:i]), pattern)
                           for i in range(1, len(parts)))
        else:
            matched = any(fnmatchcase(part, pattern) for part in parts)
        if matched:
            ignored = not negate
    return ignored


def _release_files(root):
    """Scan tracked and release-candidate files, or non-ignored archive files."""
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            check=True, capture_output=True, text=True,
        )
        return [root / name for name in result.stdout.split("\0") if name]
    rules = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [path for path in root.rglob("*")
            if path.is_file() and not _is_ignored(path.relative_to(root), rules)]


def test_release_text_has_no_experiment_machine_paths():
    forbidden = (
        "/" + "mnt/",
        "/" + "home/",
        "/" + "Users/",
        "baseline_" + "rot_" + "corr",
        "eval_" + "3R",
    )
    violations = []
    for path in _release_files(ROOT):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            if token in text:
                # Report filenames only: a failed check must not disclose private paths.
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, "\n".join(violations)


def test_machine_only_files_are_excluded_from_release_candidates():
    candidates = {path.relative_to(ROOT).as_posix() for path in _release_files(ROOT)}
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    for name in LOCAL_ONLY_FILES:
        assert _is_ignored(Path(name), rules), name
        assert name not in candidates, name
        assert f"exclude {name}" in manifest, name


def test_exported_release_candidates_include_non_python_text(tmp_path):
    (tmp_path / ".gitignore").write_text("/local.yaml\ncache/\n", encoding="utf-8")
    for name in ("README.md", "recipe.yaml", "helper.sh", "local.yaml"):
        (tmp_path / name).write_text("example", encoding="utf-8")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "generated.txt").write_text("example", encoding="utf-8")
    assert {path.name for path in _release_files(tmp_path)} == {
        ".gitignore", "README.md", "recipe.yaml", "helper.sh",
    }


def test_train_branch_has_one_training_entrypoint():
    training = ROOT / "abot_recon" / "training"
    assert (training / "cli.py").is_file()
    assert (training / "loss.py").is_file()
    assert (ROOT / "scripts" / "make_training_sample.py").is_file()
    assert (ROOT / "configs" / "finetune.yaml").is_file()
    assert not (training / "registry.py").exists()


def test_both_readmes_document_training_entrypoints_and_sample_data():
    required = (
        "scripts/make_training_sample.py",
        "scripts/merge_ema_checkpoint.py",
        "configs/finetune.yaml",
        "abot_recon/training/trainer.py",
        "tests/integration/test_real_training.py",
    )
    for name in ("README.md", "README_ZH.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert all(path in text for path in required)
        assert "TRAINING.md" not in text
    assert not (ROOT / "TRAINING.md").exists()


def test_train_branch_names_only_public_training_datasets():
    allowed = {"DL3DV", "TartanGround"}
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "abot_recon" / "training").rglob("*.py")
    )
    assert all(name in source for name in allowed)
    forbidden = ("dataset registry", "private storage", "distillation")
    assert not any(token in source.lower() for token in forbidden)


def test_release_contains_only_the_final_streaming_architecture():
    modeling = ROOT / "abot_recon" / "modeling"
    layers = modeling / "pi3" / "models" / "layers"
    assert not (modeling / ("long_" + "pi3")).exists()
    assert not (modeling / ("hybrid_" + "long_" + "pi3")).exists()
    assert (modeling / "streaming" / "network.py").is_file()
    assert (layers / "adjacent_pose_head.py").is_file()
    assert not (layers / "relative_camera_head.py").exists()
    assert not (layers / "relative_pi3_camera_head.py").exists()

    forbidden = (
        "anchor",
        "compact",
        "hybrid_" + "long_" + "pi3",
        "long_" + "pi3",
        "motion_mode",
        "use_residual_reference",
        "use_role_embed",
    )
    violations = []
    for path in (ROOT / "abot_recon").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)


def test_release_loop_backend_is_self_contained():
    forbidden = ("horizon" + "stream", "loop_" + "horizon_root")
    violations = []
    for path in (ROOT / "abot_recon").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)
    assert (ROOT / "abot_recon" / "sparse_loop" / "gpu_pgo.py").is_file()
