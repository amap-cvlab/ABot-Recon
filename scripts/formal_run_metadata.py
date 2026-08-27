#!/usr/bin/env python3
"""Write immutable provenance and standardized result links for a formal run."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))



def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256(path: Path) -> str:
    """Hash the complete checkpoint selected by the release loader."""
    from abot_recon.checkpoint import resolve_checkpoint

    selected = resolve_checkpoint(path)
    digest = sha256_file(selected)
    return digest if path.is_file() else f"{selected.name} sha256={digest}"


def git_revision(path: Path) -> str:
    if not path.exists():
        return f"missing:{path}"
    try:
        commit = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit + (" dirty" if dirty else " clean")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f"not-a-git-checkout:{path}"


def unique_file(root: Path, pattern: str) -> Path:
    candidates = sorted(
        path
        for path in root.rglob(pattern)
        if path.parent != root and "runtime_manifests" not in path.parts
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern} below {root}, found {len(candidates)}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def start(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    readme = Path(args.eval_root) / "README_ONLINE_EVAL_PROTOCOL_ZH.md"
    protocol_hash = sha256_file(readme)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = args.checkpoint_sha256 or checkpoint_sha256(checkpoint)
    atomic_text(
        out / "protocol_version.txt", f"README_ONLINE_EVAL_PROTOCOL_ZH.md sha256={protocol_hash}"
    )
    atomic_text(out / "git_commit_eval3r.txt", git_revision(Path(args.eval_root)))
    atomic_text(out / "git_commit_model_repo.txt", git_revision(Path(args.model_root)))
    atomic_text(out / "checkpoint_path.txt", str(checkpoint))
    atomic_text(out / "checkpoint_sha256.txt", checkpoint_hash)
    atomic_text(out / "command.sh", args.command)


def finish(args) -> None:
    out = Path(args.out_dir)
    config_candidates = sorted(out.glob("hydra/.hydra/config.yaml"))
    if len(config_candidates) != 1:
        raise RuntimeError(f"Missing unique Hydra run config below {out}")
    shutil.copy2(config_candidates[0], out / "run_config.yaml")
    shutil.copy2(unique_file(out, "_all_samples.csv"), out / "per_sequence_metrics.csv")
    metric_candidates = sorted(path for path in out.rglob("*-metric*.csv") if path.parent != out)
    if len(metric_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one aggregate metric CSV below {out}, "
            f"found {[str(path) for path in metric_candidates]}"
        )
    shutil.copy2(metric_candidates[0], out / "aggregate_metrics.csv")
    shutil.copy2(unique_file(out, "runtime_manifest.csv"), out / "runtime_manifest.csv")
    atomic_text(out / "formal_run_complete.txt", "complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "finish"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-root")
    parser.add_argument("--model-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--command", default="")
    args = parser.parse_args()
    if args.phase == "start":
        for name in ("eval_root", "model_root", "checkpoint"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for start")
        start(args)
    else:
        finish(args)


if __name__ == "__main__":
    main()
