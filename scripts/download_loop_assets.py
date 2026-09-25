#!/usr/bin/env python3
"""Download the official checkpoints required by ABot-Recon loop closure."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    filename: str
    url: str


ASSETS = (
    Asset(
        "dino_salad.ckpt",
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
    ),
    Asset(
        "dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    ),
)


def download(asset: Asset, output_dir: Path, *, force: bool = False) -> Path:
    destination = output_dir / asset.filename
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {destination} already exists")
        return destination

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(asset.url, headers={"User-Agent": "ABot-Recon/1.0"})

    print(f"[download] {asset.url}")
    print(f"           -> {destination}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"downloaded an empty file from {asset.url}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official SALAD and DINOv2 loop-closure checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/loop"),
        help="destination directory (default: checkpoints/loop)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the files and URLs without downloading them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        for asset in ASSETS:
            print(f"{args.output_dir / asset.filename}\t{asset.url}")
        return 0

    try:
        for asset in ASSETS:
            download(asset, args.output_dir, force=args.force)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Loop-closure assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
