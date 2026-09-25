from __future__ import annotations

import urllib.request
from pathlib import Path


SALAD_CHECKPOINT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
DINO_VITB14_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def ensure_file(path: str | Path, url: str, *, auto_download: bool) -> Path:
    target = Path(path).expanduser().resolve()
    if target.is_file():
        return target
    if not auto_download:
        raise FileNotFoundError(f"required loop-closure asset not found: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(target)
    return target


def ensure_loop_assets(
    salad_checkpoint: str | Path,
    dino_checkpoint: str | Path,
    *,
    auto_download: bool,
) -> tuple[Path, Path]:
    return (
        ensure_file(salad_checkpoint, SALAD_CHECKPOINT_URL, auto_download=auto_download),
        ensure_file(dino_checkpoint, DINO_VITB14_URL, auto_download=auto_download),
    )
