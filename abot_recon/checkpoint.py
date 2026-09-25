from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


RELEASE_CHECKPOINT = "abot_recon.safetensors"
DEFAULT_MODEL_ID = "acvlab/ABot-Recon"
SUPPORTED_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix not in SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise ValueError(f"Unsupported checkpoint suffix {path.suffix!r}; expected: {supported}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    candidate = path / RELEASE_CHECKPOINT
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"No released checkpoint in {path}; expected: {RELEASE_CHECKPOINT}"
    )


def resolve_pretrained_checkpoint(
    pretrained_model_name_or_path: str | Path,
    *,
    filename: str = RELEASE_CHECKPOINT,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local checkpoint or download one from the Hugging Face Hub."""
    source = str(pretrained_model_name_or_path)
    local = Path(source).expanduser()
    if local.exists() or isinstance(pretrained_model_name_or_path, Path):
        return resolve_checkpoint(local)
    if local.suffix in SUPPORTED_SUFFIXES or source.startswith((".", "/", "~")):
        return resolve_checkpoint(local)
    if source.count("/") != 1:
        raise FileNotFoundError(
            f"Checkpoint is neither a local path nor a Hugging Face repo ID: {source}"
        )

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=source,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    return resolve_checkpoint(downloaded)


def _load_container(path: Path) -> Any:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def extract_state_dict(container: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(container).__name__}")
    for key in ("state_dict", "model"):
        nested = container.get(key)
        if isinstance(nested, Mapping):
            container = nested
            break
    if not container or not all(isinstance(name, str) for name in container):
        raise ValueError("Checkpoint has no non-empty string-keyed state dict")
    if not all(torch.is_tensor(value) for value in container.values()):
        raise ValueError("State dict contains non-tensor values")
    return container


def load_model_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Strictly load a complete released checkpoint."""
    selected = resolve_checkpoint(path)
    model.load_state_dict(extract_state_dict(_load_container(selected)), strict=True)


def checkpoint_has_prefix(path: str | Path, prefixes: tuple[str, ...]) -> bool:
    """Inspect the released state dict without constructing the model."""
    selected = resolve_checkpoint(path)
    state = extract_state_dict(_load_container(selected))
    return any(name.startswith(prefixes) for name in state)


def sha256(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
