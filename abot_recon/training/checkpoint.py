"""Atomic, epoch-boundary checkpoints for single-process and DDP training."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
import re
import shutil
import uuid

import torch
import torch.distributed as dist
from safetensors.torch import save_file


def training_plan(cfg) -> dict:
    keys = ("epochs", "steps_per_epoch", "gradient_accumulation_steps",
            "onecycle_pct_start", "onecycle_div_factor", "onecycle_final_div_factor",
            "learning_rate", "gate_learning_rate", "corr_learning_rate", "confidence_learning_rate")
    return {key: getattr(cfg, key) for key in keys}


def _main_call(accelerator, operation):
    result = [None, None]
    if accelerator.is_main_process:
        try:
            result[0] = operation()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(f"Checkpoint save failed on rank 0: {result[1]}")
    return result[0]


def _check_rank_errors(error):
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError("Checkpoint save failed: " + "; ".join(failures))


def _required_files(folder, world_size=1, ema=False, scaler=False, exports=True):
    state = folder / "trainer_state"
    model = "model.safetensors" if (state / "model.safetensors").is_file() else "pytorch_model.bin"
    files = ["training_metadata.json", f"trainer_state/{model}",
             "trainer_state/optimizer.bin", "trainer_state/scheduler.bin"]
    files += [f"trainer_state/random_states_{rank}.pkl" for rank in range(world_size)]
    if exports:
        files.append("abot_recon.safetensors")
    if ema:
        files += ["ema.pt", "abot_recon_ema.safetensors"]
    if scaler:
        files.append("trainer_state/scaler.pt")
    return files


def is_complete_checkpoint(folder: str | Path) -> bool:
    """Validate committed manifests, or minimally validate legacy checkpoints."""
    folder = Path(folder)
    try:
        metadata = json.loads((folder / "training_metadata.json").read_text())
        if type(metadata["completed_epoch"]) is not int or metadata["completed_epoch"] < 0:
            return False
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            if "checkpoint_format" in metadata:
                return False
            return all((folder / name).is_file() and (folder / name).stat().st_size > 0
                       for name in _required_files(folder))
        manifest = json.loads(manifest_path.read_text())
        if manifest["version"] != 1 or type(manifest["world_size"]) is not int or manifest["world_size"] < 1:
            return False
        if type(manifest["ema_enabled"]) is not bool or type(manifest["scaler_enabled"]) is not bool:
            return False
        required = _required_files(folder, manifest["world_size"], manifest["ema_enabled"], manifest["scaler_enabled"])
        files = manifest["files"]
        if not set(required).issubset(manifest["required_files"]) or not set(manifest["required_files"]).issubset(files):
            return False
        for name, size in files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or type(size) is not int or size <= 0:
                return False
            if not (folder / path).is_file() or (folder / path).stat().st_size != size:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def latest_complete_checkpoint(output: str | Path) -> Path | None:
    candidates = []
    for path in Path(output).glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    for _, path in sorted(candidates, reverse=True):
        if is_complete_checkpoint(path):
            return path
    return None


def save_checkpoint(accelerator, model, ema, output, epoch, optimizer_step, scheduler, cfg):
    """Commit a new checkpoint without overwriting any existing final directory.

    Recoverable IO exceptions are reported to every rank. Process death and
    failures inside distributed collectives remain the launcher's responsibility.
    """
    output = Path(output)
    destination = output / f"checkpoint-{epoch:04d}"

    def begin():
        output.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary = output / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        return str(temporary)

    temporary = Path(_main_call(accelerator, begin))
    try:
        error = None
        try:
            accelerator.save_state(str(temporary / "trainer_state"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _check_rank_errors(error)

        def finish():
            def export(name):
                state = {key: value.detach().cpu().contiguous()
                         for key, value in accelerator.get_state_dict(model).items()}
                save_file(state, str(temporary / name))

            export("abot_recon.safetensors")
            if ema is not None:
                torch.save(ema.state_dict(), temporary / "ema.pt")
                with ema.apply(accelerator.unwrap_model(model)):
                    export("abot_recon_ema.safetensors")
            metadata = {"checkpoint_format": 1, "completed_epoch": epoch, "optimizer_step": optimizer_step,
                        "scheduler_step": scheduler.state_dict()["last_epoch"],
                        "training_plan": training_plan(cfg), "config": asdict(cfg),
                        "world_size": accelerator.num_processes, "ema_enabled": ema is not None}
            (temporary / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            scaler = accelerator.scaler is not None
            required = _required_files(temporary, accelerator.num_processes, ema is not None, scaler)
            files = {str(path.relative_to(temporary)): path.stat().st_size
                     for path in temporary.rglob("*") if path.is_file()}
            manifest = {"version": 1, "world_size": accelerator.num_processes,
                        "ema_enabled": ema is not None, "scaler_enabled": scaler,
                        "required_files": required, "files": files}
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if not is_complete_checkpoint(temporary):
                raise RuntimeError("Saved checkpoint is incomplete; refusing to commit")
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            os.rename(temporary, destination)

        _main_call(accelerator, finish)
    finally:
        if accelerator.is_main_process and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return destination
