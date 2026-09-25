from __future__ import annotations

import json
import math
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.state import GradientState
from accelerate.utils import GradientAccumulationPlugin, set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm

from abot_recon.checkpoint import checkpoint_has_prefix, load_model_checkpoint
from abot_recon.model import build_released_network

from .config import TrainConfig
from .checkpoint import (
    is_complete_checkpoint, latest_complete_checkpoint, save_checkpoint, training_plan,
)
from .data import MixedDataLoader, build_train_loaders, build_validation_loaders
from .ema import ModelEMA
from .loss import CameraLossConfig, Pi3Loss
from .validation import stream_validation_forward


def _sync_flags(accelerator: Accelerator, *local_flags: bool) -> list[bool]:
    """Return whether each condition is true on any distributed rank."""
    flags = torch.tensor(
        [int(flag) for flag in local_flags],
        device=accelerator.device,
        dtype=torch.int32,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    return [bool(value) for value in flags.tolist()]


def _nonfinite_gradient_names(
    model: torch.nn.Module,
    *,
    max_report: int = 16,
) -> list[str]:
    """Return a short list of parameters whose gradients contain NaN/Inf."""
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if bool(torch.isfinite(parameter.grad.detach()).all().item()):
            continue
        names.append(name)
        if len(names) >= max_report:
            break
    return names


def _force_zero_grad(optimizer) -> None:
    """Clear accumulated gradients even inside a non-sync Accelerate micro-step."""
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    base_optimizer.zero_grad(set_to_none=True)


def _finish_skipped_update(accelerator, optimizer, *, had_backward: bool, overflow: bool) -> None:
    """Discard one whole window and finish AMP state on every rank identically."""
    _force_zero_grad(optimizer)
    scaler = accelerator.scaler
    if scaler is None or not scaler.is_enabled() or not had_backward:
        return
    state = scaler.state_dict()
    scale = state["scale"] * (state["backoff_factor"] if overflow else 1.0)
    scaler.update(new_scale=scale)
    if overflow:
        state = scaler.state_dict()
        state["_growth_tracker"] = 0
        scaler.load_state_dict(state)


def _training_accelerator(cfg, accelerator=None):
    if cfg.gradient_accumulation_steps <= 0 or cfg.steps_per_epoch <= 0 or cfg.epochs <= 0:
        raise ValueError("epochs, steps_per_epoch and gradient_accumulation_steps must be positive")
    plugin = GradientAccumulationPlugin(
        num_steps=cfg.gradient_accumulation_steps, sync_with_dataloader=False,
    )
    if accelerator is None:
        accelerator = Accelerator(
            gradient_accumulation_plugin=plugin, mixed_precision=cfg.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
    else:
        # Mixed inner loaders must not flush partially accumulated updates.
        accelerator.gradient_state = GradientState(plugin)
        accelerator.step_scheduler_with_optimizer = False
    accelerator.step = 0  # load_state restores the saved counter on resume.
    return accelerator


def _step_scheduler_after_skipped_update(scheduler) -> None:
    """Advance the schedule intentionally without performing an optimizer update."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
            category=UserWarning,
        )
        scheduler.step()


def _build_scheduler(optimizer, cfg, peak_lrs, updates):
    # Match the source recipe: leave one final scheduler step unused.
    total_steps = updates + 1
    if math.isclose(cfg.onecycle_pct_start * total_steps, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("OneCycle warm-up has zero duration; adjust onecycle_pct_start or the update count")
    return OneCycleLR(
        optimizer, max_lr=peak_lrs, total_steps=total_steps,
        pct_start=cfg.onecycle_pct_start, div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
    )


def _restore_scheduler_progress(accelerator, scheduler, metadata, cfg) -> None:
    """Repair the legacy world-size-multiplied schedule without rewriting checkpoints."""
    expected = metadata.get("scheduler_step")
    if expected is None:
        # Old checkpoints did not record attempted updates (including skips).
        # Only infer progress when the metadata proves every batch was updated.
        completed = int(metadata["completed_epoch"]) * cfg.steps_per_epoch
        if cfg.gradient_accumulation_steps != 1 or metadata.get("optimizer_step") != completed:
            warnings.warn(
                "Legacy checkpoint has no scheduler_step and its correct schedule "
                "progress cannot be inferred safely; preserving the saved scheduler."
            )
            return
        expected = completed
    expected = int(expected)
    inner = scheduler.scheduler
    if inner.last_epoch == expected:
        return
    if not 0 <= expected <= inner.total_steps:
        raise ValueError(f"Invalid checkpoint scheduler_step={expected}")
    previous = inner.last_epoch
    # OneCycleLR.step(epoch) recomputes both LR and momentum at this position.
    # No optimizer step or weight/EMA mutation is performed here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The epoch parameter in `scheduler.step\(\)`.*")
        warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)`.*")
        inner.step(epoch=expected)
    inner._step_count = expected + 1
    accelerator.print(
        f"[SCHEDULER-RESUME] corrected step {previous} -> {expected}; "
        f"lr={inner.get_last_lr()} (checkpoint files unchanged)"
    )


def _resume_training(accelerator, optimizer, scheduler, cfg, peak_lrs):
    if cfg.resume_schedule not in {"strict", "restart"}:
        raise ValueError("resume_schedule must be strict or restart")
    resume = Path(cfg.resume)
    if not is_complete_checkpoint(resume):
        raise ValueError(f"Incomplete training checkpoint: {resume}")
    metadata = json.loads((resume / "training_metadata.json").read_text())
    if metadata.get("world_size", accelerator.num_processes) != accelerator.num_processes:
        accelerator.print("[RESUME] world size changed; this is not an exact RNG/data replay")
    epoch = int(metadata["completed_epoch"])
    if epoch > cfg.epochs:
        raise ValueError("epochs is smaller than the completed checkpoint epoch")
    saved_plan = metadata.get("training_plan")
    changed = saved_plan is not None and saved_plan != training_plan(cfg)
    if changed and cfg.resume_schedule == "strict":
        raise ValueError("Training plan changed; use resume_schedule=restart to start a new remaining cycle")
    schedule_keys = ("total_steps", "_schedule_phases", "_anneal_func_type")
    expected_schedule = {key: scheduler.state_dict()[key] for key in schedule_keys}
    lr_keys = ("initial_lr", "max_lr", "min_lr")
    expected_lrs = [{key: group[key] for key in lr_keys} for group in optimizer.param_groups]
    accelerator.load_state(str(resume / "trainer_state"))
    if saved_plan is None:
        actual_schedule = {key: scheduler.state_dict()[key] for key in schedule_keys}
        actual_lrs = [{key: group[key] for key in lr_keys} for group in optimizer.param_groups]
        changed = actual_schedule != expected_schedule or actual_lrs != expected_lrs
        if changed and cfg.resume_schedule == "strict":
            raise ValueError("Legacy checkpoint schedule differs; use resume_schedule=restart explicitly")
    if changed:
        remaining = (cfg.epochs - epoch) * (cfg.steps_per_epoch // cfg.gradient_accumulation_steps)
        if remaining <= 0:
            raise ValueError("A restarted schedule requires at least one remaining update")
        scheduler.scheduler = _build_scheduler(optimizer.optimizer, cfg, peak_lrs, remaining)
        accelerator.step = 0
        accelerator.print(f"[SCHEDULER-RESUME] explicitly restarted a cycle of {remaining} updates")
    else:
        _restore_scheduler_progress(accelerator, scheduler, metadata, cfg)
    remaining = (cfg.epochs - epoch) * (cfg.steps_per_epoch // cfg.gradient_accumulation_steps)
    if scheduler.state_dict()["last_epoch"] + remaining > scheduler.state_dict()["total_steps"]:
        raise ValueError("Checkpoint schedule cannot cover the remaining training plan")
    if accelerator.step % cfg.gradient_accumulation_steps:
        raise ValueError("Checkpoint was not saved at an accumulation boundary")
    return epoch, int(metadata.get("optimizer_step", epoch * cfg.steps_per_epoch))


def configure_trainable_scope(
    model: torch.nn.Module,
    scope: str,
    *,
    train_confidence: bool = False,
) -> list[str]:
    """Apply a fine-tuning scope while keeping the image encoder frozen."""
    trained = []
    head_tokens = ("decoder", "head", "gate")
    for name, parameter in model.named_parameters():
        is_encoder = name.startswith("encoder.") or ".encoder." in name
        is_confidence = name.startswith(("conf_decoder.", "conf_head.")) or any(
            token in name for token in (".conf_decoder.", ".conf_head.")
        )
        if is_encoder:
            enabled = False
        elif is_confidence:
            enabled = bool(train_confidence)
        elif scope == "all":
            enabled = True
        elif scope == "rot_correction_only":
            enabled = "rot_correction" in name
        else:
            enabled = any(token in name for token in head_tokens)
        parameter.requires_grad_(enabled)
        if enabled:
            trained.append(name)
    if not trained:
        raise RuntimeError(f"trainable_scope={scope!r} selected no parameters")
    return trained


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> AdamW:
    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    lr_by_group = {
        "gate": cfg.gate_learning_rate,
        "corr": cfg.corr_learning_rate,
        "confidence": cfg.confidence_learning_rate,
        "other": cfg.learning_rate,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "rot_correction" in name:
            kind = "corr"
        elif name.startswith(("conf_decoder.", "conf_head.")) or any(
            token in name for token in (".conf_decoder.", ".conf_head.")
        ):
            kind = "confidence"
        elif "gate" in name:
            kind = "gate"
        elif name.startswith("encoder.") or ".encoder." in name:
            raise RuntimeError(f"encoder parameter must remain frozen: {name}")
        else:
            kind = "other"
        decay = parameter.ndim > 1 and not name.endswith("bias")
        groups.setdefault((kind, decay), []).append(parameter)
    parameters = [
        {
            "params": values,
            "lr": lr_by_group[kind],
            "weight_decay": cfg.weight_decay if decay else 0.0,
            "group_name": kind,
        }
        for (kind, decay), values in groups.items()
    ]
    return AdamW(parameters, betas=(cfg.adam_beta1, cfg.adam_beta2))


def build_criterion(cfg: TrainConfig) -> Pi3Loss:
    loss = cfg.loss
    camera = CameraLossConfig(
        alpha_translation=loss.camera_alpha_translation,
        alpha_rotation=loss.camera_alpha_rotation,
        max_pair_distance=loss.camera_max_pair_distance,
        pair_mode=loss.camera_pair_mode,
        rotation_gap_weight=loss.camera_rotation_gap_weight,
        translation_gap_weight=loss.camera_translation_gap_weight,
        alpha_corr_magnitude=loss.camera_alpha_corr_magnitude,
        alpha_corr_smooth=loss.camera_alpha_corr_smooth,
    )
    return Pi3Loss(
        camera,
        camera_weight=loss.camera_weight,
        train_confidence=cfg.enable_confidence,
        confidence_weight=loss.confidence_weight,
        confidence_error_threshold=loss.confidence_error_threshold,
        confidence_invalid_as_zero=loss.confidence_invalid_as_zero,
    )


@torch.no_grad()
def validate(model, loader, criterion, steps: int, accelerator: Accelerator) -> dict[str, float]:
    model.eval()
    raw_model = accelerator.unwrap_model(model)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        with accelerator.autocast():
            prediction = stream_validation_forward(
                raw_model, torch.stack([view["img"] for view in batch], dim=1)
            )
        _, metrics = criterion(prediction, batch)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if steps > 0 and count >= steps:
            break
    reduced = {}
    count_tensor = torch.tensor(float(count), device=accelerator.device)
    global_count = accelerator.reduce(count_tensor, reduction="sum").item()
    for key in sorted(totals):
        value = torch.tensor(totals[key], device=accelerator.device)
        reduced[key] = accelerator.reduce(value, reduction="sum").item()
    return {
        f"val/{key}": value / max(global_count, 1.0)
        for key, value in reduced.items()
    }


def _publish_metrics(
    accelerator: Accelerator,
    output: Path,
    metrics: dict[str, Any],
    *, console: bool = True,
) -> None:
    # JSON null represents unavailable/nonfinite metrics, never invalid JSON NaN.
    metrics = {key: None if isinstance(value, float) and not math.isfinite(value) else value
               for key, value in metrics.items()}
    payload = json.dumps(metrics, sort_keys=True, allow_nan=False)
    if console:
        accelerator.print(payload)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


def _mean_metrics(accelerator, records):
    keys = sorted({key for record in records for key in record})
    if dist.is_available() and dist.is_initialized():
        all_keys = [None] * dist.get_world_size()
        dist.all_gather_object(all_keys, keys)
        keys = sorted({key for group in all_keys for key in group})
    values = [[record[key] for record in records if key in record and math.isfinite(record[key])]
              for key in keys]
    totals = torch.tensor([[sum(items), len(items)] for items in values],
                          dtype=torch.float64, device=accelerator.device)
    totals = accelerator.reduce(totals, reduction="sum").tolist()
    return {key: total / count if count else None for key, (total, count) in zip(keys, totals)}


def run_training(
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_loaders: dict[str, Any],
    mixed_loader: MixedDataLoader,
    *,
    validation_loader: Any | None = None,
    accelerator: Accelerator | None = None,
    criterion: Pi3Loss | None = None,
) -> list[dict[str, float]]:
    accelerator = _training_accelerator(cfg, accelerator)
    # Gradients are cleared at epoch boundaries, so do not leave a partial update.
    if cfg.steps_per_epoch % cfg.gradient_accumulation_steps:
        raise ValueError("steps_per_epoch must be divisible by gradient_accumulation_steps")
    updates_per_epoch = cfg.steps_per_epoch // cfg.gradient_accumulation_steps
    if hasattr(model, "train_conf"):
        model.train_conf = bool(cfg.enable_confidence)
    if cfg.enable_confidence and not hasattr(model, "conf_decoder"):
        raise RuntimeError("enable_confidence=true requires a checkpoint/model with confidence heads")
    configure_trainable_scope(
        model,
        cfg.trainable_scope,
        train_confidence=cfg.enable_confidence,
    )
    criterion = criterion or build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    peak_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler = _build_scheduler(optimizer, cfg, peak_lrs, cfg.epochs * updates_per_epoch)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    prepared = {}
    for name, loader in train_loaders.items():
        prepared[name] = accelerator.prepare(loader)
    mixed_loader.loaders = prepared
    validation_loaders = (
        validation_loader if isinstance(validation_loader, dict)
        else {"": validation_loader} if validation_loader is not None else {}
    )
    validation_loaders = {
        name: accelerator.prepare(loader) for name, loader in validation_loaders.items()
    }
    raw_model = accelerator.unwrap_model(model)
    start_epoch, optimizer_step = 0, 0
    if cfg.resume:
        start_epoch, optimizer_step = _resume_training(
            accelerator, optimizer, scheduler, cfg, peak_lrs,
        )
    # Initialize from restored online weights, never from the pre-resume model.
    ema = (
        ModelEMA(raw_model, decay=cfg.ema_decay, trainable_only=cfg.ema_trainable_only)
        if cfg.ema_enabled
        else None
    )
    if cfg.resume and ema is not None:
        ema_path = Path(cfg.resume) / "ema.pt"
        if ema_path.is_file():
            ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=False))
        else:
            accelerator.print("[EMA-RESUME] no EMA state; initialized from restored online weights")

    history = []
    output = Path(cfg.output_dir)
    _publish_metrics(accelerator, output, {
        "phase": "start", "config": asdict(cfg), "world_size": accelerator.num_processes,
        "torch_version": torch.__version__, "completed_epoch": start_epoch,
    }, console=False)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        mixed_loader.set_epoch(epoch)
        _force_zero_grad(optimizer)
        window_reason, window_overflow, window_had_backward = None, False, False
        window_records = []
        progress = tqdm(range(cfg.steps_per_epoch), disable=not accelerator.is_local_main_process)
        iterator = iter(mixed_loader)
        for step_in_epoch in progress:
            batch = next(iterator)
            nonfinite_grad_names: list[str] = []
            grad_norm = None
            with accelerator.accumulate(model):
                used_lrs = {f"train/lr/{group['group_name']}": group["lr"]
                            for group in optimizer.param_groups}
                images = torch.stack([view["img"] for view in batch], dim=1)
                prediction = model(images, causal_global_attn=True, long_sequence_parallel=True, streaming_inference=False)
                loss, metrics = criterion(prediction, batch)
                detached_loss = loss.detach()
                local_nonfinite_loss = not bool(
                    torch.isfinite(detached_loss).all().item()
                )
                local_loss_exceeds_max = (
                    not local_nonfinite_loss
                    and bool((detached_loss > cfg.max_loss).all().item())
                )
                nonfinite_loss, loss_exceeds_max = _sync_flags(
                    accelerator,
                    local_nonfinite_loss,
                    local_loss_exceeds_max,
                )

                # All ranks must make the same backward/no-backward decision or
                # DDP gradient collectives can hang.  Match the production
                # trainer: large finite losses are skipped, never clamped.
                if nonfinite_loss:
                    window_reason = window_reason or "nonfinite_loss"
                elif loss_exceeds_max:
                    window_reason = window_reason or "loss_exceeds_max"
                if window_reason is None:
                    accelerator.backward(loss)
                    window_had_backward = True
                    nonfinite_grad_names = _nonfinite_gradient_names(model)
                    (nonfinite_grad,) = _sync_flags(
                        accelerator, bool(nonfinite_grad_names)
                    )
                    if nonfinite_grad:
                        window_reason = "nonfinite_gradient"
                        window_overflow = True

                if window_reason is None and accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    if isinstance(grad_norm, torch.Tensor):
                        local_bad_grad_norm = not bool(
                            torch.isfinite(grad_norm.detach()).all().item()
                        )
                    else:
                        local_bad_grad_norm = not torch.isfinite(
                            torch.tensor(float(grad_norm))
                        ).item()
                    (bad_grad_norm,) = _sync_flags(
                        accelerator, local_bad_grad_norm
                    )
                    if bad_grad_norm:
                        window_reason = "nonfinite_grad_norm"
                        window_overflow = True

                if accelerator.sync_gradients:
                    if window_reason is None:
                        optimizer.step()
                        if not accelerator.optimizer_step_was_skipped:
                            scheduler.step()
                            optimizer_step += 1
                            if ema is not None:
                                ema.update(raw_model)
                        else:
                            window_reason = "amp_overflow"
                            _step_scheduler_after_skipped_update(scheduler)
                        _force_zero_grad(optimizer)
                    else:
                        _finish_skipped_update(
                            accelerator, optimizer, had_backward=window_had_backward,
                            overflow=window_overflow,
                        )
                        _step_scheduler_after_skipped_update(scheduler)
                elif window_reason is not None:
                    # Keep the whole window invalid; finalize AMP only at its boundary.
                    _force_zero_grad(optimizer)
            skip_reason = window_reason
            record = {
                key: float(value.detach())
                for key, value in metrics.items()
                if isinstance(value, torch.Tensor) and value.numel() == 1
            }
            record["finite"] = float(not nonfinite_loss)
            record["skipped"] = float(skip_reason is not None)
            record["loss_exceeds_max"] = float(loss_exceeds_max)
            record["nonfinite_gradient"] = float(
                skip_reason == "nonfinite_gradient"
            )
            record["nonfinite_grad_norm"] = float(
                skip_reason == "nonfinite_grad_norm"
            )
            if grad_norm is not None:
                record["grad_norm"] = float(grad_norm)
            history.append(record)
            window_records.append(record)
            if skip_reason is not None:
                detail = (
                    f" parameters={nonfinite_grad_names}"
                    if nonfinite_grad_names
                    else ""
                )
                accelerator.print(
                    f"[TRAIN-SKIP] epoch={epoch + 1} step={step_in_epoch + 1} "
                    f"reason={skip_reason}{detail}"
                )
            progress.set_postfix(loss=f"{record.get('loss', float('nan')):.4f}")
            if accelerator.sync_gradients:
                summary = {f"train/{key}": value
                           for key, value in _mean_metrics(accelerator, window_records).items()}
                summary.update(used_lrs)
                summary.update({
                    "phase": "train", "epoch": epoch + 1, "micro_step": step_in_epoch + 1,
                    "optimizer_step": optimizer_step, "scheduler_step": scheduler.state_dict()["last_epoch"],
                    "ema_updates": ema.num_updates if ema is not None else 0,
                    "train/skipped": int(skip_reason is not None), "skip_reason": skip_reason,
                    "amp_scale": accelerator.scaler.get_scale() if accelerator.scaler is not None else None,
                })
                _publish_metrics(accelerator, output, summary, console=False)
                window_reason, window_overflow, window_had_backward = None, False, False
                window_records = []

        if validation_loaders and (epoch + 1) % cfg.validate_every_epochs == 0:
            context = ema.apply(raw_model) if ema is not None and cfg.eval_with_ema else nullcontext()
            with context:
                stream_metrics = {
                    name: validate(model, loader, criterion, cfg.validation_steps, accelerator)
                    for name, loader in validation_loaders.items()
                }
            # The two streams use the same scenes/starts; retain their equal-weight
            # average under val/* and expose each stride separately as val/s1/* etc.
            validation_metrics = {
                key: sum(metrics[key] for metrics in stream_metrics.values()) / len(stream_metrics)
                for key in next(iter(stream_metrics.values()))
            }
            for name, metrics in stream_metrics.items():
                if name:
                    validation_metrics.update({
                        f"val/{name}/{key.removeprefix('val/')}": value
                        for key, value in metrics.items()
                    })
            validation_metrics["epoch"] = float(epoch + 1)
            validation_metrics["optimizer_step"] = float(optimizer_step)
            history.append(validation_metrics)
            _publish_metrics(accelerator, output, {"phase": "validation", **validation_metrics})
        if (epoch + 1) % cfg.checkpoint_every_epochs == 0 or epoch + 1 == cfg.epochs:
            save_checkpoint(accelerator, model, ema, output, epoch + 1, optimizer_step, scheduler, cfg)
    return history


def train(cfg: TrainConfig) -> list[dict[str, float]]:
    accelerator = _training_accelerator(cfg)
    set_seed(cfg.seed, device_specific=True)
    if cfg.auto_resume and cfg.resume is None:
        latest = latest_complete_checkpoint(cfg.output_dir)
        if latest is not None:
            cfg.resume = str(latest)
    if cfg.resume and not is_complete_checkpoint(cfg.resume):
        raise ValueError(f"Incomplete training checkpoint: {cfg.resume}")
    source = Path(cfg.resume) / "abot_recon.safetensors" if cfg.resume else cfg.pretrained
    confidence = checkpoint_has_prefix(source, ("conf_decoder.", "conf_head."))
    if cfg.enable_confidence and not confidence:
        raise RuntimeError(
            "enable_confidence=true requires pretrained confidence decoder/head weights"
        )
    model = build_released_network(
        local_window_frames=cfg.local_window_frames,
        max_frames=cfg.max_frames,
        enable_confidence=confidence,
        use_paged_kv=False,
        infer_mode="full",
        use_chunk_flex_attention=True,
        use_chunk_flex_compile=True,
    )
    if not (cfg.resume and cfg.skip_pretrained_when_resume):
        load_model_checkpoint(model, cfg.pretrained)
    model.train_conf = bool(cfg.enable_confidence)
    train_loaders, mixed = build_train_loaders(cfg.data, seed=cfg.data_seed)
    validation = (
        build_validation_loaders(cfg.data, seed=cfg.data_seed + 10_000)
        if cfg.validation_enabled
        else None
    )
    if validation is not None and any(len(loader) == 0 for loader in validation.values()):
        raise ValueError(
            "TartanGround validation is empty; set data.tartanground.eval_scenes "
            "or validation_enabled=false"
        )
    return run_training(
        cfg,
        model,
        train_loaders,
        mixed,
        validation_loader=validation,
        accelerator=accelerator,
    )
