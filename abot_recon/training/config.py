from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class LoaderConfig:
    root: str = ""
    weight: float = 1.0
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    z_far: float = 0.0
    min_interval: int = 1
    max_interval: int = 8


@dataclass
class DL3DVConfig(LoaderConfig):
    allow_repeat: bool = False
    blacklist: str | None = None


@dataclass
class TartanGroundConfig(LoaderConfig):
    exclude_scenes: list[str] | None = None
    eval_scenes: list[str] | None = None
    eval_starts_per_scene: int = 4


@dataclass
class SequenceConfig(LoaderConfig):
    """Opt-in CUT3R-format training source; no path is needed while disabled."""
    weight: float = 0.0
    split: str | None = "train"
    allow_repeat: bool = False


@dataclass
class ARKitHRConfig(SequenceConfig):
    timestamp_gap_threshold: float = 1.0
    forward_only: bool = False
    recent_stride_memory: int = 8
    fix_interval_prob: float = 0.5
    repeat_min_unique_divisor: int | None = None


@dataclass
class ARKitConfig(ARKitHRConfig):
    # Match the active development recipe; False enables original RGB-D supervision.
    camera_only: bool = True
    highres_root: str | None = None


@dataclass
class HyperSimSeqConfig(SequenceConfig):
    allow_repeat: bool = True
    # None/auto: root sidecar or bundled defaults; "": disable; path: custom list.
    sequence_blacklist_path: str | None = None
    max_rotation_deg: float = 20.0
    max_translation_factor: float = 5.0
    pose_knn: int = 48
    graph_neighbors: int = 24
    beam_width: int = 192
    min_unique_ratio: float = 0.5
    min_component_views: int = 32
    pose_load_workers: int = 1
    sequence_start_retries: int = 6
    sequence_scene_retries: int = 64


@dataclass
class BlendedMVSSeqConfig(SequenceConfig):
    allow_repeat: bool = True
    overlap_path: str | None = None
    # None/auto: root JSON or bundled TXT; "": disable the extra list.
    scene_blacklist_path: str | None = None
    upright_mask_path: str | None = None
    sideways_scene_ids: list[str] = field(default_factory=list)
    max_rotation_deg: float = 40.0
    max_translation_factor: float = 5.0
    pose_knn: int = 24
    graph_neighbors: int = 14
    beam_width: int = 96
    min_unique_ratio: float = 0.75
    max_abs_roll_deg: float = 45.0
    pose_load_workers: int = 1
    sequence_start_retries: int = 3
    sequence_scene_retries: int = 64


@dataclass
class ScanNetPPSeqConfig(SequenceConfig):
    max_interval: int = 1
    metadata_filename: str = "all_metadata.npz"
    max_starts_per_scene: int | None = None
    excluded_sequences: list[str] = field(default_factory=list)
    # None/auto: root sidecar or inline defaults; "": keep only legacy exclusions.
    sequence_blacklist_path: str | None = None
    forward_only: bool = False
    recent_stride_memory: int = 0


@dataclass
class WildRGBDConfig(SequenceConfig):
    mask_bg: bool | str = "rand"


@dataclass
class UnrealStereo4KConfig(SequenceConfig):
    split: str | None = None
    max_rotation_deg: float = 20.0
    max_translation_factor: float = 5.0
    pose_knn: int = 64
    graph_neighbors: int = 24
    beam_width: int = 128
    min_unique_ratio: float = 0.75
    # Sequential small-file reads by default; this does not change graph sampling.
    pose_load_workers: int = 1
    sequence_start_retries: int = 6
    sequence_scene_retries: int = 18


DATASET_NAMES = (
    "dl3dv", "tartanground", "tartanair", "pointodyssey", "spring", "mvs_synth",
    "dynamic_replica", "uasol", "arkit_hr", "wildrgbd", "unreal4k_seq",
    "scannet", "waymo", "vkitti2",
    "hypersim_seq", "blendedmvs_seq", "arkit", "scannetpp_seq",
)


@dataclass
class DataConfig:
    num_frames: int = 32
    height: int = 280
    width: int = 504
    aug_crop: int = 16
    aug_focal: float = 0.9
    preserve_fov_prob: float = 0.9
    principal_align_skip_prob: float = 0.0
    sequence_consistent_aug_prob: float = 0.2
    dl3dv: DL3DVConfig = field(
        default_factory=lambda: DL3DVConfig(max_interval=20, allow_repeat=False)
    )
    tartanground: TartanGroundConfig = field(
        default_factory=lambda: TartanGroundConfig(max_interval=8, z_far=80.0)
    )
    tartanair: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(split=None, allow_repeat=True, max_interval=20, z_far=80.0)
    )
    pointodyssey: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(max_interval=4, z_far=80.0)
    )
    spring: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(split=None, allow_repeat=True, max_interval=4, z_far=80.0)
    )
    mvs_synth: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(allow_repeat=True, max_interval=4)
    )
    dynamic_replica: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(max_interval=16, z_far=80.0)
    )
    uasol: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(max_interval=40, z_far=80.0)
    )
    arkit_hr: ARKitHRConfig = field(
        default_factory=lambda: ARKitHRConfig(max_interval=1, z_far=80.0)
    )
    wildrgbd: WildRGBDConfig = field(
        default_factory=lambda: WildRGBDConfig(allow_repeat=True, max_interval=4)
    )
    unreal4k_seq: UnrealStereo4KConfig = field(default_factory=UnrealStereo4KConfig)
    scannet: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(max_interval=30, z_far=80.0)
    )
    waymo: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(split=None, max_interval=8, z_far=80.0)
    )
    vkitti2: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(split=None, max_interval=5, z_far=80.0)
    )
    hypersim_seq: HyperSimSeqConfig = field(
        default_factory=lambda: HyperSimSeqConfig(z_far=80.0)
    )
    blendedmvs_seq: BlendedMVSSeqConfig = field(default_factory=BlendedMVSSeqConfig)
    arkit: ARKitConfig = field(
        default_factory=lambda: ARKitConfig(allow_repeat=True, max_interval=1, z_far=80.0)
    )
    scannetpp_seq: ScanNetPPSeqConfig = field(default_factory=ScanNetPPSeqConfig)


@dataclass
class LossConfig:
    camera_weight: float = 0.1
    camera_alpha_translation: float = 100.0
    camera_alpha_rotation: float = 1.0
    camera_max_pair_distance: int = 11
    camera_pair_mode: str = "causal_upper"
    camera_rotation_gap_weight: str = "pow0.75"
    camera_translation_gap_weight: str = "none"
    camera_alpha_corr_magnitude: float = 1e-3
    camera_alpha_corr_smooth: float = 1e-3
    confidence_weight: float = 0.05
    confidence_error_threshold: float = 0.02
    confidence_invalid_as_zero: bool = True


@dataclass
class TrainConfig:
    pretrained: str = "checkpoints/abot_recon.safetensors"
    output_dir: str = "outputs/finetune"
    resume: str | None = None
    auto_resume: bool = True
    skip_pretrained_when_resume: bool = True
    resume_schedule: str = "strict"  # strict | restart (only when the training plan changes)
    epochs: int = 20
    steps_per_epoch: int = 800
    gradient_accumulation_steps: int = 1
    trainable_scope: str = "all"
    enable_confidence: bool = False
    learning_rate: float = 1.0e-6
    gate_learning_rate: float = 1.0e-6
    corr_learning_rate: float = 1.0e-6
    confidence_learning_rate: float = 1.0e-6
    weight_decay: float = 5.0e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    onecycle_pct_start: float = 0.05
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 2.0
    max_grad_norm: float = 1.0
    max_loss: float = 10.0
    mixed_precision: str = "bf16"
    seed: int = 1111
    data_seed: int = 666
    local_window_frames: int = 12
    # RoPE3D temporal-index / paged-KV capacity; the sampled clip length is
    # ``data.num_frames``. Keep this aligned with the Wenli training recipe.
    max_frames: int = 22_000
    checkpoint_every_epochs: int = 1
    validate_every_epochs: int = 1
    validation_steps: int = -1  # -1 evaluates both fixed 128-frame streams in full
    validation_enabled: bool = True
    ema_enabled: bool = True
    ema_decay: float = 0.999
    ema_trainable_only: bool = True
    eval_with_ema: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def load_config(path: str | Path) -> TrainConfig:
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(path))
    cfg = OmegaConf.to_object(merged)
    if cfg.resume_schedule not in {"strict", "restart"}:
        raise ValueError("resume_schedule must be strict or restart")
    for name in ("epochs", "steps_per_epoch", "gradient_accumulation_steps",
                 "checkpoint_every_epochs", "validate_every_epochs"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.validation_steps != -1 and cfg.validation_steps <= 0:
        raise ValueError("validation_steps must be -1 (full validation) or positive")
    if cfg.steps_per_epoch % cfg.gradient_accumulation_steps:
        raise ValueError("steps_per_epoch must be divisible by gradient_accumulation_steps")
    for name in ("max_grad_norm", "max_loss", "onecycle_div_factor", "onecycle_final_div_factor"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < cfg.onecycle_pct_start < 1:
        raise ValueError("onecycle_pct_start must be in (0, 1)")
    if cfg.data.height % 14 or cfg.data.width % 14:
        raise ValueError("training height and width must be divisible by patch size 14")
    if cfg.max_frames < cfg.data.num_frames:
        raise ValueError("max_frames must be greater than or equal to data.num_frames")
    if cfg.validation_enabled and cfg.max_frames < 128:
        raise ValueError("max_frames must be at least 128 for the fixed validation clips")
    if not 0 <= cfg.data.sequence_consistent_aug_prob <= 1:
        raise ValueError("data.sequence_consistent_aug_prob must be in [0, 1]")
    if not 0 <= cfg.data.principal_align_skip_prob <= 1:
        raise ValueError("data.principal_align_skip_prob must be in [0, 1]")
    if cfg.trainable_scope not in {"all", "heads", "rot_correction_only"}:
        raise ValueError("trainable_scope must be all, heads, or rot_correction_only")
    if cfg.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16")
    if cfg.loss.camera_pair_mode not in {"causal_upper", "dense"}:
        raise ValueError("loss.camera_pair_mode must be causal_upper or dense")
    gap_modes = {"none", "sqrt", "pow0.75", "linear"}
    if cfg.loss.camera_rotation_gap_weight not in gap_modes:
        raise ValueError("invalid loss.camera_rotation_gap_weight")
    if cfg.loss.camera_translation_gap_weight not in gap_modes:
        raise ValueError("invalid loss.camera_translation_gap_weight")
    if not 0 <= cfg.ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if cfg.loss.confidence_weight < 0 or cfg.loss.confidence_error_threshold <= 0:
        raise ValueError("confidence weight must be >=0 and error threshold must be >0")
    for name in (
        "learning_rate",
        "gate_learning_rate",
        "corr_learning_rate",
        "confidence_learning_rate",
    ):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in DATASET_NAMES:
        item = getattr(cfg.data, name)
        if not math.isfinite(item.weight) or item.weight < 0 or item.batch_size <= 0 or item.num_workers < 0:
            raise ValueError(f"invalid weight/batch_size/num_workers for data.{name}")
        if item.weight > 0 and not item.root:
            raise ValueError(f"data.{name}.root is required when weight > 0")
        if item.min_interval < 1 or item.max_interval < item.min_interval:
            raise ValueError(f"invalid interval bounds for data.{name}")
    if not any(getattr(cfg.data, name).weight > 0 for name in DATASET_NAMES):
        raise ValueError("At least one training dataset must have weight > 0")
    if cfg.validation_enabled and not cfg.data.tartanground.root:
        raise ValueError("data.tartanground.root is required for validation")
    return cfg
