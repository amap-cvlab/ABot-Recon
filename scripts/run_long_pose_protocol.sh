#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
METHOD="${METHOD:-}"
DATASET="${DATASET:-}"
STRIDE="${STRIDE:-1}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-}"
CKPT="${CKPT:-}"
RESUME_EXISTING="${RESUME_EXISTING:-true}"
HORIZON_AMP_DTYPE="${HORIZON_AMP_DTYPE:-fp16}"
HORIZON_LOOP_MODE="${HORIZON_LOOP_MODE:-off}"
HORIZON_LOOP_PRESET="${HORIZON_LOOP_PRESET:-auto}"
HORIZON_LOOP_RESUME_EXISTING="${HORIZON_LOOP_RESUME_EXISTING:-false}"
ABOT_RECON_LOOP_MODE="${ABOT_RECON_LOOP_MODE:-auto}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-paged}"
ROPE2D_BACKEND="${ABOT_RECON_ROPE2D_BACKEND:-cuda}"
CUT3R_TTT3R_RESET_INTERVAL="${CUT3R_TTT3R_RESET_INTERVAL:-100}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$ROOT/third_party}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$OFFICIAL_ROOT/LingBot-Map}"
HS_ROOT="${HS_ROOT:-$ROOT/third_party/HorizonStream}"
HORIZON_LOOP_ASSET_ROOT="${HORIZON_LOOP_ASSET_ROOT:-$ROOT/checkpoints/loop}"
HORIZON_SALAD_CKPT="${HORIZON_SALAD_CKPT:-$HORIZON_LOOP_ASSET_ROOT/dino_salad.ckpt}"
HORIZON_SALAD_DINO_WEIGHTS="${HORIZON_SALAD_DINO_WEIGHTS:-$HORIZON_LOOP_ASSET_ROOT/dinov2_vitb14_pretrain.pth}"
CONFIG_CHECK="${CONFIG_CHECK:-false}"
DRY_RUN="${DRY_RUN:-false}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
SEQUENCE="${SEQUENCE:-}"
MEASURE_FORWARD_FPS="${MEASURE_FORWARD_FPS:-false}"
FPS_FRAMES="${FPS_FRAMES:-0}"
FPS_PRELOAD_LIMIT="${FPS_PRELOAD_LIMIT:-256}"
FPS_WARMUP_RUNS="${FPS_WARMUP_RUNS:-1}"
FPS_WARMUP_FRAMES="${FPS_WARMUP_FRAMES:-64}"

usage() {
  cat <<'EOF'
Unified pose-only ATE/RPE evaluation on KITTI/VBR/Oxford.

Required:
  --method cut3r|ttt3r|longstream|infinitevggt|ovggt|stream3r_window5|horizon|lingbot|abot_recon
  --dataset kitti|vbr|oxford
  --out-dir PATH
  --ckpt PATH

Optional:
  --stride N                 Default: 1
  --gpu INDEX                Default: 0
  --resume-existing true|false
  --horizon-amp-dtype bf16|fp16|fp32|auto
  --horizon-loop-mode off|both
                              both emits no-loop and SALAD+PGO loop metrics
  --horizon-loop-preset auto|kitti|vbr|generic
                              auto uses KITTI/VBR presets and generic Oxford
  --horizon-loop-resume-existing true|false
                              Default: false; enable only for an unchanged loop config
  --horizon-salad-ckpt PATH
  --horizon-salad-dino-weights PATH
  --abot-recon-loop-mode auto|both|off
                              auto/both run one no-loop model forward, then
                              reuse its cached poses for loop closure (default)
  --attention-backend auto|paged|sdpa
                              ABot-Recon only; auto prefers paged FlashInfer
  --reset-interval N        CUT3R/TTT3R reset interval; 0=no-reset, default=100
  --max-eval-frames N        Smoke/debug only; 0 runs complete sequences
  --sequence NAME            Restrict to one exact sequence (smoke/debug)
  --measure-forward-fps      Time model forward only; excludes image preprocessing
  --fps-frames N             Frames used by FPS measurement (required with flag)
  --fps-preload-limit N      Safety limit for lazy streaming inputs (default: 256)
  --fps-warmup-runs N        Untimed full-sequence warmups (default: 1)
  --fps-warmup-frames N      Frames in each untimed warmup (default: 64)
  --official-root PATH       Root containing official baseline source trees
  --lingbot-root PATH        LingBot-Map source tree
  --config-check             Resolve Hydra config only
  --dry-run                  Print command only

Each model keeps its native released RGB preprocessing. Metrics are per-scene
Sim3 ATE/RPE-t/RPE-r with delta=1 on the forwarded frame sequence.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --resume-existing) RESUME_EXISTING="$2"; shift 2 ;;
    --horizon-amp-dtype) HORIZON_AMP_DTYPE="$2"; shift 2 ;;
    --horizon-loop-mode) HORIZON_LOOP_MODE="$2"; shift 2 ;;
    --horizon-loop-preset) HORIZON_LOOP_PRESET="$2"; shift 2 ;;
    --horizon-loop-resume-existing) HORIZON_LOOP_RESUME_EXISTING="$2"; shift 2 ;;
    --horizon-salad-ckpt) HORIZON_SALAD_CKPT="$2"; shift 2 ;;
    --horizon-salad-dino-weights) HORIZON_SALAD_DINO_WEIGHTS="$2"; shift 2 ;;
    --abot-recon-loop-mode) ABOT_RECON_LOOP_MODE="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --reset-interval) CUT3R_TTT3R_RESET_INTERVAL="$2"; shift 2 ;;
    --max-eval-frames) MAX_EVAL_FRAMES="$2"; shift 2 ;;
    --sequence) SEQUENCE="$2"; shift 2 ;;
    --measure-forward-fps) MEASURE_FORWARD_FPS=true; shift ;;
    --fps-frames) FPS_FRAMES="$2"; shift 2 ;;
    --fps-preload-limit) FPS_PRELOAD_LIMIT="$2"; shift 2 ;;
    --fps-warmup-runs) FPS_WARMUP_RUNS="$2"; shift 2 ;;
    --fps-warmup-frames) FPS_WARMUP_FRAMES="$2"; shift 2 ;;
    --official-root) OFFICIAL_ROOT="$2"; shift 2 ;;
    --lingbot-root) LINGBOT_ROOT="$2"; shift 2 ;;
    --config-check) CONFIG_CHECK=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "$METHOD" =~ ^(cut3r|ttt3r|longstream|infinitevggt|ovggt|stream3r_window5|horizon|lingbot|abot_recon)$ ]] || die "invalid --method=$METHOD"
[[ "$DATASET" =~ ^(kitti|vbr|oxford)$ ]] || die "invalid --dataset=$DATASET"
[[ "$STRIDE" =~ ^[1-9][0-9]*$ ]] || die "--stride must be positive"
[[ "$RESUME_EXISTING" == true || "$RESUME_EXISTING" == false ]] || die "invalid --resume-existing"
[[ "$HORIZON_AMP_DTYPE" =~ ^(auto|bf16|fp16|fp32)$ ]] || die "invalid Horizon dtype"
[[ "$HORIZON_LOOP_MODE" =~ ^(off|both)$ ]] || die "invalid --horizon-loop-mode"
[[ "$HORIZON_LOOP_PRESET" =~ ^(auto|kitti|vbr|generic)$ ]] || die "invalid --horizon-loop-preset"
[[ "$HORIZON_LOOP_RESUME_EXISTING" == true || "$HORIZON_LOOP_RESUME_EXISTING" == false ]] || die "invalid --horizon-loop-resume-existing"
if [[ "$HORIZON_LOOP_MODE" == both && "$METHOD" != horizon ]]; then
  die "--horizon-loop-mode=both is only valid with --method=horizon"
fi
[[ "$ABOT_RECON_LOOP_MODE" =~ ^(auto|both|on|off)$ ]] || die "invalid --abot-recon-loop-mode"
[[ "$ATTENTION_BACKEND" =~ ^(auto|paged|sdpa)$ ]] || die "invalid --attention-backend"
[[ "$ROPE2D_BACKEND" =~ ^(auto|cuda|torch)$ ]] || die "invalid ABOT_RECON_ROPE2D_BACKEND"
if [[ "$METHOD" != abot_recon ]]; then
  if [[ "$ABOT_RECON_LOOP_MODE" == auto ]]; then
    ABOT_RECON_LOOP_MODE=off
  elif [[ "$ABOT_RECON_LOOP_MODE" != off ]]; then
    die "--abot-recon-loop-mode is only valid with --method=abot_recon"
  fi
fi
# The formal ABot-Recon forward always emits the no-loop trajectory. Loop
# closure is a separate cached-pose post-process, matching HorizonStream.
ABOT_RECON_LOOP_ENABLED=false
ABOT_RECON_RUN_LOOP=false
[[ "$METHOD" == abot_recon && "$ABOT_RECON_LOOP_MODE" != off ]] && ABOT_RECON_RUN_LOOP=true

[[ "$CUT3R_TTT3R_RESET_INTERVAL" =~ ^[0-9]+$ ]] || die "--reset-interval must be non-negative"
[[ "$MAX_EVAL_FRAMES" =~ ^[0-9]+$ ]] || die "--max-eval-frames must be non-negative"
[[ "$FPS_FRAMES" =~ ^[0-9]+$ ]] || die "--fps-frames must be non-negative"
[[ "$FPS_PRELOAD_LIMIT" =~ ^[1-9][0-9]*$ ]] || die "--fps-preload-limit must be positive"
[[ "$FPS_WARMUP_RUNS" =~ ^[0-9]+$ ]] || die "--fps-warmup-runs must be non-negative"
[[ "$FPS_WARMUP_FRAMES" =~ ^[1-9][0-9]*$ ]] || die "--fps-warmup-frames must be positive"
if [[ "$MEASURE_FORWARD_FPS" == true ]]; then
  [[ "$FPS_FRAMES" -gt 0 ]] || die "--fps-frames is required with --measure-forward-fps"
  [[ "$FPS_FRAMES" -le "$FPS_PRELOAD_LIMIT" ]] || die "--fps-frames exceeds --fps-preload-limit"
  MAX_EVAL_FRAMES="$FPS_FRAMES"
  RESUME_EXISTING=false
  ABOT_RECON_RUN_LOOP=false
fi
[[ -n "$OUT_DIR" ]] || die "--out-dir is required"
[[ -n "$CKPT" && -e "$CKPT" ]] || die "--ckpt must exist: $CKPT"

case "$DATASET" in
  kitti) DATA_KEY=kitti-long ;;
  vbr) DATA_KEY=vbr-long ;;
  oxford) DATA_KEY=oxford_spires_processed-long ;;
esac

case "$METHOD" in
  horizon) MODEL_KEY=horizonstream ;;
  lingbot) MODEL_KEY=lingbot_map ;;
  abot_recon) MODEL_KEY=abot_recon ;;
  *) MODEL_KEY="$METHOD" ;;
esac

if [[ "$METHOD" == cut3r || "$METHOD" == ttt3r ]]; then
  if [[ "$CUT3R_TTT3R_RESET_INTERVAL" -eq 0 ]]; then
    RESET_TAG=noreset
  else
    RESET_TAG="reset${CUT3R_TTT3R_RESET_INTERVAL}"
  fi
  RUN_NAME="${METHOD}_${RESET_TAG}_${DATASET}_s${STRIDE}_sim3"
else
  RUN_NAME="${METHOD}_${DATASET}_s${STRIDE}_sim3"
fi
RUN_DIR="$OUT_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR/hydra"

CMD=(
  "$PY" relpose/eval_dist.py
  evaluation=relpose_stride1
  "eval_models=[$MODEL_KEY]"
  "eval_datasets=[$DATA_KEY]"
  "data.$DATA_KEY.pose_eval_stride=$STRIDE"
  "resume_existing=$RESUME_EXISTING"
  "max_eval_frames=$MAX_EVAL_FRAMES"
  "measure_forward_fps=$MEASURE_FORWARD_FPS"
  "+formal_pose_protocol=true"
  "forward_timing_preload_limit=$FPS_PRELOAD_LIMIT"
  "abot_recon_loop_enabled=$ABOT_RECON_LOOP_ENABLED"
  "abot_recon_loop_salad_ckpt=$HORIZON_SALAD_CKPT"
  "abot_recon_loop_dino_weights=$HORIZON_SALAD_DINO_WEIGHTS"
  "forward_timing_warmup_runs=$FPS_WARMUP_RUNS"
  "+forward_timing_warmup_frames=$FPS_WARMUP_FRAMES"
  "save_suffix=${RUN_NAME}"
  "name=${RUN_NAME}"
  "output_dir=$RUN_DIR"
  "hydra.run.dir=$RUN_DIR/hydra"
)

if [[ -n "$SEQUENCE" ]]; then
  CMD+=("data.$DATA_KEY.ls_all_seqs=[$SEQUENCE]")
fi

case "$METHOD" in
  cut3r|ttt3r)
    CMD+=(
      "model.$MODEL_KEY.cfg.pretrained_model_name_or_path=$CKPT"
      "model.$MODEL_KEY.cfg.source_root=$OFFICIAL_ROOT/${METHOD^^}"
      "model.$MODEL_KEY.cfg.input_size=512"
      "+cut3r_ttt3r_pose_reset_interval=$CUT3R_TTT3R_RESET_INTERVAL"
    )
    ;;
  longstream)
    CMD+=("model.longstream.cfg.checkpoint=$CKPT" "model.longstream.cfg.source_root=$OFFICIAL_ROOT/LongStream")
    ;;
  infinitevggt)
    CMD+=("model.infinitevggt.cfg.checkpoint=$CKPT" "model.infinitevggt.cfg.source_root=$OFFICIAL_ROOT/InfiniteVGGT")
    ;;
  ovggt)
    CMD+=("model.ovggt.cfg.checkpoint=$CKPT" "model.ovggt.cfg.source_root=$OFFICIAL_ROOT/OVGGT")
    ;;
  stream3r_window5)
    CMD+=("model.stream3r_window5.cfg.checkpoint=$CKPT" "model.stream3r_window5.cfg.source_root=$OFFICIAL_ROOT/STream3R")
    ;;
  horizon)
    CMD+=(
      "model.horizonstream.cfg.checkpoint=$CKPT"
      "model.horizonstream.cfg.horizonstream_root=$HS_ROOT"
      "model.horizonstream.cfg.amp_dtype=$HORIZON_AMP_DTYPE"
    )
    ;;
  lingbot)
    CMD+=(
      "model.lingbot_map.cfg.pretrained_model_name_or_path=$CKPT"
      "model.lingbot_map.cfg.source_root=$LINGBOT_ROOT"
      "model.lingbot_map.cfg.preprocess_mode=long_pose_official"
      "model.lingbot_map.cfg.max_frame_num=22000"
    )
    ;;
  abot_recon)
    CMD+=(
      "model.abot_recon.cfg.ckpt=$CKPT"
      "model.abot_recon.cfg.attention_backend=$ATTENTION_BACKEND"
      "model.abot_recon.eval_output_name=$RUN_NAME"
    )
    ;;
esac

if [[ "$METHOD" == abot_recon ]]; then
  export ABOT_RECON_ROPE2D_BACKEND="$ROPE2D_BACKEND"
fi
printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
printf '%q ' "${CMD[@]}"
printf '\n'

LOOP_CMD=()
if [[ "$METHOD" == horizon && "$HORIZON_LOOP_MODE" == both ]]; then
  [[ -f "$HORIZON_SALAD_CKPT" ]] || die "SALAD checkpoint not found: $HORIZON_SALAD_CKPT"
  [[ -f "$HORIZON_SALAD_DINO_WEIGHTS" ]] || die "SALAD DINO weights not found: $HORIZON_SALAD_DINO_WEIGHTS"
  LOOP_CMD=(
    "$PY" relpose/eval_horizon_loop.py
    --dataset "$DATASET"
    --base-output-root "$RUN_DIR/horizonstream"
    --output-root "$RUN_DIR/horizonstream_loop_salad"
    --checkpoint "$CKPT"
    --horizon-root "$HS_ROOT"
    --device cuda
    --preset "$HORIZON_LOOP_PRESET"
    --salad-ckpt "$HORIZON_SALAD_CKPT"
    --salad-dino-weights "$HORIZON_SALAD_DINO_WEIGHTS"
    --resume-existing "$HORIZON_LOOP_RESUME_EXISTING"
    --max-eval-frames "$MAX_EVAL_FRAMES"
  )
  if [[ -n "$SEQUENCE" ]]; then
    LOOP_CMD+=(--sequence "$SEQUENCE")
  fi
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${LOOP_CMD[@]}"
  printf '\n'
fi
if [[ "$METHOD" == abot_recon && "$ABOT_RECON_RUN_LOOP" == true ]]; then
  [[ -f "$HORIZON_SALAD_CKPT" ]] || die "SALAD checkpoint not found: $HORIZON_SALAD_CKPT"
  [[ -f "$HORIZON_SALAD_DINO_WEIGHTS" ]] || die "SALAD DINO weights not found: $HORIZON_SALAD_DINO_WEIGHTS"
  LOOP_CMD=(
    "$PY" relpose/eval_abot_recon_loop.py
    --dataset "$DATASET"
    --base-output-root "$RUN_DIR/$RUN_NAME"
    --output-root "$RUN_DIR/${RUN_NAME}_loop"
    --checkpoint "$CKPT"
    --device cuda
    --attention-backend "$ATTENTION_BACKEND"
    --salad-ckpt "$HORIZON_SALAD_CKPT"
    --salad-dino-weights "$HORIZON_SALAD_DINO_WEIGHTS"
    --resume-existing "$RESUME_EXISTING"
    --max-eval-frames "$MAX_EVAL_FRAMES"
  )
  if [[ -n "$SEQUENCE" ]]; then
    LOOP_CMD+=(--sequence "$SEQUENCE")
  fi
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${LOOP_CMD[@]}"
  printf '\n'
fi
[[ "$DRY_RUN" == true ]] && exit 0

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
if [[ "$CONFIG_CHECK" == true ]]; then
  exec "${CMD[@]}" --cfg job --resolve
fi
if [[ ${#LOOP_CMD[@]} -gt 0 ]]; then
  "${CMD[@]}"
  exec "${LOOP_CMD[@]}"
fi
exec "${CMD[@]}"
