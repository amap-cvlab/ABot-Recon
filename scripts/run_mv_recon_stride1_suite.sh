#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-python}"
RUNNER="${RUNNER:-$ROOT/scripts/run_mv_recon_protocol.sh}"
METHOD="${METHOD:-}"
CKPT="${CKPT:-}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-}"
ALIGN="${ALIGN:-sim3}"
ROT_CORRECTION_MODE="${ROT_CORRECTION_MODE:-causal_conv}"
ROT_CORRECTION_KERNEL="${ROT_CORRECTION_KERNEL:-10}"
LINGBOT_MAX_FRAMES="${LINGBOT_MAX_FRAMES:-8000}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-$ROOT/third_party}"
HS_ROOT="${HS_ROOT:-$ROOT/third_party/HorizonStream}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
STREAM3R_CHUNK_SIZE="${STREAM3R_CHUNK_SIZE:-1}"
SAVE_PLY="${SAVE_PLY:-false}"
SAVE_TRAJ="${SAVE_TRAJ:-false}"
OXFORD_METRIC_INTERVAL="${OXFORD_METRIC_INTERVAL:-10}"
DRY_RUN=false
CONFIG_CHECK=false

usage() {
  cat <<'EOF'
Run the standard dense stride-1 reconstruction suite sequentially on one GPU.

Suite:
  1. 7Scenes: seven <scene>/seq-01 sequences, stride 1
  2. TUM-Dynamics-Full: all associated RGB frames, stride 1
  3. Oxford Spires: ten rectified sequences, stride-1 forward and interval-10
     TLS point-cloud metrics

Required:
  --method abot_recon|lingbot|horizon|cut3r|ttt3r|longstream|infinitevggt|ovggt|stream3r_window5
  --ckpt PATH
  --out-root PATH

Optional:
  --gpu INDEX
  --align sim3|se3                  Default: sim3 for fair cross-model comparison
  --rot-correction-mode causal_conv|none
  --rot-correction-kernel N
  --lingbot-max-frames N
  --hs-root PATH
  --official-root PATH
  --attention-backend auto|paged|sdpa  ABot-Recon; auto prefers paged FlashInfer
  --stream3r-chunk-size N          Default: 1; stream3r_window5 is reported as
                                   native world-points, not depth+K
  --oxford-metric-interval N       Default: 10; formal Oxford TLS metric
                                   interval (stride-1 model inference)
  --save-ply true|false             Default: false
  --save-traj true|false            Default: false (formal point-cloud runs)
  --dry-run
  --config-check

Each dataset writes to <out-root>/{7scenes_seq01_s1,tum_full_s1,oxford_s1_i10}/.
The suite stops at the first failed dataset. Detailed logs are stored inside
each dataset output directory as run.log.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --align) ALIGN="$2"; shift 2 ;;
    --rot-correction-mode) ROT_CORRECTION_MODE="$2"; shift 2 ;;
    --rot-correction-kernel) ROT_CORRECTION_KERNEL="$2"; shift 2 ;;
    --lingbot-max-frames) LINGBOT_MAX_FRAMES="$2"; shift 2 ;;
    --hs-root) HS_ROOT="$2"; shift 2 ;;
    --official-root) OFFICIAL_ROOT="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --stream3r-chunk-size) STREAM3R_CHUNK_SIZE="$2"; shift 2 ;;
    --oxford-metric-interval) OXFORD_METRIC_INTERVAL="$2"; shift 2 ;;
    --save-ply) SAVE_PLY="$2"; shift 2 ;;
    --save-traj) SAVE_TRAJ="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --config-check) CONFIG_CHECK=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 2; }

cd "$ROOT"

[[ -n "$METHOD" ]] || die "--method is required"
[[ -n "$CKPT" ]] || die "--ckpt is required"
[[ -n "$OUT_ROOT" ]] || die "--out-root is required"
[[ "$METHOD" =~ ^(abot_recon|lingbot|horizon|cut3r|ttt3r|longstream|infinitevggt|ovggt|stream3r_window5)$ ]] || die "invalid method: $METHOD"
[[ "$ALIGN" =~ ^(sim3|se3)$ ]] || die "--align must be sim3 or se3"
[[ "$ATTENTION_BACKEND" =~ ^(auto|paged|sdpa)$ ]] || die "invalid --attention-backend"
[[ "$OXFORD_METRIC_INTERVAL" == 10 ]] || die "formal Oxford protocol requires --oxford-metric-interval 10"
[[ -x "$RUNNER" ]] || die "protocol launcher is not executable: $RUNNER"
command -v "$PY" >/dev/null || die "python is not executable: $PY"
export PY

mkdir -p "$OUT_ROOT"

if [[ "$DRY_RUN" == false && "$CONFIG_CHECK" == false ]]; then
  if [[ -z "${FORMAL_CHECKPOINT_SHA256:-}" ]]; then
    echo "[$(date '+%F %T')] Computing checkpoint SHA256 once for the suite"
    FORMAL_CHECKPOINT_SHA256=$("$PY" -c \
      'import sys; from pathlib import Path; from scripts.formal_run_metadata import checkpoint_sha256; print(checkpoint_sha256(Path(sys.argv[1])))' \
      "$CKPT")
    export FORMAL_CHECKPOINT_SHA256
  fi
  printf '%s\n' "$FORMAL_CHECKPOINT_SHA256" > "$OUT_ROOT/checkpoint_sha256.txt"
fi

run_dataset() {
  local tag="$1"
  shift
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir/logs"

  local cmd=(
    "$RUNNER"
    --method "$METHOD"
    --ckpt "$CKPT"
    --gpu "$GPU"
    --align "$ALIGN"
    --out-dir "$out_dir"
    --save-ply "$SAVE_PLY"
    --save-traj "$SAVE_TRAJ"
    --official-root "$OFFICIAL_ROOT"
    --attention-backend "$ATTENTION_BACKEND"
    "$@"
  )

  case "$METHOD" in
    abot_recon)
      cmd+=(
        --rot-correction-mode "$ROT_CORRECTION_MODE"
        --rot-correction-kernel "$ROT_CORRECTION_KERNEL"
      )
      ;;
    abot_recon)
      ;;
    lingbot)
      cmd+=(--lingbot-max-frames "$LINGBOT_MAX_FRAMES")
      ;;
    horizon)
      cmd+=(--hs-root "$HS_ROOT")
      ;;
    stream3r_window5)
      cmd+=(--stream3r-chunk-size "$STREAM3R_CHUNK_SIZE")
      ;;
  esac

  if [[ "$DRY_RUN" == true ]]; then
    cmd+=(--dry-run)
  elif [[ "$CONFIG_CHECK" == true ]]; then
    cmd+=(--config-check)
  fi

  echo "[$(date '+%F %T')] START $tag"
  if [[ "$DRY_RUN" == true || "$CONFIG_CHECK" == true ]]; then
    "${cmd[@]}"
  else
    "${cmd[@]}" >"$out_dir/logs/run.log" 2>&1
  fi
  echo "[$(date '+%F %T')] DONE  $tag"
}

run_dataset 7scenes_seq01_s1 --dataset 7scenes --stride 1
run_dataset tum_full_s1 --dataset tum --tum-mode full
run_dataset oxford_s1_i10 --dataset oxford --stride 1 \
  --oxford-metric-interval "$OXFORD_METRIC_INTERVAL"

echo "[$(date '+%F %T')] Standard stride-1 suite complete: $OUT_ROOT"
