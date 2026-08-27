#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/all_models_pose}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
STATUS="$OUT_ROOT/status"
LOGS="$OUT_ROOT/logs"
METRICS="$OUT_ROOT/metrics"
mkdir -p "$STATUS" "$LOGS" "$METRICS"

checkpoint_for() {
  case "$1" in
    cut3r) echo "${CUT3R_CKPT:-$ROOT/checkpoints/cut3r/cut3r_512_dpt_4_64.pth}" ;;
    ttt3r) echo "${TTT3R_CKPT:-$ROOT/checkpoints/cut3r/cut3r_512_dpt_4_64.pth}" ;;
    longstream) echo "${LONGSTREAM_CKPT:-$ROOT/checkpoints/longstream/50_longstream.pt}" ;;
    infinitevggt) echo "${INFINITEVGGT_CKPT:-$ROOT/checkpoints/infinitevggt/checkpoints.pth}" ;;
    ovggt) echo "${OVGGT_CKPT:-$ROOT/checkpoints/infinitevggt/checkpoints.pth}" ;;
    stream3r_window5) echo "${STREAM3R_CKPT:-$ROOT/checkpoints/stream3r/model.safetensors}" ;;
    horizon) echo "${HORIZON_CKPT:-$ROOT/checkpoints/horizonstream/HorizonStream.pt}" ;;
    lingbot) echo "${LINGBOT_CKPT:-$ROOT/checkpoints/lingbot-map/lingbot-map.pt}" ;;
    abot_recon) echo "${ABOT_RECON_CKPT:-$ROOT/checkpoints/abot_recon.safetensors}" ;;
    *) return 2 ;;
  esac
}

run_metric() {
  local gpu="$1" method="$2" dataset="$3" task="${2}__${3}"
  local checkpoint log rc
  checkpoint="$(checkpoint_for "$method")" || return 2
  log="$LOGS/${task}.log"
  if [[ -f "$STATUS/${task}.done" ]]; then
    echo "[$(date '+%F %T')] skip completed $task" | tee -a "$log"
    return 0
  fi
  rm -f "$STATUS/${task}.failed"
  echo "[$(date '+%F %T')] START gpu=$gpu $task" | tee -a "$log"
  HORIZON_AMP_DTYPE=fp16 "$ROOT/scripts/run_long_pose_protocol.sh" \
    --method "$method" --dataset "$dataset" --stride 1 --gpu "$gpu" \
    --out-dir "$METRICS" --ckpt "$checkpoint" --resume-existing true \
    >>"$log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    touch "$STATUS/${task}.done"
    echo "[$(date '+%F %T')] DONE $task" | tee -a "$log"
  else
    printf '%s\n' "$rc" > "$STATUS/${task}.failed"
    echo "[$(date '+%F %T')] FAILED rc=$rc $task" | tee -a "$log"
  fi
  return "$rc"
}

run_method() {
  local gpu="$1" method="$2" dataset failures=0
  for dataset in kitti oxford vbr; do
    run_metric "$gpu" "$method" "$dataset" || failures=1
  done
  return "$failures"
}

queue_a() {
  local method failures=0
  for method in lingbot longstream ovggt horizon cut3r; do
    run_method "$GPU_A" "$method" || failures=1
  done
  return "$failures"
}

queue_b() {
  local method failures=0
  for method in abot_recon infinitevggt stream3r_window5 ttt3r; do
    run_method "$GPU_B" "$method" || failures=1
  done
  return "$failures"
}

cat > "$OUT_ROOT/protocol.txt" <<EOF
Formal protocol: stride=1, complete sequences, per-sequence Sim3 ATE,
delta=1-frame RPE-r (degrees) and RPE-t, sequence-macro average.
CUT3R/TTT3R/LongStream: FP32. InfiniteVGGT/OVGGT: BF16 autocast.
STream3R-window5: official mixed FP32/BF16. Horizon: FP16 autocast (official README minimal inference).
EOF

echo $$ > "$OUT_ROOT/launcher.pid"
queue_a > "$LOGS/queue_gpu${GPU_A}.log" 2>&1 & pid_a=$!
queue_b > "$LOGS/queue_gpu${GPU_B}.log" 2>&1 & pid_b=$!
printf '%s\n' "$pid_a" > "$OUT_ROOT/queue_gpu${GPU_A}.pid"
printf '%s\n' "$pid_b" > "$OUT_ROOT/queue_gpu${GPU_B}.pid"
wait "$pid_a"; rc_a=$?
wait "$pid_b"; rc_b=$?
"$ROOT/scripts/collect_pose_benchmark.py" "$OUT_ROOT" > "$LOGS/final_collect.log" 2>&1
rc_collect=$?
printf 'queue_a=%s queue_b=%s collect=%s\n' "$rc_a" "$rc_b" "$rc_collect" > "$OUT_ROOT/launcher.finished"
if [[ "$rc_a" -ne 0 || "$rc_b" -ne 0 || "$rc_collect" -ne 0 ]]; then
  exit 1
fi
