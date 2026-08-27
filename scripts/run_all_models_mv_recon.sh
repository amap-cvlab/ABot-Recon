#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/all_models_mv_recon}"
GPUS_CSV="${GPUS:-0,1,2}"
ALIGN="${ALIGN:-sim3}"
SAVE_PLY="${SAVE_PLY:-false}"
SAVE_TRAJ="${SAVE_TRAJ:-false}"

IFS=',' read -r -a GPU_ARR <<<"$GPUS_CSV"
if [[ ${#GPU_ARR[@]} -lt 1 || ${#GPU_ARR[@]} -gt 3 ]]; then
  echo "ERROR: GPUS must contain between one and three GPU indices" >&2
  exit 2
fi

METHODS=(lingbot horizon cut3r ttt3r stream3r_window5 longstream infinitevggt ovggt abot_recon)
declare -A CKPTS=(
  [lingbot]="${LINGBOT_CKPT:-$ROOT/checkpoints/lingbot-map/lingbot-map.pt}"
  [horizon]="${HORIZON_CKPT:-$ROOT/checkpoints/horizonstream/HorizonStream.pt}"
  [cut3r]="${CUT3R_CKPT:-$ROOT/checkpoints/cut3r/cut3r_512_dpt_4_64.pth}"
  [ttt3r]="${TTT3R_CKPT:-$ROOT/checkpoints/cut3r/cut3r_512_dpt_4_64.pth}"
  [stream3r_window5]="${STREAM3R_CKPT:-$ROOT/checkpoints/stream3r/model.safetensors}"
  [longstream]="${LONGSTREAM_CKPT:-$ROOT/checkpoints/longstream/50_longstream.pt}"
  [infinitevggt]="${INFINITEVGGT_CKPT:-$ROOT/checkpoints/infinitevggt/checkpoints.pth}"
  [ovggt]="${OVGGT_CKPT:-$ROOT/checkpoints/infinitevggt/checkpoints.pth}"
  [abot_recon]="${ABOT_RECON_CKPT:-$ROOT/checkpoints/abot_recon.safetensors}"
)

mkdir -p "$OUT_ROOT/logs"
STATUS_FILE="$OUT_ROOT/status.tsv"
printf 'timestamp\tgpu\tmethod\tstatus\n' >"$STATUS_FILE"

run_one() {
  local gpu="$1"
  local method="$2"
  local out_dir="$OUT_ROOT/$method"
  local log="$OUT_ROOT/logs/$method.log"
  mkdir -p "$out_dir"

  printf '%s\t%s\t%s\tSTART\n' "$(date '+%F %T')" "$gpu" "$method" >>"$STATUS_FILE"
  echo "[$(date '+%F %T')] gpu=$gpu START $method" | tee -a "$log"

  if bash "$ROOT/scripts/run_mv_recon_stride1_suite.sh" \
      --method "$method" \
      --ckpt "${CKPTS[$method]}" \
      --gpu "$gpu" \
      --align "$ALIGN" \
      --out-root "$out_dir" \
      --stream3r-chunk-size 1 \
      --save-ply "$SAVE_PLY" \
      --save-traj "$SAVE_TRAJ" >>"$log" 2>&1; then
    printf '%s\t%s\t%s\tDONE\n' "$(date '+%F %T')" "$gpu" "$method" >>"$STATUS_FILE"
    echo "[$(date '+%F %T')] gpu=$gpu DONE $method" | tee -a "$log"
    return 0
  else
    local rc=$?
    printf '%s\t%s\t%s\tFAILED_%s\n' "$(date '+%F %T')" "$gpu" "$method" "$rc" >>"$STATUS_FILE"
    echo "[$(date '+%F %T')] gpu=$gpu FAILED rc=$rc $method" | tee -a "$log"
    return "$rc"
  fi
}

worker() {
  local rank="$1"
  local gpu="${GPU_ARR[$rank]}"
  local i
  local worker_rc=0
  for ((i=rank; i<${#METHODS[@]}; i+=${#GPU_ARR[@]})); do
    run_one "$gpu" "${METHODS[$i]}" || worker_rc=1
  done
  return "$worker_rc"
}

pids=()
for rank in "${!GPU_ARR[@]}"; do
  worker "$rank" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done

echo "[$(date '+%F %T')] all workers finished; status=$STATUS_FILE"
exit "$rc"
