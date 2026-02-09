#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Defaults
# -----------------------------
EVAL_PY="eval.py"
GPUS=""
OUTPUT_DIR="results"
SLEEP_SECS=2

RUNS=()
TAG=""

usage() {
  cat <<EOF
Usage:
  ./run_eval.sh --gpus 0,1,2 [--eval_py eval.py] [--output_dir results] [--tag mytag] \\
      --run '<args for eval.py>' --run '<args for eval.py>' ...

Notes:
  - Each --run is launched on exactly one GPU from --gpus.
  - Runs are scheduled as GPUs free up (max concurrency = #GPUs).
  - You may omit --pc_ckpt/--pc_temperature/--pc_frac in any run.
  - Logs go to: <output_dir>/logs/

Examples:
  ./run_eval.sh --gpus 0,1 \\
    --run '--model_alias llada --task math500 --alg low_confidence --tokens_per_step 2 --num_steps 256 --pc_ckpt "/nfs-shared-2/anji/test1221.jpc" --pc_temperature 0.2 --pc_frac 0.7' \\
    --run '--model_alias llada --task gsm8k --alg low_confidence --tokens_per_step 2 --num_steps 256'

EOF
}

die() { echo "Error: $*" >&2; exit 1; }

# -----------------------------
# Parse args (simple long-opts)
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="${2:-}"; shift 2;;
    --eval_py)
      EVAL_PY="${2:-}"; shift 2;;
    --output_dir)
      OUTPUT_DIR="${2:-}"; shift 2;;
    --tag)
      TAG="${2:-}"; shift 2;;
    --sleep_secs)
      SLEEP_SECS="${2:-}"; shift 2;;
    --run)
      RUNS+=("${2:-}"); shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      die "Unknown argument: $1 (use --help)";;
  esac
done

[[ -n "$GPUS" ]] || die "--gpus is required (e.g. --gpus 0,1,2)"
[[ -f "$EVAL_PY" ]] || die "Cannot find --eval_py file: $EVAL_PY"
[[ ${#RUNS[@]} -gt 0 ]] || die "Provide at least one --run '<args>'"

# -----------------------------
# Prepare GPU slots
# -----------------------------
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[[ ${#GPU_ARR[@]} -gt 0 ]] || die "No GPUs parsed from --gpus"

mkdir -p "${OUTPUT_DIR}/logs"

# One PID per GPU slot; empty means free slot
declare -a SLOT_PIDS
declare -a SLOT_RUNIDS
for ((i=0; i<${#GPU_ARR[@]}; i++)); do
  SLOT_PIDS[i]=""
  SLOT_RUNIDS[i]=""
done

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_run_on_slot() {
  local slot_idx="$1"
  local run_idx="$2"
  local gpu="${GPU_ARR[$slot_idx]}"
  local args="${RUNS[$run_idx]}"

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local safe_tag="$TAG"
  if [[ -n "$safe_tag" ]]; then safe_tag="_${safe_tag}"; fi

  local log_file="${OUTPUT_DIR}/logs/run${run_idx}_gpu${gpu}${safe_tag}_${ts}.log"

  echo "==> [slot ${slot_idx}] GPU ${gpu} : run ${run_idx}"
  echo "    python ${EVAL_PY} ${args}"
  echo "    log: ${log_file}"

  # Launch
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python "${EVAL_PY}" ${args}
  ) > "${log_file}" 2>&1 &

  local pid=$!
  SLOT_PIDS[$slot_idx]="$pid"
  SLOT_RUNIDS[$slot_idx]="$run_idx"
}

# -----------------------------
# Scheduler: max concurrency = #GPUs
# -----------------------------
next_run=0
total_runs=${#RUNS[@]}

while [[ $next_run -lt $total_runs ]]; do
  assigned=false

  # Find a free slot
  for ((s=0; s<${#GPU_ARR[@]}; s++)); do
    pid="${SLOT_PIDS[$s]}"
    if ! is_running "$pid"; then
      # Slot is free (or completed); clear it and assign next run
      SLOT_PIDS[$s]=""
      SLOT_RUNIDS[$s]=""
      start_run_on_slot "$s" "$next_run"
      next_run=$((next_run + 1))
      assigned=true
      break
    fi
  done

  if [[ "$assigned" == false ]]; then
    sleep "${SLEEP_SECS}"
  fi
done

# Wait for all remaining jobs
echo "==> All runs launched. Waiting for completion..."
for ((s=0; s<${#GPU_ARR[@]}; s++)); do
  pid="${SLOT_PIDS[$s]}"
  if is_running "$pid"; then
    runid="${SLOT_RUNIDS[$s]}"
    echo "    waiting: slot ${s} (GPU ${GPU_ARR[$s]}) pid=${pid} run=${runid}"
    wait "$pid"
  fi
done

echo "==> Done. Logs are in: ${OUTPUT_DIR}/logs/"
