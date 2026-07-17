#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/late_branching_s35_wan22_480p.yaml}"
RUN_ID="${RUN_ID:-late_branch_s35_m4_$(date -u +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "runs/late_branching/$RUN_ID/_logs"
echo "[launch] run_id=$RUN_ID config=$CONFIG GPUs=$GPU_IDS_CSV"

pids=()
for shard_index in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$shard_index]}"
  log="runs/late_branching/$RUN_ID/_logs/shard_${shard_index}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ttsd.runners.generate.late_branching \
    --config "$CONFIG" \
    --run-id "$RUN_ID" \
    --shard-index "$shard_index" \
    --num-shards "$NUM_SHARDS" \
    >"$log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "[launch] shard=$shard_index gpu=$gpu pid=$pid log=$log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "[launch] at least one shard failed; inspect runs/late_branching/$RUN_ID/_logs" >&2
  exit "$status"
fi

echo "[launch] complete: runs/late_branching/$RUN_ID"
