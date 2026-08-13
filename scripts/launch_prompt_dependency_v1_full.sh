#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS_CSV="${GPU_IDS:-0,3,4,5,6,7}"
BASE_RUN_ID="${BASE_RUN_ID:-prompt_dependency_v1_baseline_5b_20260728_001}"
BRANCH_RUN_ID="${BRANCH_RUN_ID:-prompt_dependency_v1_s35_m4_5b_20260728_001}"
BASE_CONFIG="${BASE_CONFIG:-configs/prompt_dependency_baseline_wan22_480p.yaml}"
BRANCH_CONFIG="${BRANCH_CONFIG:-configs/prompt_dependency_late_branch_s35_m4_wan22_480p.yaml}"
PAIRS_FILE="${PAIRS_FILE:-configs/prompt_dependency_v1_roots_3seeds.csv}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

mkdir -p "runs/baseline/$BASE_RUN_ID/_logs"
mkdir -p "runs/late_branching/$BRANCH_RUN_ID/_logs"
mkdir -p "runs/logs"

STATUS_FILE="runs/logs/prompt_dependency_v1_full_status.txt"
PIPELINE_LOG="runs/logs/prompt_dependency_v1_full_pipeline.log"
exec >> "$PIPELINE_LOG" 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

done_count() {
  local path="$1"
  if [[ -d "$path" ]]; then
    find "$path" -name DONE | wc -l
  else
    echo 0
  fi
}

write_status() {
  {
    echo "timestamp=$(timestamp)"
    echo "base_run_id=$BASE_RUN_ID"
    echo "branch_run_id=$BRANCH_RUN_ID"
    echo "gpus=$GPU_IDS_CSV"
    echo "num_shards=$NUM_SHARDS"
    echo "baseline_done=$(done_count "runs/baseline/$BASE_RUN_ID")"
    echo "branch_done=$(done_count "runs/late_branching/$BRANCH_RUN_ID")"
    echo "pipeline_log=$PIPELINE_LOG"
  } > "$STATUS_FILE"
}

launch_baseline() {
  echo "[$(timestamp)] baseline start: $BASE_RUN_ID"
  local pids=()
  for shard_index in "${!GPU_IDS_ARRAY[@]}"; do
    local gpu="${GPU_IDS_ARRAY[$shard_index]}"
    local log="runs/baseline/$BASE_RUN_ID/_logs/shard_${shard_index}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ttsd.runners.generate.baseline \
      --config "$BASE_CONFIG" \
      --run-id "$BASE_RUN_ID" \
      --shard-index "$shard_index" \
      --num-shards "$NUM_SHARDS" \
      > "$log" 2>&1 &
    pids+=("$!")
    echo "[$(timestamp)] baseline shard=$shard_index gpu=$gpu pid=$! log=$log"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  write_status
  if (( status != 0 )); then
    echo "[$(timestamp)] baseline failed; inspect runs/baseline/$BASE_RUN_ID/_logs" >&2
    exit "$status"
  fi
  echo "[$(timestamp)] baseline complete: $(done_count "runs/baseline/$BASE_RUN_ID") DONE"
}

launch_branching() {
  echo "[$(timestamp)] branching start: $BRANCH_RUN_ID"
  local pids=()
  for shard_index in "${!GPU_IDS_ARRAY[@]}"; do
    local gpu="${GPU_IDS_ARRAY[$shard_index]}"
    local log="runs/late_branching/$BRANCH_RUN_ID/_logs/shard_${shard_index}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ttsd.runners.generate.late_branching \
      --config "$BRANCH_CONFIG" \
      --run-id "$BRANCH_RUN_ID" \
      --pairs-file "$PAIRS_FILE" \
      --shard-index "$shard_index" \
      --num-shards "$NUM_SHARDS" \
      > "$log" 2>&1 &
    pids+=("$!")
    echo "[$(timestamp)] branching shard=$shard_index gpu=$gpu pid=$! log=$log"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  write_status
  if (( status != 0 )); then
    echo "[$(timestamp)] branching failed; inspect runs/late_branching/$BRANCH_RUN_ID/_logs" >&2
    exit "$status"
  fi
  echo "[$(timestamp)] branching complete: $(done_count "runs/late_branching/$BRANCH_RUN_ID") DONE"
}

echo "[$(timestamp)] prompt_dependency_v1 full pipeline launch"
echo "repo=$REPO_ROOT"
echo "gpus=$GPU_IDS_CSV"
echo "baseline=$BASE_RUN_ID config=$BASE_CONFIG"
echo "branching=$BRANCH_RUN_ID config=$BRANCH_CONFIG pairs=$PAIRS_FILE"
write_status
launch_baseline
launch_branching
write_status
echo "[$(timestamp)] pipeline complete"
