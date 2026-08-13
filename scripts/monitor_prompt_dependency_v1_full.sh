#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS_CSV="${GPU_IDS:-0,3,4,5,6,7}"
BASE_RUN_ID="${BASE_RUN_ID:-prompt_dependency_v1_baseline_5b_20260728_001}"
BRANCH_RUN_ID="${BRANCH_RUN_ID:-prompt_dependency_v1_s35_m4_5b_20260728_001}"
BRANCH_CONFIG="${BRANCH_CONFIG:-configs/prompt_dependency_late_branch_s35_m4_wan22_480p.yaml}"
PAIRS_FILE="${PAIRS_FILE:-configs/prompt_dependency_v1_roots_3seeds.csv}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
EXPECTED_BASE_DONE="${EXPECTED_BASE_DONE:-360}"
EXPECTED_BRANCH_DONE="${EXPECTED_BRANCH_DONE:-1800}"
POLL_SECONDS="${POLL_SECONDS:-60}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"

mkdir -p "runs/late_branching/$BRANCH_RUN_ID/_logs"
mkdir -p "runs/logs"

LOG="runs/logs/prompt_dependency_v1_full_monitor.log"
STATUS_FILE="runs/logs/prompt_dependency_v1_full_status.txt"
exec >> "$LOG" 2>&1

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

live_sessions() {
  local pattern="$1"
  tmux ls 2>/dev/null | grep -E "$pattern" || true
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
    echo "monitor_log=$LOG"
  } > "$STATUS_FILE"
}

echo "[$(timestamp)] monitor start"
echo "baseline=$BASE_RUN_ID expected=$EXPECTED_BASE_DONE"
echo "branching=$BRANCH_RUN_ID expected=$EXPECTED_BRANCH_DONE"
write_status

while true; do
  base_done="$(done_count "runs/baseline/$BASE_RUN_ID")"
  base_live="$(live_sessions '^pdv1_base_g')"
  echo "[$(timestamp)] baseline progress: $base_done/$EXPECTED_BASE_DONE"
  write_status
  if (( base_done >= EXPECTED_BASE_DONE )); then
    break
  fi
  if [[ -z "$base_live" ]]; then
    echo "[$(timestamp)] ERROR: no baseline tmux sessions live but baseline is incomplete" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(timestamp)] baseline complete; launching branching"
for shard_index in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$shard_index]}"
  session="pdv1_branch_g${gpu}"
  log="runs/late_branching/$BRANCH_RUN_ID/_logs/shard_${shard_index}_gpu${gpu}.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[$(timestamp)] branching session exists: $session"
    continue
  fi
  cmd="cd '$REPO_ROOT' && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=$gpu python -u -m ttsd.runners.generate.late_branching --config '$BRANCH_CONFIG' --run-id '$BRANCH_RUN_ID' --pairs-file '$PAIRS_FILE' --shard-index $shard_index --num-shards $NUM_SHARDS > '$log' 2>&1"
  tmux new-session -d -s "$session" "bash -lc \"$cmd\""
  echo "[$(timestamp)] started $session gpu=$gpu shard=$shard_index/$NUM_SHARDS log=$log"
done

while true; do
  branch_done="$(done_count "runs/late_branching/$BRANCH_RUN_ID")"
  branch_live="$(live_sessions '^pdv1_branch_g')"
  echo "[$(timestamp)] branching progress: $branch_done/$EXPECTED_BRANCH_DONE"
  write_status
  if (( branch_done >= EXPECTED_BRANCH_DONE )); then
    break
  fi
  if [[ -z "$branch_live" ]]; then
    echo "[$(timestamp)] ERROR: no branching tmux sessions live but branching is incomplete" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

write_status
echo "[$(timestamp)] full generation complete"
