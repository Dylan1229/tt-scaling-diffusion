#!/usr/bin/env bash
# Launch the sweep_v2 generation across 4 GPUs (default: 4,5,6,7) as 4
# detached tmux sessions. Safe to close your SSH session / laptop after launch;
# the work continues on the server until completion.
#
# Each shard:
#   - has its own GPU pinned via CUDA_VISIBLE_DEVICES
#   - shares the same --run-id so outputs land in ONE runs/baseline/<run_id>/
#   - skips clips that already have a DONE marker (so re-launching is safe)
#
# Usage:
#   ./scripts/generate_sweep_v2_4gpu.sh                 # defaults: GPUs 4,5,6,7
#   GPUS=2,3,4,5 ./scripts/generate_sweep_v2_4gpu.sh    # override GPU set
#   RUN_ID=test_v2 ./scripts/generate_sweep_v2_4gpu.sh  # override run-id
#
# After launch:
#   tmux ls                       # confirm 4 sessions live
#   tmux attach -t sweep_v2_gpu4  # detach with Ctrl-b d
#   tail -f runs/logs/sweep_v2_<run_id>_gpu*.log
#   find runs/baseline/<run_id> -name DONE | wc -l   # progress (max 450)
#   tmux kill-session -t sweep_v2_gpu4  # cancel one shard

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"

RUN_ID="${RUN_ID:-sweep_v2_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/sweep_v2_wan22_480p.yaml}"
VENV="${VENV:-$REPO_ROOT/.venv}"
LOG_DIR="$REPO_ROOT/runs/logs"
mkdir -p "$LOG_DIR"

echo "[launch] run_id  = $RUN_ID"
echo "[launch] config  = $CONFIG"
echo "[launch] gpus    = $GPUS  ($NUM_SHARDS shards)"
echo "[launch] venv    = $VENV"
echo "[launch] log dir = $LOG_DIR"
echo

for i in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$i]}"
  SHARD="$i"
  SESSION="sweep_v2_gpu${GPU}"
  LOG="$LOG_DIR/sweep_v2_${RUN_ID}_gpu${GPU}.log"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[launch] tmux session '$SESSION' already exists — skipping (kill it first if you want to re-launch)"
    continue
  fi

  CMD="cd '$REPO_ROOT' && source '$VENV/bin/activate' && \
CUDA_VISIBLE_DEVICES=$GPU python -u -m ttsd.runners.generate.baseline \
  --config '$CONFIG' \
  --capture-posterior-means \
  --run-id '$RUN_ID' \
  --shard-index $SHARD --num-shards $NUM_SHARDS \
  2>&1 | tee -a '$LOG'"

  echo "[launch] starting $SESSION : GPU $GPU shard $SHARD/$NUM_SHARDS  → $LOG"
  tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""
done

echo
echo "[launch] all shards started. Monitor with:"
echo "    tmux ls"
echo "    tail -f $LOG_DIR/sweep_v2_${RUN_ID}_gpu*.log"
echo "    watch -n 30 'find runs/baseline/$RUN_ID -name DONE | wc -l'"
