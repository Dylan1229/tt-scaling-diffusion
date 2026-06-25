#!/usr/bin/env bash
# Launch the chunk-branch I2V pilot across four GPUs, one fixed root per GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"

RUN_ID="${RUN_ID:-chunk_branch_i2v_p01_p02_s0_1}"
CONFIG="${CONFIG:-configs/chunk_branch_i2v_p01_p02_s0_1.yaml}"
VENV="${VENV:-$REPO_ROOT/.venv}"
LOG_DIR="$REPO_ROOT/runs/logs"
mkdir -p "$LOG_DIR"

echo "[launch] run_id  = $RUN_ID"
echo "[launch] config  = $CONFIG"
echo "[launch] gpus    = $GPUS  ($NUM_SHARDS shards)"
echo "[launch] output  = runs/baseline_long/$RUN_ID"
echo

for i in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$i]}"
  SESSION="chunk_branch_${RUN_ID}_gpu${GPU}"
  LOG="$LOG_DIR/chunk_branch_${RUN_ID}_gpu${GPU}.log"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[launch] tmux session '$SESSION' already exists - skipping"
    continue
  fi

  CMD="cd '$REPO_ROOT' && source '$VENV/bin/activate' && \
CUDA_VISIBLE_DEVICES=$GPU python -u -m ttsd.runners.generate.chunk_branch_i2v \
  --config '$CONFIG' \
  --run-id '$RUN_ID' \
  --shard-index $i --num-shards $NUM_SHARDS \
  2>&1 | tee -a '$LOG'"

  echo "[launch] starting $SESSION : GPU $GPU shard $i/$NUM_SHARDS -> $LOG"
  tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""
done

echo
echo "[launch] monitor with:"
echo "    tmux ls"
echo "    tail -f $LOG_DIR/chunk_branch_${RUN_ID}_gpu*.log"
echo "    watch -n 30 'find runs/baseline_long/$RUN_ID -name DONE | wc -l'"
