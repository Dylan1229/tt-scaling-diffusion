#!/usr/bin/env bash
# Launch 30s long-video baseline generation across 4 GPUs (default: 4,5,6,7)
# as detached tmux sessions.
#
# Each shard:
#   - pins one GPU via CUDA_VISIBLE_DEVICES
#   - shares the same --run-id so outputs land in one runs/baseline_long/<run_id>/
#   - skips clips that already have a DONE marker, so re-launching is safe
#
# Usage:
#   ./scripts/generate_long_vbench_4gpu.sh
#   GPUS=2,3,4,5 ./scripts/generate_long_vbench_4gpu.sh
#   RUN_ID=long_vbench_30s ./scripts/generate_long_vbench_4gpu.sh
#   CONFIG=configs/long_wan22_480p_direct.yaml LIMIT_PROMPTS=1 LIMIT_SEEDS=5 ./scripts/generate_long_vbench_4gpu.sh
#
# After launch:
#   tmux ls
#   tmux attach -t long_vbench_gpu4
#   tail -f runs/logs/long_vbench_<run_id>_gpu*.log
#   find runs/baseline_long/<run_id> -name DONE | wc -l   # progress, max 150

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"

RUN_ID="${RUN_ID:-long_vbench_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/long_wan22_480p.yaml}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-}"
LIMIT_SEEDS="${LIMIT_SEEDS:-}"
SMOKE="${SMOKE:-}"
VENV="${VENV:-$REPO_ROOT/.venv}"
LOG_DIR="$REPO_ROOT/runs/logs"
mkdir -p "$LOG_DIR"

EXTRA_ARGS=()
if [ -n "$LIMIT_PROMPTS" ]; then
  EXTRA_ARGS+=(--limit-prompts "$LIMIT_PROMPTS")
fi
if [ -n "$LIMIT_SEEDS" ]; then
  EXTRA_ARGS+=(--limit-seeds "$LIMIT_SEEDS")
fi
if [ -n "$SMOKE" ]; then
  EXTRA_ARGS+=(--smoke)
fi
EXTRA_ARGS_STR=""
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  printf -v EXTRA_ARGS_STR ' %q' "${EXTRA_ARGS[@]}"
fi

echo "[launch] run_id  = $RUN_ID"
echo "[launch] config  = $CONFIG"
echo "[launch] gpus    = $GPUS  ($NUM_SHARDS shards)"
if [ -n "$EXTRA_ARGS_STR" ]; then
  echo "[launch] extra   =$EXTRA_ARGS_STR"
fi
echo "[launch] venv    = $VENV"
echo "[launch] log dir = $LOG_DIR"
echo "[launch] output  = runs/baseline_long/$RUN_ID"
echo

for i in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$i]}"
  SHARD="$i"
  SESSION="long_vbench_${RUN_ID}_gpu${GPU}"
  LOG="$LOG_DIR/long_vbench_${RUN_ID}_gpu${GPU}.log"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[launch] tmux session '$SESSION' already exists - skipping (kill it first if you want to re-launch)"
    continue
  fi

  CMD="cd '$REPO_ROOT' && source '$VENV/bin/activate' && \
CUDA_VISIBLE_DEVICES=$GPU python -u -m ttsd.runners.generate.long_video \
  --config '$CONFIG' \
  --run-id '$RUN_ID' \
  --shard-index $SHARD --num-shards $NUM_SHARDS$EXTRA_ARGS_STR \
  2>&1 | tee -a '$LOG'"

  echo "[launch] starting $SESSION : GPU $GPU shard $SHARD/$NUM_SHARDS -> $LOG"
  tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""
done

echo
echo "[launch] shards requested. Monitor with:"
echo "    tmux ls"
echo "    tail -f $LOG_DIR/long_vbench_${RUN_ID}_gpu*.log"
echo "    watch -n 30 'find runs/baseline_long/$RUN_ID -name DONE | wc -l'"
