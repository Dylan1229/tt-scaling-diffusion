#!/usr/bin/env bash
# Launch VBench-Long scoring for a long-video run in a detached tmux session.
#
# Usage:
#   ./scripts/eval_vbench_long_gpu.sh runs/baseline_long/<run_id>
#   GPU=5 ./scripts/eval_vbench_long_gpu.sh runs/baseline_long/<run_id>
#   WAIT_FOR_DONE=150 ./scripts/eval_vbench_long_gpu.sh runs/baseline_long/<run_id>
#   GPU=5 SEEDS=3-5 SKIP_EXISTING=1 ./scripts/eval_vbench_long_gpu.sh runs/baseline_long/<run_id>
#   GPU=5 PROMPT_IDS=p01 SEEDS=0-4 ./scripts/eval_vbench_long_gpu.sh runs/baseline_long/<run_id>
#   OUTPUT=runs/vbench_long/<run_id>_pilot_p01_seed0-4 ./scripts/eval_vbench_long_gpu.sh runs/baseline_long/<run_id>
#
# Monitor:
#   tmux attach -t vbench_long_eval_gpu4
#   tail -f runs/logs/vbench_long_eval_<run_id>_gpu4.log
#   ls runs/vbench_long/<run_id>/

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 runs/baseline_long/<run_id>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="$1"
RUN_ID="$(basename "$RUN_DIR")"
GPU="${GPU:-4}"
VENV="${VENV:-$REPO_ROOT/.venv}"
LOG_DIR="$REPO_ROOT/runs/logs"
WAIT_FOR_DONE="${WAIT_FOR_DONE:-}"
DIMENSIONS="${DIMENSIONS:-}"
SEEDS="${SEEDS:-}"
PROMPT_IDS="${PROMPT_IDS:-}"
OUTPUT="${OUTPUT:-}"
SKIP_STAGED="${SKIP_STAGED:-}"
SKIP_EXISTING="${SKIP_EXISTING:-}"
MERGE_ONLY="${MERGE_ONLY:-}"
mkdir -p "$LOG_DIR"

SESSION="vbench_long_eval_${RUN_ID}_gpu${GPU}"
LOG="$LOG_DIR/vbench_long_eval_${RUN_ID}_gpu${GPU}.log"

if [ "${IN_TMUX:-}" = "1" ]; then
  if [ -n "$WAIT_FOR_DONE" ]; then
    while [ "$(find "$RUN_DIR" -name DONE 2>/dev/null | wc -l)" -lt "$WAIT_FOR_DONE" ]; do
      echo "[wait] $(date -u '+%Y-%m-%d %H:%M:%S UTC') DONE=$(find "$RUN_DIR" -name DONE 2>/dev/null | wc -l)/$WAIT_FOR_DONE"
      sleep 300
    done
  fi

  eval_cmd=(python -u -m ttsd.eval.vbench_long --run "$RUN_DIR" --device cuda)
  if [ -n "$OUTPUT" ]; then
    eval_cmd+=(--output "$OUTPUT")
  fi
  if [ -n "$DIMENSIONS" ]; then
    eval_cmd+=(--dimensions "$DIMENSIONS")
  fi
  if [ -n "$SEEDS" ]; then
    eval_cmd+=(--seeds "$SEEDS")
  fi
  if [ -n "$PROMPT_IDS" ]; then
    eval_cmd+=(--prompt-ids "$PROMPT_IDS")
  fi
  if [ -n "$SKIP_STAGED" ]; then
    eval_cmd+=(--skip-staged)
  fi
  if [ -n "$SKIP_EXISTING" ]; then
    eval_cmd+=(--skip-existing)
  fi
  if [ -n "$MERGE_ONLY" ]; then
    eval_cmd+=(--merge-only)
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "${eval_cmd[@]}"
  exit $?
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] tmux session '$SESSION' already exists - skipping"
  exit 0
fi

CMD="cd '$REPO_ROOT' && source '$VENV/bin/activate' && \
IN_TMUX=1 GPU='$GPU' WAIT_FOR_DONE='$WAIT_FOR_DONE' DIMENSIONS='$DIMENSIONS' \
SEEDS='$SEEDS' PROMPT_IDS='$PROMPT_IDS' OUTPUT='$OUTPUT' SKIP_STAGED='$SKIP_STAGED' SKIP_EXISTING='$SKIP_EXISTING' MERGE_ONLY='$MERGE_ONLY' \
'$REPO_ROOT/scripts/eval_vbench_long_gpu.sh' '$RUN_DIR' \
2>&1 | tee -a '$LOG'"

echo "[launch] session = $SESSION"
echo "[launch] run     = $RUN_DIR"
echo "[launch] gpu     = $GPU"
echo "[launch] log     = $LOG"
if [ -n "$WAIT_FOR_DONE" ]; then
  echo "[launch] wait    = $WAIT_FOR_DONE DONE markers"
fi
if [ -n "$SEEDS" ]; then
  echo "[launch] seeds   = $SEEDS"
fi
if [ -n "$PROMPT_IDS" ]; then
  echo "[launch] prompts = $PROMPT_IDS"
fi
if [ -n "$OUTPUT" ]; then
  echo "[launch] output  = $OUTPUT"
fi
if [ -n "$SKIP_STAGED" ]; then
  echo "[launch] stage   = reuse existing staging"
fi
if [ -n "$SKIP_EXISTING" ]; then
  echo "[launch] reuse   = existing raw JSON"
fi
if [ -n "$MERGE_ONLY" ]; then
  echo "[launch] mode    = merge only"
fi

tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""

echo "[launch] scoring session started. Monitor with:"
echo "    tmux attach -t $SESSION"
echo "    tail -f $LOG"
