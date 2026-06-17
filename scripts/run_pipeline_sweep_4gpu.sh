#!/usr/bin/env bash
# Multi-GPU pipeline sweep launcher.
#
# Launches one detached tmux session per GPU. Each shard runs
# `ttsd.runners.pipeline.sweep` over (prompt × seed × strategy) and is
# resumable — re-launch the same SWEEP_ID to skip completed items.
#
# Defaults (P5 user request): GPUs 0,1,2,3 + the EFD&I-vs-BoN 5×5 sweep.
# Override via env vars:
#   GPUS="2,3"             — only use GPUs 2 and 3
#   CONFIG=path/to.yaml    — different sweep config
#   SWEEP_ID=my_run        — resume / share-id across shards
#   VENV=/path/to/venv
#   LIMIT_PROMPTS=2 LIMIT_SEEDS=2   — quick test
#
# Monitor:
#   tmux ls
#   tail -f runs/pipeline_sweeps/_logs/${SWEEP_ID}_gpu*.log
#   ls -d runs/pipeline_sweeps/${SWEEP_ID}/*/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/pipeline/sweeps/efdi_vs_bon_5x5.yaml}"
GPUS="${GPUS:-0,1,2,3}"
SWEEP_ID="${SWEEP_ID:-pipeline_sweep_$(date +%Y%m%d_%H%M%S)}"
VENV="${VENV:-/data/datasets/fanjiang/venv_envs/tt-scaling-diffusion}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-}"
LIMIT_SEEDS="${LIMIT_SEEDS:-}"
LIMIT_STRATEGIES="${LIMIT_STRATEGIES:-}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_SHARDS=${#GPU_ARR[@]}

LOG_DIR="runs/pipeline_sweeps/_logs"
mkdir -p "$LOG_DIR"

EXTRA_FLAGS=""
[ -n "$LIMIT_PROMPTS"    ] && EXTRA_FLAGS+=" --limit-prompts $LIMIT_PROMPTS"
[ -n "$LIMIT_SEEDS"      ] && EXTRA_FLAGS+=" --limit-seeds $LIMIT_SEEDS"
[ -n "$LIMIT_STRATEGIES" ] && EXTRA_FLAGS+=" --limit-strategies $LIMIT_STRATEGIES"

echo "[launcher] config    = $CONFIG"
echo "[launcher] sweep_id  = $SWEEP_ID"
echo "[launcher] GPUs      = $GPUS (n_shards=$N_SHARDS)"
echo "[launcher] log_dir   = $LOG_DIR"
[ -n "$EXTRA_FLAGS" ] && echo "[launcher] extra     =$EXTRA_FLAGS"

for i in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$i]}"
    SESSION="pipeline_sweep_gpu${GPU}"
    LOG_PATH="$LOG_DIR/${SWEEP_ID}_gpu${GPU}.log"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[launcher] session '$SESSION' already exists; leaving it alone"
        continue
    fi

    CMD="cd '$REPO_ROOT' && \
source '$VENV/bin/activate' && \
CUDA_VISIBLE_DEVICES=$GPU python -u -m ttsd.runners.pipeline.sweep \
    --config '$CONFIG' \
    --sweep-id '$SWEEP_ID' \
    --shard-index $i \
    --num-shards $N_SHARDS \
    $EXTRA_FLAGS \
    2>&1 | tee '$LOG_PATH'"

    tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""
    echo "[launcher] started $SESSION (shard $i/$N_SHARDS, GPU $GPU)  → $LOG_PATH"
done

cat <<EOF

Sweep launched.

Monitor:
  tmux ls
  tail -f $LOG_DIR/${SWEEP_ID}_gpu*.log

Output dir:
  runs/pipeline_sweeps/${SWEEP_ID}/

When the sweep finishes, generate the Pareto plot:
  python -m ttsd.runners.pipeline.pareto_plot \\
      --runs runs/pipeline_sweeps/${SWEEP_ID}/*/* \\
      --output-dir runs/pipeline_sweeps/${SWEEP_ID}/_pareto

EOF
