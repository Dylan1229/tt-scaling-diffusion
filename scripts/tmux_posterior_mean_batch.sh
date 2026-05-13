#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="posterior-mean-batch"

usage() {
  cat <<'EOF'
Usage:
  scripts/tmux_posterior_mean_batch.sh \
    --source-run-root /path/to/source/run_root \
    --replay-run-root /path/to/replay/output/root \
    --decode-run-root /path/to/decode/output/root

Optional:
  --session-name posterior-mean-batch
  --replay-cuda-visible-devices 1
  --gpu-indices 0,1,2,3
  --tile-sample-min-height 64
  --tile-sample-min-width 64
  --tile-sample-stride-height 48
  --tile-sample-stride-width 48
  --limit N
  --skip-existing
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

LOG_DIR="$REPO_ROOT/runs/tmux_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"

CMD="cd '$REPO_ROOT' && bash '$REPO_ROOT/scripts/run_posterior_mean_batch.sh' ${ARGS[*]} |& tee '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo "started tmux session: $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "log: $LOG_FILE"