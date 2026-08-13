#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_RUN_ID="${BASE_RUN_ID:-prompt_dependency_v1_baseline_5b_20260728_001}"
BRANCH_RUN_ID="${BRANCH_RUN_ID:-prompt_dependency_v1_s35_m4_5b_20260728_001}"
OUTPUT="runs/analysis/${BRANCH_RUN_ID}_prompt_dependency"
LOG="runs/logs/prompt_dependency_v1_analysis.log"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

while [[ ! -f "runs/vbench/$BASE_RUN_ID/vbench_targets.csv" ]] || \
      [[ ! -f "runs/vbench/$BRANCH_RUN_ID/vbench_targets.csv" ]]; do
  if ! tmux has-session -t pdv1_vbench 2>/dev/null; then
    echo "VBench stopped before targets were produced" >&2
    exit 1
  fi
  sleep 120
done

.venv/bin/python -u -m ttsd.runners.report.prompt_dependency_report \
  --manifest configs/prompt_dependency_v1_roots_3seeds.csv \
  --baseline-targets "runs/vbench/$BASE_RUN_ID/vbench_targets.csv" \
  --branch-targets "runs/vbench/$BRANCH_RUN_ID/vbench_targets.csv" \
  --branch-run "runs/late_branching/$BRANCH_RUN_ID" \
  --output "$OUTPUT"

echo "Prompt-dependency report complete: $OUTPUT"
