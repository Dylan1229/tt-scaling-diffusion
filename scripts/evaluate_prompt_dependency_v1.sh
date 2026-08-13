#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_RUN_ID="${BASE_RUN_ID:-prompt_dependency_v1_baseline_5b_20260728_001}"
BRANCH_RUN_ID="${BRANCH_RUN_ID:-prompt_dependency_v1_s35_m4_5b_20260728_001}"
GPU_IDS_CSV="${GPU_IDS:-0,2,3,4,5,6,7}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

BASE_RUN="runs/baseline/$BASE_RUN_ID"
BRANCH_RUN="runs/late_branching/$BRANCH_RUN_ID"
BASE_OUT="runs/vbench/$BASE_RUN_ID"
BRANCH_OUT="runs/vbench/$BRANCH_RUN_ID"
PARTS_ROOT="runs/vbench/_prompt_dependency_v1_parts"
LOG_ROOT="runs/logs/prompt_dependency_v1_vbench"

dims=(
  subject_consistency
  background_consistency
  motion_smoothness
  dynamic_degree
  aesthetic_quality
  imaging_quality
  overall_consistency
)

IFS=',' read -r -a gpu_ids <<< "$GPU_IDS_CSV"
if (( ${#gpu_ids[@]} < ${#dims[@]} )); then
  echo "Need at least ${#dims[@]} GPUs, got ${#gpu_ids[@]}" >&2
  exit 1
fi

mkdir -p "$PARTS_ROOT" "$LOG_ROOT" "$BASE_OUT/raw" "$BRANCH_OUT/raw"

pids=()
for i in "${!dims[@]}"; do
  dim="${dims[$i]}"
  gpu="${gpu_ids[$i]}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ttsd.eval.vbench \
      --run "$BRANCH_RUN" \
      --output "$PARTS_ROOT/branch/$dim" \
      --dimensions "$dim" \
      >"$LOG_ROOT/branch_${dim}.log" 2>&1

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ttsd.eval.vbench \
      --run "$BASE_RUN" \
      --output "$PARTS_ROOT/baseline/$dim" \
      --dimensions "$dim" \
      >"$LOG_ROOT/baseline_${dim}.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one VBench worker failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

for dim in "${dims[@]}"; do
  cp "$PARTS_ROOT/branch/$dim/raw/"*.json "$BRANCH_OUT/raw/"
  cp "$PARTS_ROOT/baseline/$dim/raw/"*.json "$BASE_OUT/raw/"
done

"$PYTHON" -u -m ttsd.eval.vbench \
  --run "$BRANCH_RUN" \
  --output "$BRANCH_OUT" \
  --aggregate-only \
  >"$LOG_ROOT/branch_aggregate.log" 2>&1

"$PYTHON" -u -m ttsd.eval.vbench \
  --run "$BASE_RUN" \
  --output "$BASE_OUT" \
  --aggregate-only \
  >"$LOG_ROOT/baseline_aggregate.log" 2>&1

"$PYTHON" -u -m ttsd.eval.late_branch_oracle \
  --run "$BRANCH_RUN" \
  --targets "$BRANCH_OUT/vbench_targets.csv" \
  --baseline-targets "$BASE_OUT/vbench_targets.csv" \
  --output "runs/analysis/${BRANCH_RUN_ID}_oracle" \
  >"$LOG_ROOT/oracle.log" 2>&1

echo "VBench evaluation complete"
