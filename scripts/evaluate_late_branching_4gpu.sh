#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:?Set RUN_ID to a completed late-branching run}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
RUN_ROOT="runs/late_branching/$RUN_ID"
OUT_ROOT="runs/vbench/$RUN_ID"
PARTS_ROOT="$OUT_ROOT/_parts"
LOG_ROOT="$RUN_ROOT/_logs/vbench_parts"
IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"

dims=(
  subject_consistency
  background_consistency
  motion_smoothness
  dynamic_degree
  aesthetic_quality
  imaging_quality
  overall_consistency
)

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Run directory not found: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$PARTS_ROOT" "$LOG_ROOT"
echo "[vbench] run_id=$RUN_ID GPUs=$GPU_IDS_CSV"

pids=()
for worker_index in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$worker_index]}"
  (
    for dim_index in "${!dims[@]}"; do
      if (( dim_index % ${#GPU_IDS_ARRAY[@]} != worker_index )); then
        continue
      fi
      dim="${dims[$dim_index]}"
      echo "[vbench] gpu=$gpu dim=$dim"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m ttsd.eval.vbench \
        --run "$RUN_ROOT" \
        --output "$PARTS_ROOT/$dim" \
        --dimensions "$dim" \
        >"$LOG_ROOT/$dim.log" 2>&1
    done
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
  echo "[vbench] at least one worker failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

mkdir -p "$OUT_ROOT/raw"
for dim in "${dims[@]}"; do
  cp "$PARTS_ROOT/$dim/raw/"*.json "$OUT_ROOT/raw/"
done

"$PYTHON" -u -m ttsd.eval.vbench \
  --run "$RUN_ROOT" \
  --output "$OUT_ROOT" \
  --aggregate-only \
  | tee "$LOG_ROOT/aggregate.log"

"$PYTHON" -u -m ttsd.eval.late_branch_oracle \
  --run "$RUN_ROOT" \
  --targets "$OUT_ROOT/vbench_targets.csv" \
  --baseline-targets runs/baseline/20260511_224405/vbench/vbench_targets.csv \
  | tee "$LOG_ROOT/oracle.log"

echo "[vbench] complete: $OUT_ROOT"
