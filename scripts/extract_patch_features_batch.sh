#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/extract_patch_features_batch.sh --run-id <run_id>
      # → decoded runs/dino_input_frames/<run_id>, output runs/patch_features/<run_id>
  # explicit roots override the --run-id defaults:
  scripts/extract_patch_features_batch.sh \
    --decoded-run-root runs/dino_input_frames/<run_id> \
    --output-run-root runs/patch_features/<run_id>

Optional:
  --gpu-indices 4,5,6,7
  --batch-size 16
  --frame-stride 4
  --limit N
  --skip-existing

Extracts DINOv2 patch tokens for each seed, one worker per selected GPU.
EOF
}

DECODED_RUN_ROOT=""
OUTPUT_RUN_ROOT=""
RUN_ID=""
GPU_INDICES="4,5,6,7"
BATCH_SIZE=16
FRAME_STRIDE=4
LIMIT=""
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --decoded-run-root) DECODED_RUN_ROOT="$2"; shift 2 ;;
    --output-run-root)  OUTPUT_RUN_ROOT="$2"; shift 2 ;;
    --run-id)           RUN_ID="$2"; shift 2 ;;
    --gpu-indices)      GPU_INDICES="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    --frame-stride)     FRAME_STRIDE="$2"; shift 2 ;;
    --limit)            LIMIT="$2"; shift 2 ;;
    --skip-existing)    SKIP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -n "$RUN_ID" ]]; then
  : "${DECODED_RUN_ROOT:=runs/dino_input_frames/$RUN_ID}"
  : "${OUTPUT_RUN_ROOT:=runs/patch_features/$RUN_ID}"
fi

if [[ -z "$DECODED_RUN_ROOT" || -z "$OUTPUT_RUN_ROOT" ]]; then
  usage >&2; exit 1
fi

cd "$REPO_ROOT"
[[ -x .venv/bin/python ]] || { echo "Missing .venv at $REPO_ROOT" >&2; exit 1; }

IFS=',' read -r -a gpu_array <<< "$GPU_INDICES"
for idx in "${!gpu_array[@]}"; do
  gpu_array[$idx]="$(echo "${gpu_array[$idx]}" | xargs)"
done

mapfile -t seed_dirs < <(find "$DECODED_RUN_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seed*' | sort)
if [[ -n "$LIMIT" ]]; then
  seed_dirs=("${seed_dirs[@]:0:$LIMIT}")
fi

echo "[patch_batch] decoded=$DECODED_RUN_ROOT  output=$OUTPUT_RUN_ROOT  gpus=${gpu_array[*]}  count=${#seed_dirs[@]}  stride=$FRAME_STRIDE"
mkdir -p "$OUTPUT_RUN_ROOT"
FAILURES_FILE="$OUTPUT_RUN_ROOT/_patch_failures.txt"
: > "$FAILURES_FILE"

run_worker() {
  local worker_id="$1"
  local gpu_index="$2"
  local i seed_dir rel_path output_dir

  for ((i=worker_id; i<${#seed_dirs[@]}; i+=${#gpu_array[@]})); do
    seed_dir="${seed_dirs[$i]}"
    rel_path="${seed_dir#${DECODED_RUN_ROOT}/}"
    output_dir="$OUTPUT_RUN_ROOT/$rel_path"

    if [[ "$SKIP_EXISTING" -eq 1 \
      && -f "$output_dir/posterior_mean_patch_features.npy" \
      && -f "$output_dir/posterior_mean_patch_features_metadata.json" ]]; then
      echo "[patch_batch gpu${gpu_index}] skip $rel_path"
      continue
    fi

    mkdir -p "$output_dir"
    echo "[patch_batch gpu${gpu_index}] extract $rel_path"
    if ! env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -m ttsd.runners.features.patch_features \
        --dino-input-frames-dir "$seed_dir" \
        --output-dir "$output_dir" \
        --gpu-indices "$gpu_index" \
        --batch-size "$BATCH_SIZE" \
        --frame-stride "$FRAME_STRIDE"; then
      echo "[patch_batch gpu${gpu_index}] failed $rel_path" | tee -a "$FAILURES_FILE"
    fi
  done
}

worker_pids=()
for worker_id in "${!gpu_array[@]}"; do
  run_worker "$worker_id" "${gpu_array[$worker_id]}" &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid"; done
echo "[patch_batch] done"
