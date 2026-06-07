#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_dino_inputs_batch.sh --run-id <run_id>
      # → source runs/baseline/<run_id>, output runs/dino_input_frames/<run_id>
  # explicit roots override the --run-id defaults:
  scripts/prepare_dino_inputs_batch.sh \
    --source-run-root runs/baseline/<run_id> \
    --output-run-root runs/dino_input_frames/<run_id>

Optional:
  --gpu-indices 4,5,6
  --tile-sample-min-height 64
  --tile-sample-min-width 64
  --tile-sample-stride-height 48
  --tile-sample-stride-width 48
  --limit N
  --skip-existing

Walks <source-run-root>/<p>/<seed>/ for seed dirs that contain posterior_means/
and decodes each to mp4. One worker per --gpu-indices entry, round-robin work
assignment. Skip-existing checks the last expected output (step_049.mp4).
EOF
}

SOURCE_RUN_ROOT=""
OUTPUT_RUN_ROOT=""
RUN_ID=""
GPU_INDICES="4,5,6"
TILE_SAMPLE_MIN_HEIGHT=64
TILE_SAMPLE_MIN_WIDTH=64
TILE_SAMPLE_STRIDE_HEIGHT=48
TILE_SAMPLE_STRIDE_WIDTH=48
LIMIT=""
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-run-root)
      SOURCE_RUN_ROOT="$2"
      shift 2
      ;;
    --output-run-root)
      OUTPUT_RUN_ROOT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --gpu-indices)
      GPU_INDICES="$2"
      shift 2
      ;;
    --tile-sample-min-height)
      TILE_SAMPLE_MIN_HEIGHT="$2"
      shift 2
      ;;
    --tile-sample-min-width)
      TILE_SAMPLE_MIN_WIDTH="$2"
      shift 2
      ;;
    --tile-sample-stride-height)
      TILE_SAMPLE_STRIDE_HEIGHT="$2"
      shift 2
      ;;
    --tile-sample-stride-width)
      TILE_SAMPLE_STRIDE_WIDTH="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$RUN_ID" ]]; then
  : "${SOURCE_RUN_ROOT:=runs/baseline/$RUN_ID}"
  : "${OUTPUT_RUN_ROOT:=runs/dino_input_frames/$RUN_ID}"
fi

if [[ -z "$SOURCE_RUN_ROOT" || -z "$OUTPUT_RUN_ROOT" ]]; then
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing repo-local .venv at $REPO_ROOT/.venv" >&2
  exit 1
fi

IFS=',' read -r -a gpu_array <<< "$GPU_INDICES"
if [[ "${#gpu_array[@]}" -lt 1 ]]; then
  echo "--gpu-indices must contain at least 1 physical GPU index" >&2
  exit 1
fi

for idx in "${!gpu_array[@]}"; do
  gpu_array[$idx]="$(echo "${gpu_array[$idx]}" | xargs)"
done

mapfile -t seed_dirs < <(find "$SOURCE_RUN_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seed*' | sort)

if [[ -n "$LIMIT" ]]; then
  seed_dirs=("${seed_dirs[@]:0:$LIMIT}")
fi

echo "[posterior_decode_batch] repo_root=$REPO_ROOT"
echo "[posterior_decode_batch] source_run_root=$SOURCE_RUN_ROOT"
echo "[posterior_decode_batch] output_run_root=$OUTPUT_RUN_ROOT"
echo "[posterior_decode_batch] gpu_indices=${gpu_array[*]}"
echo "[posterior_decode_batch] tile=${TILE_SAMPLE_MIN_HEIGHT}/${TILE_SAMPLE_MIN_WIDTH}/${TILE_SAMPLE_STRIDE_HEIGHT}/${TILE_SAMPLE_STRIDE_WIDTH}"
echo "[posterior_decode_batch] count=${#seed_dirs[@]}"

mkdir -p "$OUTPUT_RUN_ROOT"
FAILURES_FILE="$OUTPUT_RUN_ROOT/_decode_failures.txt"
: > "$FAILURES_FILE"

run_worker() {
  local worker_id="$1"
  local gpu_index="$2"
  local i
  local seed_dir
  local rel_path
  local output_dir
  local latents_dir

  for ((i=worker_id; i<${#seed_dirs[@]}; i+=${#gpu_array[@]})); do
    seed_dir="${seed_dirs[$i]}"
    rel_path="${seed_dir#${SOURCE_RUN_ROOT}/}"
    output_dir="$OUTPUT_RUN_ROOT/$rel_path"
    latents_dir="$seed_dir/posterior_means"

    if [[ ! -d "$latents_dir" ]]; then
      echo "[posterior_decode_batch gpu${gpu_index}] skip $rel_path (no posterior_means/)" | tee -a "$FAILURES_FILE"
      continue
    fi

    if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_dir/step_049.mp4" ]]; then
      echo "[posterior_decode_batch gpu${gpu_index}] skip $rel_path"
      continue
    fi

    mkdir -p "$output_dir"
    echo "[posterior_decode_batch gpu${gpu_index}] decode $rel_path"
    if ! env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -m ttsd.runners.generate.decode_latents \
        --latents-dir "$latents_dir" \
        --output-dir "$output_dir" \
        --device cuda \
        --gpu-indices "$gpu_index" \
        --dtype bf16 \
        --tile-sample-min-height "$TILE_SAMPLE_MIN_HEIGHT" \
        --tile-sample-min-width "$TILE_SAMPLE_MIN_WIDTH" \
        --tile-sample-stride-height "$TILE_SAMPLE_STRIDE_HEIGHT" \
        --tile-sample-stride-width "$TILE_SAMPLE_STRIDE_WIDTH"; then
      echo "[posterior_decode_batch gpu${gpu_index}] failed $rel_path" | tee -a "$FAILURES_FILE"
    fi
  done
}

worker_pids=()
for worker_id in "${!gpu_array[@]}"; do
  run_worker "$worker_id" "${gpu_array[$worker_id]}" &
  worker_pids+=("$!")
done

for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

echo "[posterior_decode_batch] done"
