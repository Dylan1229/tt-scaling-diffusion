#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_posterior_mean_heatmap_batch.sh \
    --decoded-run-root /path/to/decoded/posterior_mean/root \
    --output-run-root /path/to/output/root

Optional:
  --gpu-indices 4,5
  --batch-size 32
  --limit N
  --skip-existing

This script processes all decoded posterior-mean seed directories using exactly
two workers, each pinned to one physical GPU.
EOF
}

DECODED_RUN_ROOT=""
OUTPUT_RUN_ROOT=""
GPU_INDICES="4,5"
BATCH_SIZE=32
LIMIT=""
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --decoded-run-root)
      DECODED_RUN_ROOT="$2"
      shift 2
      ;;
    --output-run-root)
      OUTPUT_RUN_ROOT="$2"
      shift 2
      ;;
    --gpu-indices)
      GPU_INDICES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
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

if [[ -z "$DECODED_RUN_ROOT" || -z "$OUTPUT_RUN_ROOT" ]]; then
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing repo-local .venv at $REPO_ROOT/.venv" >&2
  exit 1
fi

IFS=',' read -r -a gpu_array <<< "$GPU_INDICES"
if [[ "${#gpu_array[@]}" -ne 2 ]]; then
  echo "--gpu-indices must contain exactly 2 physical GPU indices" >&2
  exit 1
fi

for idx in "${!gpu_array[@]}"; do
  gpu_array[$idx]="$(echo "${gpu_array[$idx]}" | xargs)"
done

mapfile -t seed_dirs < <(find "$DECODED_RUN_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seed*' | sort)

if [[ -n "$LIMIT" ]]; then
  seed_dirs=("${seed_dirs[@]:0:$LIMIT}")
fi

echo "[posterior_heatmap_batch] repo_root=$REPO_ROOT"
echo "[posterior_heatmap_batch] decoded_run_root=$DECODED_RUN_ROOT"
echo "[posterior_heatmap_batch] output_run_root=$OUTPUT_RUN_ROOT"
echo "[posterior_heatmap_batch] gpu_indices=${gpu_array[*]}"
echo "[posterior_heatmap_batch] batch_size=$BATCH_SIZE"
echo "[posterior_heatmap_batch] count=${#seed_dirs[@]}"

mkdir -p "$OUTPUT_RUN_ROOT"
FAILURES_FILE="$OUTPUT_RUN_ROOT/_heatmap_failures.txt"
: > "$FAILURES_FILE"

run_worker() {
  local worker_id="$1"
  local gpu_index="$2"
  local i
  local seed_dir
  local rel_path
  local output_dir

  for ((i=worker_id; i<${#seed_dirs[@]}; i+=${#gpu_array[@]})); do
    seed_dir="${seed_dirs[$i]}"
    rel_path="${seed_dir#${DECODED_RUN_ROOT}/}"
    output_dir="$OUTPUT_RUN_ROOT/$rel_path"

    if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_dir/posterior_mean_diagonal_similarity.npy" && -f "$output_dir/posterior_mean_diagonal_similarity_heatmap.png" && -f "$output_dir/posterior_mean_diagonal_similarity_metadata.json" ]]; then
      echo "[posterior_heatmap_batch gpu${gpu_index}] skip $rel_path"
      continue
    fi

    mkdir -p "$output_dir"
    echo "[posterior_heatmap_batch gpu${gpu_index}] heatmap $rel_path"
    if ! env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -m ttsd.runners.posterior_mean_heatmap \
        --posterior-mean-video-dir "$seed_dir" \
        --output-dir "$output_dir" \
        --gpu-indices "$gpu_index" \
        --batch-size "$BATCH_SIZE"; then
      echo "[posterior_heatmap_batch gpu${gpu_index}] failed $rel_path" | tee -a "$FAILURES_FILE"
    fi
  done
}

run_worker 0 "${gpu_array[0]}" &
pid0=$!
run_worker 1 "${gpu_array[1]}" &
pid1=$!

wait "$pid0"
wait "$pid1"

echo "[posterior_heatmap_batch] done"