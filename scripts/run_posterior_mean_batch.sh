#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_posterior_mean_batch.sh \
    --source-run-root /path/to/source/run_root \
    --replay-run-root /path/to/replay/output/root \
    --decode-run-root /path/to/decode/output/root

Optional:
  --replay-cuda-visible-devices 1
  --gpu-indices 0,1,2,3
  --tile-sample-min-height 64
  --tile-sample-min-width 64
  --tile-sample-stride-height 48
  --tile-sample-stride-width 48
  --limit N
  --skip-existing

This script iterates over all <prompt>/<seed> dirs in the source run, replays
the sample while saving posterior means, then decodes those posterior means.
EOF
}

SOURCE_RUN_ROOT=""
REPLAY_RUN_ROOT=""
DECODE_RUN_ROOT=""
REPLAY_CUDA_VISIBLE_DEVICES=""
GPU_INDICES="0,1,2,3"
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
    --replay-run-root)
      REPLAY_RUN_ROOT="$2"
      shift 2
      ;;
    --decode-run-root)
      DECODE_RUN_ROOT="$2"
      shift 2
      ;;
    --replay-cuda-visible-devices)
      REPLAY_CUDA_VISIBLE_DEVICES="$2"
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

if [[ -z "$SOURCE_RUN_ROOT" || -z "$REPLAY_RUN_ROOT" || -z "$DECODE_RUN_ROOT" ]]; then
  usage >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing repo-local .venv at $REPO_ROOT/.venv" >&2
  exit 1
fi

mapfile -t seed_dirs < <(find "$SOURCE_RUN_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'seed*' | sort)

if [[ -n "$LIMIT" ]]; then
  seed_dirs=("${seed_dirs[@]:0:$LIMIT}")
fi

echo "[posterior_batch] repo_root=$REPO_ROOT"
echo "[posterior_batch] source_run_root=$SOURCE_RUN_ROOT"
echo "[posterior_batch] replay_run_root=$REPLAY_RUN_ROOT"
echo "[posterior_batch] decode_run_root=$DECODE_RUN_ROOT"
echo "[posterior_batch] replay_cuda_visible_devices=${REPLAY_CUDA_VISIBLE_DEVICES:-<default>}"
echo "[posterior_batch] count=${#seed_dirs[@]}"

FAILURES_FILE="$REPLAY_RUN_ROOT/_batch_failures.txt"
mkdir -p "$REPLAY_RUN_ROOT"
: > "$FAILURES_FILE"

for source_seed_dir in "${seed_dirs[@]}"; do
  rel_path="${source_seed_dir#${SOURCE_RUN_ROOT}/}"
  replay_dir="$REPLAY_RUN_ROOT/$rel_path"
  decode_dir="$DECODE_RUN_ROOT/$rel_path"

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$replay_dir/replay_summary.json" && -f "$decode_dir/step_049.mp4" ]]; then
    echo "[posterior_batch] skip $rel_path"
    continue
  fi

  echo "[posterior_batch] replay $rel_path"
  replay_cmd=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
  if [[ -n "$REPLAY_CUDA_VISIBLE_DEVICES" ]]; then
    replay_cmd+=(CUDA_VISIBLE_DEVICES="$REPLAY_CUDA_VISIBLE_DEVICES")
  fi
  replay_cmd+=(.venv/bin/python -m ttsd.runners.replay_with_posterior_means
    --source-seed-dir "$source_seed_dir"
    --output-dir "$replay_dir")
  if ! "${replay_cmd[@]}"; then
    echo "[posterior_batch] replay_failed $rel_path" | tee -a "$FAILURES_FILE"
    continue
  fi

  echo "[posterior_batch] decode $rel_path"
  if ! env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m ttsd.runners.decode_latents \
      --latents-dir "$replay_dir/posterior_means" \
      --output-dir "$decode_dir" \
      --device cuda \
      --gpu-indices "$GPU_INDICES" \
      --dtype bf16 \
      --tile-sample-min-height "$TILE_SAMPLE_MIN_HEIGHT" \
      --tile-sample-min-width "$TILE_SAMPLE_MIN_WIDTH" \
      --tile-sample-stride-height "$TILE_SAMPLE_STRIDE_HEIGHT" \
      --tile-sample-stride-width "$TILE_SAMPLE_STRIDE_WIDTH"; then
    echo "[posterior_batch] decode_failed $rel_path" | tee -a "$FAILURES_FILE"
    continue
  fi
done

echo "[posterior_batch] done"