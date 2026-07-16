#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_microstep_vbench_variants.sh --run-root runs/microstep_grid/<run_id> [options]

Options:
  --output-root DIR     Default: runs/vbench_microstep_grid/<run_id>
  --dimensions CSV      Optional comma-separated VBench dimensions.
  --device DEVICE       Default: cuda
  --skip-staged         Reuse existing VBench staging dirs.
  --variants CSV        Optional comma-separated variant names.

Environment:
  PYTHON                Python executable. Default: .venv/bin/python
USAGE
}

RUN_ROOT=""
OUTPUT_ROOT=""
DIMENSIONS=""
DEVICE="cuda"
SKIP_STAGED=0
VARIANTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dimensions)
      DIMENSIONS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --skip-staged)
      SKIP_STAGED=1
      shift
      ;;
    --variants)
      VARIANTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  echo "--run-root is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Run root not found: $RUN_ROOT" >&2
  exit 1
fi

RUN_ID="$(basename "$RUN_ROOT")"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="runs/vbench_microstep_grid/$RUN_ID"
fi

PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUTPUT_ROOT"

declare -A WANT=()
if [[ -n "$VARIANTS" ]]; then
  IFS=',' read -r -a parts <<< "$VARIANTS"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    [[ -n "$part" ]] && WANT["$part"]=1
  done
fi

for variant_dir in "$RUN_ROOT"/*; do
  [[ -d "$variant_dir" ]] || continue
  variant="$(basename "$variant_dir")"
  [[ "$variant" == _* ]] && continue
  if [[ ${#WANT[@]} -gt 0 && -z "${WANT[$variant]:-}" ]]; then
    continue
  fi
  if ! find "$variant_dir" -mindepth 3 -maxdepth 3 -name meta.json -print -quit | grep -q .; then
    continue
  fi

  args=(
    -m ttsd.eval.vbench
    --run "$variant_dir"
    --output "$OUTPUT_ROOT/$variant"
    --device "$DEVICE"
  )
  if [[ -n "$DIMENSIONS" ]]; then
    args+=(--dimensions "$DIMENSIONS")
  fi
  if [[ "$SKIP_STAGED" -eq 1 ]]; then
    args+=(--skip-staged)
  fi

  echo "[microstep_vbench] variant=$variant output=$OUTPUT_ROOT/$variant"
  "$PYTHON" "${args[@]}"
done
