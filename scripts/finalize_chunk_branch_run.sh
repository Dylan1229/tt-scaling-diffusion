#!/usr/bin/env bash
# Finalize a chunk-branch run after generation and queued VBench shards finish.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:?set RUN_ID}"
CONFIG="${CONFIG:?set CONFIG}"
EXPECTED_DONE="${EXPECTED_DONE:-960}"
EXPECTED_RAW="${EXPECTED_RAW:-384}"
SEEDS="${SEEDS:-0-31,1000-1031}"
INDEPENDENT_SCORES="${INDEPENDENT_SCORES:-runs/vbench_long/long_vbench_30s_20260618_190202}"
RUN_DIR="${RUN_DIR:-runs/baseline_long/$RUN_ID}"
SCORE_DIR="${SCORE_DIR:-runs/vbench_long/$RUN_ID}"
COMPARE_DIR="${COMPARE_DIR:-runs/vbench_long_compare/$RUN_ID}"

timestamp() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

count_done() {
  find "$RUN_DIR" -name DONE 2>/dev/null | wc -l
}

count_raw() {
  find "$SCORE_DIR/raw" -name '*_eval_results.json' 2>/dev/null | wc -l
}

echo "[$(timestamp)] finalize RUN_ID=$RUN_ID"
echo "[$(timestamp)] waiting for $EXPECTED_DONE DONE markers under $RUN_DIR"
while [ "$(count_done)" -lt "$EXPECTED_DONE" ]; do
  echo "[$(timestamp)] generation DONE=$(count_done)/$EXPECTED_DONE"
  sleep 600
done

echo "[$(timestamp)] generation complete; merging branch manifest"
".venv/bin/python" -m ttsd.runners.generate.chunk_branch_i2v \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --merge-manifest

manifest_lines="$(wc -l < "$RUN_DIR/branch_manifest.csv")"
echo "[$(timestamp)] branch manifest lines=$manifest_lines"

echo "[$(timestamp)] waiting for queued VBench sessions to finish"
while tmux ls 2>/dev/null | rg -q "vbench_long_eval_${RUN_ID}_gpu"; do
  echo "[$(timestamp)] raw eval JSON=$(count_raw)/$EXPECTED_RAW"
  sleep 600
done

raw_count="$(count_raw)"
echo "[$(timestamp)] VBench sessions done; raw eval JSON=$raw_count/$EXPECTED_RAW"
if [ "$raw_count" -ne "$EXPECTED_RAW" ]; then
  echo "[$(timestamp)] ERROR: expected $EXPECTED_RAW raw eval JSON files, got $raw_count" >&2
  exit 1
fi

echo "[$(timestamp)] merging VBench CSVs from raw outputs"
".venv/bin/python" -m ttsd.eval.vbench_long \
  --run "$RUN_DIR" \
  --output "$SCORE_DIR" \
  --seeds "$SEEDS" \
  --merge-only

echo "[$(timestamp)] summarizing branch selection against independent concat"
".venv/bin/python" -m ttsd.eval.summarize_chunk_branch \
  --branch-run "$RUN_DIR" \
  --branch-scores "$SCORE_DIR" \
  --independent-scores "$INDEPENDENT_SCORES" \
  --output-dir "$COMPARE_DIR"

echo "[$(timestamp)] done: $COMPARE_DIR"
