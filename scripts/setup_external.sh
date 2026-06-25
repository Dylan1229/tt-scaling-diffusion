#!/usr/bin/env bash
# Clone the vendored upstream repos that this project builds on.
# Run once after cloning tt-scaling-diffusion; idempotent (skips if present).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$REPO_ROOT/external"
mkdir -p "$EXT"

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -d "$dest/.git" ] || [ -f "$dest/setup.py" ] || [ -f "$dest/pyproject.toml" ]; then
    echo "[setup_external] SKIP $(basename "$dest") — already present at $dest"
  else
    echo "[setup_external] CLONE $url → $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

# 1) shim0114 / T2V-Diffusion-Search — vendored Wan T2V driver + VBench prompts.
clone_if_missing https://github.com/shim0114/T2V-Diffusion-Search.git "$EXT/t2v-search"

# 2) Vchitect / VBench — official benchmark.
clone_if_missing https://github.com/Vchitect/VBench.git "$EXT/VBench"

# 3) Patch VBench setup.py: remove the CUDA<=12.1 install-time guard, which
#    rejects cu128 (Blackwell) builds even though the actual metric code works.
SETUP="$EXT/VBench/setup.py"
if [ -f "$SETUP" ] && grep -q "^check_torch_version()" "$SETUP"; then
  echo "[setup_external] PATCH $SETUP — disable CUDA<=12.1 guard"
  sed -i 's/^check_torch_version()/# check_torch_version()  # patched: works on cu128/' "$SETUP"
fi

# VBench-Long imports moviepy.editor only for an unused VideoFileClip symbol.
# moviepy 1.x forces decorator<5, which conflicts with the repo's dev stack,
# and moviepy 2.x does not provide moviepy.editor. Patch out the unused import.
LONG_UTILS="$EXT/VBench/vbench2_beta_long/utils.py"
if [ -f "$LONG_UTILS" ] && grep -q "^from moviepy.editor import VideoFileClip" "$LONG_UTILS"; then
  echo "[setup_external] PATCH $LONG_UTILS — remove unused moviepy import"
  sed -i 's/^from moviepy.editor import VideoFileClip/# from moviepy.editor import VideoFileClip  # patched: unused, avoids moviepy dependency/' "$LONG_UTILS"
fi

cat <<EOF

[setup_external] done.

Next:
  uv pip install --no-build-isolation -e external/VBench
EOF
