"""Phase 0 dev prompt set — 15 real VBench prompts spanning the dimensions
most likely to expose seed-dependent failure on Wan 2.2.

Loaded at import time from the VBench prompt files vendored under
`external/t2v-search/CogVideoX/verifiers/VBench/prompts/prompts_per_dimension/`.

Selection rationale:
- 2 prompts each from the six TTS-relevant dimensions where seed-to-seed
  variance is most likely to surface (subject_consistency, multiple_objects,
  spatial_relationship, human_action, overall_consistency, scene) → 12.
- 1 prompt each from three more (object_class, appearance_style, color) → 3.
- Indices are *fixed* (not random) so the dev set is reproducible. To swap to
  different prompts, edit the index list below — do not introduce randomness.

Using real VBench prompts (not hand-written ones) means later VBench scores
on this dev set are directly comparable to numbers reported elsewhere.
"""

from __future__ import annotations

from pathlib import Path

# Repo-relative path to the vendored VBench prompt directory. Resolved
# relative to this module so the dev set survives `pip install -e .`.
_PROMPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "external" / "t2v-search" / "CogVideoX" / "verifiers" / "VBench"
    / "prompts" / "prompts_per_dimension"
)

# (vbench_dimension, index_in_file) — 15 entries.
_SELECTION: list[tuple[str, int]] = [
    # Single-subject consistency
    ("subject_consistency", 0),
    ("subject_consistency", 5),
    # Multi-object compositionality
    ("multiple_objects", 0),
    ("multiple_objects", 7),
    # Spatial relations (relational reasoning)
    ("spatial_relationship", 0),
    ("spatial_relationship", 10),
    # Human action / motion
    ("human_action", 0),
    ("human_action", 8),
    # Overall text-video alignment
    ("overall_consistency", 0),
    ("overall_consistency", 4),
    # Scene
    ("scene", 0),
    ("scene", 12),
    # Object class
    ("object_class", 5),
    # Appearance style
    ("appearance_style", 0),
    # Color binding
    ("color", 3),
]


def _load_dim(dim: str) -> list[str]:
    path = _PROMPT_DIR / f"{dim}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"VBench prompt file not found: {path}. Did you vendor the shim0114 "
            "repo at external/t2v-search?"
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _build() -> list[dict]:
    out: list[dict] = []
    for i, (dim, idx) in enumerate(_SELECTION, start=1):
        prompts = _load_dim(dim)
        if idx >= len(prompts):
            raise IndexError(
                f"Index {idx} out of range for dimension '{dim}' "
                f"({len(prompts)} prompts available)."
            )
        out.append({
            "id": f"p{i:02d}",
            "axis": dim,
            "vbench_index": idx,
            "text": prompts[idx],
        })
    return out


DEV_PROMPTS: list[dict] = _build()
