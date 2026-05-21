"""Sweep v2 prompt set (2026-05-20) — extends the original Phase 0 dev set.

Two groups, both loaded from the vendored VBench prompt files (same source as
`dev_set.py`):

  GROUP A (p01–p15): the original 15 VBench prompts used for the 2026-05-04
  Phase 0 run, with seeds 10–19 (the original run covered seeds 0–9 WITHOUT
  posterior means). These 150 new clips backfill 10 fresh seeds that DO have
  posterior means inline, doubling our per-prompt seed coverage to 20.

  GROUP B (p16–p30): 15 NEW prompts, with seeds 0–19 each (300 clips).
  Diversifies coverage: adds temporal_style, temporal_flickering, and extra
  samples in the harder compositional/relational dimensions where the
  original sweep showed the most seed variance.

Total: 150 (A) + 300 (B) = 450 clips. All clips capture posterior means.

Each entry has a `seeds` field — the baseline runner respects per-prompt
seed lists when present (otherwise falls back to the global seeds.* config).
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "external" / "t2v-search" / "CogVideoX" / "verifiers" / "VBench"
    / "prompts" / "prompts_per_dimension"
)


# Group A: same (dim, index) selection as `dev_set.py:_SELECTION` — keep in sync.
_GROUP_A: list[tuple[str, int]] = [
    ("subject_consistency", 0),
    ("subject_consistency", 5),
    ("multiple_objects", 0),
    ("multiple_objects", 7),
    ("spatial_relationship", 0),
    ("spatial_relationship", 10),
    ("human_action", 0),
    ("human_action", 8),
    ("overall_consistency", 0),
    ("overall_consistency", 4),
    ("scene", 0),
    ("scene", 12),
    ("object_class", 5),
    ("appearance_style", 0),
    ("color", 3),
]

# Group B: 15 new prompts, deliberately heavier on the dimensions where the
# 2026-05-04 results showed the highest seed-to-seed variance, plus two new
# dimensions (temporal_style, temporal_flickering).
_GROUP_B: list[tuple[str, int]] = [
    ("multiple_objects", 1),         # "a cat and a dog"
    ("multiple_objects", 15),
    ("spatial_relationship", 1),     # "a car on the right of a motorcycle..."
    ("spatial_relationship", 30),
    ("human_action", 1),             # "A person is marching"
    ("human_action", 20),
    ("human_action", 40),
    ("subject_consistency", 10),
    ("subject_consistency", 20),
    ("overall_consistency", 1),      # "Turtle swimming in ocean."
    ("overall_consistency", 2),      # "A storm trooper vacuuming the beach."
    ("scene", 30),
    ("object_class", 20),
    ("temporal_style", 0),           # "...in super slow motion"
    ("temporal_flickering", 0),      # "In a still frame, a stop sign"
]


def _load_dim(dim: str) -> list[str]:
    path = _PROMPT_DIR / f"{dim}.txt"
    if not path.exists():
        raise FileNotFoundError(f"VBench prompt file missing: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _build() -> list[dict]:
    out: list[dict] = []
    pid = 1
    for dim, idx in _GROUP_A:
        prompts = _load_dim(dim)
        if idx >= len(prompts):
            raise IndexError(f"Group A: index {idx} OOR for '{dim}' (have {len(prompts)})")
        out.append({
            "id": f"p{pid:02d}",
            "axis": dim,
            "vbench_index": idx,
            "text": prompts[idx],
            "seeds": list(range(10, 20)),   # 10 NEW seeds, not overlapping the 2026-05-04 run
        })
        pid += 1
    for dim, idx in _GROUP_B:
        prompts = _load_dim(dim)
        if idx >= len(prompts):
            raise IndexError(f"Group B: index {idx} OOR for '{dim}' (have {len(prompts)})")
        out.append({
            "id": f"p{pid:02d}",
            "axis": dim,
            "vbench_index": idx,
            "text": prompts[idx],
            "seeds": list(range(0, 20)),    # 20 fresh seeds (new prompt)
        })
        pid += 1
    return out


SWEEP_V2_PROMPTS: list[dict] = _build()
