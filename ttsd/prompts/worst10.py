"""Worst-10 baseline samples for targeted rescue probes.

Ranking source:
``runs/analysis/all150_comprehensive_vbench_001/worst10_base_no_dynamic6_manifest.csv``

The primary ranking metric is baseline ``no_dynamic6``. Dynamic degree is kept
as a separate reporting dimension so the rescue probe does not reward static
videos through an averaged score.
"""

from __future__ import annotations

from ttsd.prompts.dev_set import DEV_PROMPTS as _DEV_PROMPTS

_SEEDS_BY_PROMPT_ID: dict[str, list[int]] = {
    "p01": [2, 3, 7, 9],
    "p02": [7],
    "p08": [0, 5],
    "p11": [3, 6, 7],
}

WORST10_BASE_NO_DYNAMIC6_PROMPTS: list[dict] = [
    {**prompt, "seeds": _SEEDS_BY_PROMPT_ID[prompt["id"]]}
    for prompt in _DEV_PROMPTS
    if prompt["id"] in _SEEDS_BY_PROMPT_ID
]
