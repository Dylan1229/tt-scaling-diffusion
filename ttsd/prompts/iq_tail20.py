"""Top-20 lowest baseline imaging-quality samples from run 20260511_224405.

This prompt list is for a focused microstep probe. Each prompt keeps only the
seed values that appeared in the baseline imaging-quality tail-20 ranking.
"""

from __future__ import annotations

from ttsd.prompts.dev_set import DEV_PROMPTS as _DEV_PROMPTS

_SEEDS_BY_PROMPT_ID: dict[str, list[int]] = {
    "p01": [2, 7],
    "p02": [5, 7],
    "p07": [3, 7],
    "p08": [0, 3, 5],
    "p11": [3, 4, 7, 9],
    "p14": [2, 3, 5, 6, 7, 8, 9],
}

DEV_PROMPTS: list[dict] = [
    {**prompt, "seeds": _SEEDS_BY_PROMPT_ID[prompt["id"]]}
    for prompt in _DEV_PROMPTS
    if prompt["id"] in _SEEDS_BY_PROMPT_ID
]
