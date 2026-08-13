"""Small fixed prompt set for late-branching compute benchmarks.

The set spans the prompt-dependency taxonomy, but uses one root seed per prompt
because the goal is wall-clock cost, not statistical quality analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ttsd.prompts.prompt_dependency_v1 import PROMPT_DEPENDENCY_V1_PROMPTS


_PROMPT_IDS = [
    "pd001",
    "pd006",
    "pd013",
    "pd020",
    "pd025",
    "pd032",
    "pd040",
    "pd041",
    "pd048",
    "pd056",
    "pd057",
    "pd064",
    "pd076",
    "pd077",
    "pd084",
    "pd096",
    "pd097",
    "pd104",
    "pd113",
    "pd120",
]


def _build() -> list[dict]:
    by_id = {prompt["id"]: prompt for prompt in PROMPT_DEPENDENCY_V1_PROMPTS}
    missing = [prompt_id for prompt_id in _PROMPT_IDS if prompt_id not in by_id]
    if missing:
        raise KeyError(f"unknown prompt ids: {', '.join(missing)}")
    prompts: list[dict] = []
    for prompt_id in _PROMPT_IDS:
        prompt = dict(by_id[prompt_id])
        prompt["seeds"] = [0]
        prompts.append(prompt)
    return prompts


COMPUTE_COST_BENCHMARK_V1_PROMPTS: list[dict] = _build()


def write_root_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "prompt_id",
        "root_seed",
        "axis",
        "vbench_index",
        "prompt_class",
        "motion_bucket",
        "subject_count",
        "relation_class",
        "camera_class",
        "complexity_class",
        "prompt_text",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for prompt in COMPUTE_COST_BENCHMARK_V1_PROMPTS:
            writer.writerow(
                {
                    "prompt_id": prompt["id"],
                    "root_seed": prompt["seeds"][0],
                    "axis": prompt["axis"],
                    "vbench_index": prompt["vbench_index"],
                    "prompt_class": prompt["prompt_class"],
                    "motion_bucket": prompt["motion_bucket"],
                    "subject_count": prompt["subject_count"],
                    "relation_class": prompt["relation_class"],
                    "camera_class": prompt["camera_class"],
                    "complexity_class": prompt["complexity_class"],
                    "prompt_text": prompt["text"],
                }
            )

