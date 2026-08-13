"""Prompt-dependency study set for late-stage branching.

The goal is to cover enough prompt types to test whether Step35 branching has
prompt-dependent behavior, while keeping the first full sweep affordable.
Prompts are drawn from the vendored VBench prompt files so the evaluator can use
the original axis-bound dimensions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_PROMPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "t2v-search"
    / "CogVideoX"
    / "verifiers"
    / "VBench"
    / "prompts"
    / "prompts_per_dimension"
)

_SEEDS = [0, 1, 2]

# 120 prompts total:
# - 56 static/compositional prompts
# - 64 motion/camera/mixed prompts
_SELECTIONS: list[tuple[str, list[int], str]] = [
    ("object_class", list(range(0, 12)), "static_single_or_scene"),
    ("scene", list(range(0, 12)), "static_single_or_scene"),
    ("multiple_objects", list(range(0, 16)), "multi_object"),
    ("spatial_relationship", list(range(0, 16)), "spatial_relation"),
    ("human_action", list(range(0, 20)), "human_action"),
    ("subject_consistency", list(range(0, 20)), "subject_motion"),
    ("temporal_style", list(range(0, 16)), "camera_motion"),
    ("overall_consistency", list(range(0, 8)), "mixed_story"),
]


def _load_dim(dim: str) -> list[str]:
    path = _PROMPT_DIR / f"{dim}.txt"
    if not path.exists():
        raise FileNotFoundError(f"VBench prompt file missing: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _camera_class(text: str, axis: str) -> str:
    lowered = text.lower()
    if axis != "temporal_style":
        return "none"
    if "zoom in" in lowered:
        return "zoom_in"
    if "zoom out" in lowered:
        return "zoom_out"
    if "pan left" in lowered:
        return "pan_left"
    if "pan right" in lowered:
        return "pan_right"
    if "tilt up" in lowered:
        return "tilt_up"
    if "tilt down" in lowered:
        return "tilt_down"
    if "shaking" in lowered:
        return "shake"
    if "steady and smooth" in lowered:
        return "steady"
    if "racking focus" in lowered:
        return "racking_focus"
    if "slow motion" in lowered:
        return "slow_motion"
    return "temporal_style_other"


def _motion_bucket(axis: str, camera_class: str) -> str:
    if axis in {"object_class", "scene", "multiple_objects", "spatial_relationship"}:
        return "static_prompt"
    if axis == "human_action":
        return "human_motion"
    if axis == "subject_consistency":
        return "subject_motion"
    if axis == "temporal_style":
        if camera_class in {"slow_motion", "steady"}:
            return "low_or_stylized_motion"
        return "camera_motion"
    return "mixed_motion"


def _subject_count(axis: str, text: str) -> str:
    lowered = text.lower()
    if axis in {"multiple_objects", "spatial_relationship"}:
        return "two_subjects"
    if axis == "scene":
        return "scene_only"
    if "two " in lowered or "couple" in lowered or "colleagues" in lowered:
        return "multi_subject"
    if axis == "human_action" or "person" in lowered:
        return "single_person"
    return "single_subject"


def _relation_class(axis: str) -> str:
    if axis == "spatial_relationship":
        return "explicit_spatial_relation"
    if axis == "multiple_objects":
        return "co_presence"
    return "none"


def _complexity_class(axis: str, prompt_class: str) -> str:
    if prompt_class in {"static_single_or_scene", "multi_object"}:
        return "simple"
    if prompt_class in {"spatial_relation", "human_action", "subject_motion", "camera_motion"}:
        return "moderate"
    return "complex"


def _build() -> list[dict]:
    out: list[dict] = []
    pid = 1
    for axis, indices, prompt_class in _SELECTIONS:
        prompts = _load_dim(axis)
        for idx in indices:
            if idx >= len(prompts):
                raise IndexError(f"Index {idx} out of range for '{axis}'")
            text = prompts[idx]
            camera_class = _camera_class(text, axis)
            out.append(
                {
                    "id": f"pd{pid:03d}",
                    "axis": axis,
                    "vbench_index": idx,
                    "text": text,
                    "seeds": list(_SEEDS),
                    "prompt_class": prompt_class,
                    "motion_bucket": _motion_bucket(axis, camera_class),
                    "subject_count": _subject_count(axis, text),
                    "relation_class": _relation_class(axis),
                    "camera_class": camera_class,
                    "complexity_class": _complexity_class(axis, prompt_class),
                }
            )
            pid += 1
    return out


PROMPT_DEPENDENCY_V1_PROMPTS: list[dict] = _build()


def write_prompt_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "prompt_id",
        "axis",
        "vbench_index",
        "prompt_class",
        "motion_bucket",
        "subject_count",
        "relation_class",
        "camera_class",
        "complexity_class",
        "seeds",
        "prompt_text",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for prompt in PROMPT_DEPENDENCY_V1_PROMPTS:
            writer.writerow(
                {
                    "prompt_id": prompt["id"],
                    "axis": prompt["axis"],
                    "vbench_index": prompt["vbench_index"],
                    "prompt_class": prompt["prompt_class"],
                    "motion_bucket": prompt["motion_bucket"],
                    "subject_count": prompt["subject_count"],
                    "relation_class": prompt["relation_class"],
                    "camera_class": prompt["camera_class"],
                    "complexity_class": prompt["complexity_class"],
                    "seeds": " ".join(str(seed) for seed in prompt["seeds"]),
                    "prompt_text": prompt["text"],
                }
            )


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
        for prompt in PROMPT_DEPENDENCY_V1_PROMPTS:
            for root_seed in prompt["seeds"]:
                writer.writerow(
                    {
                        "prompt_id": prompt["id"],
                        "root_seed": root_seed,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--root-manifest", type=Path)
    args = parser.parse_args(argv)

    if args.prompt_manifest is None and args.root_manifest is None:
        print(f"prompts={len(PROMPT_DEPENDENCY_V1_PROMPTS)} roots={len(PROMPT_DEPENDENCY_V1_PROMPTS) * len(_SEEDS)}")
        return
    if args.prompt_manifest is not None:
        write_prompt_manifest(args.prompt_manifest)
    if args.root_manifest is not None:
        write_root_manifest(args.root_manifest)


if __name__ == "__main__":
    main()
