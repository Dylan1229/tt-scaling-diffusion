from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ttsd.runners.generate.step2_renoise_pilot import (
    _amplitude_slug,
    _row_from_artifacts,
    build_comparison_html,
    validate_config,
)


@pytest.fixture
def config() -> dict:
    return {
        "generation": {"num_inference_steps": 50},
        "renoise": {
            "branch_step": 2,
            "amplitudes": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "root_seed": 0,
            "independent_seed": None,
            "noise_seed": 10_000_000,
        },
        "prompts": {"ids": ["p01", "p03", "p05"]},
    }


@pytest.fixture
def step35_config(config: dict) -> dict:
    changed = deepcopy(config)
    changed["renoise"].update(
        branch_step=35,
        amplitudes=[0.0, 0.4, 0.6, 0.8, 1.0],
        independent_seed=None,
    )
    return changed


def test_build_comparison_html_contains_three_by_six_synchronized_grid() -> None:
    rows = []
    for prompt_id in ("p01", "p03", "p05"):
        videos = [
            {"label": f"alpha={alpha:.1f}", "path": f"{prompt_id}/alpha_{alpha}/video.mp4"}
            for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        rows.append({"prompt_id": prompt_id, "prompt_text": prompt_id, "videos": videos})

    html = build_comparison_html({"branch_step": 2, "rows": rows})

    assert "Step-2 RENOISE Visual Pilot" in html
    assert html.count("<video") == 18
    assert "repeat(6, minmax(260px, 1fr))" in html
    assert "independent seed" not in html
    for label in (
        "alpha=0.0",
        "alpha=0.2",
        "alpha=0.4",
        "alpha=0.6",
        "alpha=0.8",
        "alpha=1.0",
    ):
        assert label in html
    for attribute in ("autoplay", "muted", "loop", "controls"):
        assert html.count(attribute) >= 18
    assert "currentTime = 0" in html
    assert ".play()" in html


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("renoise", "branch_step"), 3, "branch_step must be one of"),
        (("renoise", "amplitudes"), [0.0, 0.4, 0.8], "amplitudes must be"),
        (("renoise", "root_seed"), 2, "root_seed must be 0"),
        (("renoise", "independent_seed"), 3, "independent_seed must be None"),
        (("prompts", "ids"), ["p01"], "prompt ids must be"),
    ],
)
def test_validate_config_rejects_changes_to_fixed_pilot(
    config: dict, path: tuple[str, str], value, match: str
) -> None:
    changed = deepcopy(config)
    changed[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=match):
        validate_config(changed)


def test_validate_config_accepts_exact_pilot(config: dict) -> None:
    validate_config(config)


def test_validate_config_accepts_step35_pilot(step35_config: dict) -> None:
    validate_config(step35_config)


def test_step35_comparison_uses_five_amplitudes_without_independent_seed(
    tmp_path: Path,
) -> None:
    amplitudes = [0.0, 0.4, 0.6, 0.8, 1.0]
    for amplitude in amplitudes:
        path = tmp_path / "p01" / _amplitude_slug(amplitude) / "video.mp4"
        path.parent.mkdir(parents=True)
        path.touch()

    row = _row_from_artifacts(
        tmp_path,
        {"id": "p01", "text": "a person swimming in ocean"},
        amplitudes,
        independent_seed=None,
    )
    manifest = {"branch_step": 35, "rows": [row]}
    comparison = build_comparison_html(manifest)

    assert [video["label"] for video in row["videos"]] == [
        "alpha=0.0",
        "alpha=0.4",
        "alpha=0.6",
        "alpha=0.8",
        "alpha=1.0",
    ]
    assert "Step-35 RENOISE Visual Pilot" in comparison
    assert comparison.count("<video") == 5
    assert "repeat(5, minmax(260px, 1fr))" in comparison
