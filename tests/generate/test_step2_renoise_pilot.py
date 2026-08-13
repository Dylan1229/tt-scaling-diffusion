from __future__ import annotations

from copy import deepcopy

import pytest

from ttsd.runners.generate.step2_renoise_pilot import (
    build_comparison_html,
    validate_config,
)


@pytest.fixture
def config() -> dict:
    return {
        "generation": {"num_inference_steps": 50},
        "renoise": {
            "branch_step": 2,
            "amplitudes": [0.0, 0.2, 0.4, 0.8],
            "root_seed": 0,
            "independent_seed": 1,
            "noise_seed": 10_000_000,
        },
        "prompts": {"ids": ["p01", "p03", "p05"]},
    }


def test_build_comparison_html_contains_three_by_five_synchronized_grid() -> None:
    rows = []
    for prompt_id in ("p01", "p03", "p05"):
        videos = [
            {"label": f"alpha={alpha:.1f}", "path": f"{prompt_id}/alpha_{alpha}/video.mp4"}
            for alpha in (0.0, 0.2, 0.4, 0.8)
        ]
        videos.append(
            {
                "label": "independent seed=1",
                "path": f"{prompt_id}/independent_seed_1/video.mp4",
            }
        )
        rows.append({"prompt_id": prompt_id, "prompt_text": prompt_id, "videos": videos})

    html = build_comparison_html({"rows": rows})

    assert html.count("<video") == 15
    for label in ("alpha=0.0", "alpha=0.2", "alpha=0.4", "alpha=0.8", "independent seed=1"):
        assert label in html
    for attribute in ("autoplay", "muted", "loop", "controls"):
        assert html.count(attribute) >= 15
    assert "currentTime = 0" in html
    assert ".play()" in html


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("renoise", "branch_step"), 3, "branch_step must be 2"),
        (("renoise", "amplitudes"), [0.0, 0.4, 0.8], "amplitudes must be"),
        (("renoise", "root_seed"), 2, "root_seed must be 0"),
        (("renoise", "independent_seed"), 3, "independent_seed must be 1"),
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
