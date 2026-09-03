from __future__ import annotations

import math
from pathlib import Path

import pytest

from ttsd.runners.generate.decision_diameter import (
    expected_rms_radius,
    make_shell_plan,
    radial_specs,
    scan_config_sha256,
    validate_sample_plan,
    validate_scan_config,
)


def test_radial_specs_reuse_each_direction_seed_at_every_alpha() -> None:
    assert radial_specs((0.2, 1.0), (10000, 10001), start_index=7) == [
        {
            "index": 7,
            "direction_index": 0,
            "alpha": 0.2,
            "perturb_seed": 10000,
            "sample_id": "d00_a02000",
        },
        {
            "index": 8,
            "direction_index": 1,
            "alpha": 0.2,
            "perturb_seed": 10001,
            "sample_id": "d01_a02000",
        },
        {
            "index": 9,
            "direction_index": 0,
            "alpha": 1.0,
            "perturb_seed": 10000,
            "sample_id": "d00_a10000",
        },
        {
            "index": 10,
            "direction_index": 1,
            "alpha": 1.0,
            "perturb_seed": 10001,
            "sample_id": "d01_a10000",
        },
    ]


def test_radial_specs_reject_invalid_or_unrepresentable_inputs() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        radial_specs((0.2, 0.1), (10000,))
    with pytest.raises(ValueError, match="unique"):
        radial_specs((0.2,), (10000, 10000))
    with pytest.raises(ValueError, match="four decimal places"):
        radial_specs((0.20001,), (10000,))


def test_expected_rms_radius_has_independent_noise_endpoint() -> None:
    assert expected_rms_radius(0.0) == 0.0
    assert expected_rms_radius(1.0) == pytest.approx(math.sqrt(2.0))


def scan_config() -> dict[str, object]:
    return {
        "version": 1,
        "prompt": "A bud opens.",
        "input_path": "runs/flower/input.png",
        "input_sha256": "a" * 64,
        "parent_seed": 0,
        "parent_label": "failure",
        "semantic_criterion": "The petals visibly open.",
        "direction_seeds": [10000, 10001],
        "coarse_alphas": [0.2, 0.4, 1.0],
        "alpha_tolerance": 0.02,
    }


def test_scan_config_validation_and_hash_are_deterministic(tmp_path: Path) -> None:
    first = validate_scan_config(scan_config())
    reordered = dict(reversed(list(scan_config().items())))
    second = validate_scan_config(reordered)
    assert first == second
    assert scan_config_sha256(first) == scan_config_sha256(second)
    assert len(scan_config_sha256(first)) == 64


def test_scan_config_requires_independent_endpoint_binary_parent_and_representable_tolerance() -> None:
    bad_endpoint = scan_config()
    bad_endpoint["coarse_alphas"] = [0.2, 0.8]
    with pytest.raises(ValueError, match="end at 1.0"):
        validate_scan_config(bad_endpoint)

    bad_label = scan_config()
    bad_label["parent_label"] = "ambiguous"
    with pytest.raises(ValueError, match="success or failure"):
        validate_scan_config(bad_label)

    bad_tolerance = scan_config()
    bad_tolerance["alpha_tolerance"] = 0.00001
    with pytest.raises(ValueError, match="at least 0.0001"):
        validate_scan_config(bad_tolerance)


def test_shell_plan_uses_configured_direction_seeds() -> None:
    config = validate_scan_config(scan_config())
    assert make_shell_plan(config, 0.2, start_index=4) == {
        "version": 1,
        "kind": "coarse",
        "samples": [
            {
                "index": 4,
                "direction_index": 0,
                "alpha": 0.2,
                "perturb_seed": 10000,
                "sample_id": "d00_a02000",
            },
            {
                "index": 5,
                "direction_index": 1,
                "alpha": 0.2,
                "perturb_seed": 10001,
                "sample_id": "d01_a02000",
            },
        ],
    }


def test_sample_plan_rejects_direction_seed_mismatch() -> None:
    config = validate_scan_config(scan_config())
    plan = make_shell_plan(config, 0.2, start_index=0)
    plan["samples"][0]["perturb_seed"] = 999
    with pytest.raises(ValueError, match="direction 0.*10000"):
        validate_sample_plan(plan, config)
