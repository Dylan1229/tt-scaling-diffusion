from __future__ import annotations

import math

import pytest

from ttsd.runners.generate.decision_diameter import (
    analyze_scan,
    expected_rms_radius,
    make_shell_plan,
    next_sample_plan,
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


def test_scan_config_validation_and_hash_are_deterministic() -> None:
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


def test_validate_scan_config_rejects_non_integer_fields() -> None:
    bad_parent_seed = scan_config()
    bad_parent_seed["parent_seed"] = "0"
    with pytest.raises(ValueError, match="parent_seed must be an integer"):
        validate_scan_config(bad_parent_seed)

    bad_version = scan_config()
    bad_version["version"] = True
    with pytest.raises(ValueError, match="version must be an integer"):
        validate_scan_config(bad_version)

    bad_seeds = scan_config()
    bad_seeds["direction_seeds"] = [10000, "10001"]
    with pytest.raises(ValueError, match=r"direction_seeds\[1\] must be an integer"):
        validate_scan_config(bad_seeds)


def test_validate_scan_config_rejects_non_real_alpha_tolerance() -> None:
    bad_alpha = scan_config()
    bad_alpha["coarse_alphas"] = ["0.2", 0.4, 1.0]
    with pytest.raises(ValueError, match=r"coarse_alphas\[0\] must be a real number"):
        validate_scan_config(bad_alpha)

    bad_tolerance = scan_config()
    bad_tolerance["alpha_tolerance"] = "0.02"
    with pytest.raises(ValueError, match="alpha_tolerance must be a real number"):
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


def test_sample_plan_rejects_non_integer_and_non_real_fields() -> None:
    config = validate_scan_config(scan_config())
    plan = make_shell_plan(config, 0.2, start_index=0)

    string_index_plan = {
        "version": plan["version"],
        "kind": plan["kind"],
        "samples": [dict(plan["samples"][0]), dict(plan["samples"][1])],
    }
    string_index_plan["samples"][0]["index"] = "0"
    with pytest.raises(ValueError, match="sample index must be an integer"):
        validate_sample_plan(string_index_plan, config)

    bool_direction_plan = make_shell_plan(config, 0.2, start_index=0)
    bool_direction_plan["samples"][0]["direction_index"] = True
    with pytest.raises(ValueError, match="direction_index must be an integer"):
        validate_sample_plan(bool_direction_plan, config)

    string_seed_plan = make_shell_plan(config, 0.2, start_index=0)
    string_seed_plan["samples"][0]["perturb_seed"] = "10000"
    with pytest.raises(ValueError, match="perturb_seed must be an integer"):
        validate_sample_plan(string_seed_plan, config)

    string_alpha_plan = make_shell_plan(config, 0.2, start_index=0)
    string_alpha_plan["samples"][0]["alpha"] = "0.2"
    with pytest.raises(ValueError, match="alpha must be a real number"):
        validate_sample_plan(string_alpha_plan, config)


def four_direction_config() -> dict[str, object]:
    config = scan_config()
    config["direction_seeds"] = [10000, 10001, 10002, 10003]
    config["coarse_alphas"] = [0.2, 0.4, 0.6, 1.0]
    return validate_scan_config(config)


def manifest_and_labels(label_grid: dict[int, list[str]]):
    config = four_direction_config()
    specs = radial_specs((0.2, 0.4, 0.6, 1.0), config["direction_seeds"])
    for spec in specs:
        spec["metrics"] = {
            "rms_distance": spec["alpha"],
            "cosine_similarity": 1.0 - spec["alpha"],
            "norm_ratio": 1.0,
        }
    labels = [
        {
            "sample_id": spec["sample_id"],
            "label": label_grid[spec["direction_index"]][
                (0.2, 0.4, 0.6, 1.0).index(spec["alpha"])
            ],
        }
        for spec in specs
    ]
    return config, {"neighbors": specs}, labels


def test_analysis_reports_nearest_and_r50_brackets() -> None:
    config, manifest, labels = manifest_and_labels(
        {
            0: ["failure", "success", "success", "success"],
            1: ["failure", "failure", "success", "success"],
            2: ["failure", "failure", "failure", "failure"],
            3: ["failure", "failure", "failure", "failure"],
        }
    )
    profile = analyze_scan(config, manifest, labels)
    assert profile["nearest"]["alpha_interval"] == [0.2, 0.4]
    assert profile["typical"]["alpha_interval"] == [0.4, 0.6]
    assert profile["directions"][0]["status"] == "crossed"
    assert profile["directions"][0]["upper_sample_metrics"] == {
        "rms_distance": 0.4,
        "cosine_similarity": 0.6,
        "norm_ratio": 1.0,
    }
    assert profile["directions"][2]["status"] == "censored"


def test_all_parent_labels_at_alpha_one_are_censored() -> None:
    config, manifest, labels = manifest_and_labels(
        {direction: ["failure"] * 4 for direction in range(4)}
    )
    profile = analyze_scan(config, manifest, labels)
    assert profile["status"] == "censored"
    assert profile["nearest"]["alpha_interval"] is None
    assert profile["typical"]["alpha_interval"] is None
    assert next_sample_plan(config, manifest, labels) is None


def test_ambiguous_label_blocks_new_sampling() -> None:
    config = four_direction_config()
    specs = radial_specs((0.2,), config["direction_seeds"])
    for spec in specs:
        spec["metrics"] = {"rms_distance": 0.2, "cosine_similarity": 0.98, "norm_ratio": 1.0}
    labels = [{"sample_id": spec["sample_id"], "label": "failure"} for spec in specs]
    labels[0]["label"] = "ambiguous"
    manifest = {"neighbors": specs}
    profile = analyze_scan(config, manifest, labels)
    assert profile["status"] == "needs_adjudication"
    assert profile["ambiguous_sample_ids"] == ["d00_a02000"]
    assert next_sample_plan(config, manifest, labels) is None


def test_next_plan_expands_then_refines_crossing_midpoints() -> None:
    config = four_direction_config()
    first_shell = radial_specs((0.2,), config["direction_seeds"])
    for spec in first_shell:
        spec["metrics"] = {"rms_distance": 0.2, "cosine_similarity": 0.98, "norm_ratio": 1.0}
    first_labels = [{"sample_id": spec["sample_id"], "label": "failure"} for spec in first_shell]
    expansion = next_sample_plan(config, {"neighbors": first_shell}, first_labels)
    assert [sample["alpha"] for sample in expansion["samples"]] == [0.4] * 4

    _, manifest, labels = manifest_and_labels(
        {
            0: ["failure", "success", "success", "success"],
            1: ["failure", "failure", "success", "success"],
            2: ["failure", "failure", "failure", "failure"],
            3: ["failure", "failure", "failure", "failure"],
        }
    )
    refinement = next_sample_plan(config, manifest, labels)
    assert refinement["kind"] == "refinement"
    assert [(sample["direction_index"], sample["alpha"]) for sample in refinement["samples"]] == [
        (0, 0.3),
        (1, 0.5),
    ]


def test_analysis_flags_label_return_after_first_flip() -> None:
    config, manifest, labels = manifest_and_labels(
        {
            0: ["failure", "success", "failure", "success"],
            1: ["failure", "failure", "failure", "failure"],
            2: ["failure", "failure", "failure", "failure"],
            3: ["failure", "failure", "failure", "failure"],
        }
    )
    profile = analyze_scan(config, manifest, labels)
    assert profile["directions"][0]["non_monotonic"] is True
