from __future__ import annotations

import pytest

from ttsd.eval.late_branch_oracle import analyze_groups, build_summary


def _meta(
    prompt_id: str,
    root_seed: int,
    candidate_seed: int,
    candidate_index: int,
    kind: str,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt_text": f"prompt {prompt_id}",
        "axis": "test",
        "seed": candidate_seed,
        "root_seed": root_seed,
        "candidate_index": candidate_index,
        "branch_kind": kind,
        "branch_step": 35,
    }


def test_oracles_keep_dynamic_and_overall_visible() -> None:
    candidates = {
        ("p01", 0): _meta("p01", 0, 0, 0, "batched_control"),
        ("p01", 1): _meta("p01", 0, 1, 1, "noise"),
        ("p02", 100): _meta("p02", 1, 100, 0, "batched_control"),
        ("p02", 101): _meta("p02", 1, 101, 1, "noise"),
    }
    targets = {
        ("p01", 0): {
            "vbench_quality": 0.2,
            "dynamic_degree": 1.0,
            "overall_consistency": 0.5,
        },
        ("p01", 1): {
            "vbench_quality": 0.3,
            "dynamic_degree": 1.0,
            "overall_consistency": 0.6,
        },
        ("p02", 100): {
            "vbench_quality": 0.8,
            "dynamic_degree": 1.0,
            "overall_consistency": 0.7,
        },
        ("p02", 101): {
            "vbench_quality": 0.9,
            "dynamic_degree": 0.0,
            "overall_consistency": 0.8,
        },
    }

    baseline_targets = {
        ("p01", 0): targets[("p01", 0)],
        ("p02", 1): targets[("p02", 100)],
    }

    rows = analyze_groups(candidates, targets, baseline_targets)
    summary = build_summary(rows, bottom_n=1, bottom_fraction=0.5)
    all_rows = summary["strata"]["all"]

    assert all_rows["quality_oracle_win_rate"] == 1.0
    assert all_rows["same_dynamic_quality_win_rate"] == 0.5
    assert all_rows["safe_quality_win_rate"] == 0.5
    assert all_rows["pareto_improvement_rate"] == 0.5
    assert all_rows["best_noise_over_batched_control_win_rate"] == 1.0
    assert all_rows["quality_selected_mean_dynamic_delta"] == pytest.approx(-0.5)
    assert summary["strata"]["bottom_1_by_baseline_quality"][
        "safe_quality_win_rate"
    ] == 1.0
