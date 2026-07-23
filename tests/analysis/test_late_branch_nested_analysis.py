from __future__ import annotations

import pandas as pd

from ttsd.runners.analysis.late_branch_nested_analysis import (
    _nested_curve,
    _training_threshold,
)


def test_nested_oracle_curve_is_monotonic_in_m() -> None:
    candidates = pd.DataFrame(
        [
            {
                "prompt_id": "p01",
                "root_seed": 0,
                "candidate_index": 1,
                "quality_delta": -0.1,
                "quality_win": False,
                "safe_win": False,
            },
            {
                "prompt_id": "p01",
                "root_seed": 0,
                "candidate_index": 2,
                "quality_delta": 0.2,
                "quality_win": True,
                "safe_win": True,
            },
        ]
    )

    curve = _nested_curve(candidates, [1, 2], "test", "test")

    assert curve["quality_win"].tolist() == [0.0, 1.0]
    assert curve["safe_win"].tolist() == [0.0, 1.0]
    assert curve["quality_gain"].tolist() == [0.0, 0.2]


def test_training_threshold_abstains_when_precision_target_is_unmet() -> None:
    top_rows = pd.DataFrame(
        {
            "probability": [0.9, 0.8, 0.7, 0.6],
            "safe_win": [False, False, True, False],
        }
    )

    assert _training_threshold(top_rows, 0.8, minimum_accepts=2) == float("inf")
