from __future__ import annotations

import math

import pytest

from ttsd.runners.generate.decision_diameter import expected_rms_radius, radial_specs


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
