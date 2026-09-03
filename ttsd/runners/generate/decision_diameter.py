from __future__ import annotations

import math
from collections.abc import Sequence


def _alpha_code(alpha: float) -> int:
    alpha = float(alpha)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    code = round(alpha * 10000)
    if not math.isclose(alpha, code / 10000, rel_tol=0, abs_tol=1e-12):
        raise ValueError("alpha must use at most four decimal places")
    return code


def format_sample_id(direction_index: int, alpha: float) -> str:
    if direction_index < 0:
        raise ValueError("direction_index must be non-negative")
    return f"d{direction_index:02d}_a{_alpha_code(alpha):05d}"


def expected_rms_radius(alpha: float) -> float:
    alpha = float(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must satisfy 0 <= alpha <= 1")
    return math.sqrt(2.0 - 2.0 * math.sqrt(1.0 - alpha**2))


def radial_specs(
    alphas: Sequence[float],
    direction_seeds: Sequence[int],
    *,
    start_index: int = 0,
) -> list[dict[str, int | float | str]]:
    values = tuple(float(alpha) for alpha in alphas)
    seeds = tuple(int(seed) for seed in direction_seeds)
    if not values or any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("alphas must be non-empty and strictly increasing")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("direction seeds must be non-empty and unique")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    specs = []
    for alpha in values:
        _alpha_code(alpha)
        for direction_index, perturb_seed in enumerate(seeds):
            specs.append(
                {
                    "index": start_index + len(specs),
                    "direction_index": direction_index,
                    "alpha": alpha,
                    "perturb_seed": perturb_seed,
                    "sample_id": format_sample_id(direction_index, alpha),
                }
            )
    return specs
