"""Utilities for refining a diffusion inference schedule with local microsteps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MicrostepSchedule:
    """A raw flow-sigma schedule with extra points inserted into selected intervals.

    ``sigmas`` are pre-shift flow sigmas. Diffusers schedulers apply the model's
    configured flow shift inside ``set_timesteps(sigmas=...)``.
    """

    base_num_steps: int
    effective_num_steps: int
    index_base: int
    extra_steps_by_step: dict[int, int]
    base_to_refined_start: dict[int, int]
    sigmas: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_extra_steps_by_step(
    extra_steps_by_step: Mapping[int | str, int] | None,
    *,
    base_num_steps: int,
    index_base: int = 1,
) -> dict[int, int]:
    """Validate and normalize user-facing step indices.

    ``index_base=1`` means a config key of ``40`` refines the interval that
    starts at the 40th denoising model call. ``index_base=0`` uses Python-style
    zero-based step indices.
    """

    if index_base not in (0, 1):
        raise ValueError(f"index_base must be 0 or 1, got {index_base}")
    if base_num_steps <= 0:
        raise ValueError(f"base_num_steps must be positive, got {base_num_steps}")

    normalized: dict[int, int] = {}
    for raw_step, raw_extra in (extra_steps_by_step or {}).items():
        step = int(raw_step)
        extra = int(raw_extra)
        if extra < 0:
            raise ValueError(f"extra microsteps must be non-negative for step {step}, got {extra}")
        if extra == 0:
            continue
        base_index = step - index_base
        if not 0 <= base_index < base_num_steps:
            lo = index_base
            hi = index_base + base_num_steps - 1
            raise ValueError(f"step {step} is outside [{lo}, {hi}] for {base_num_steps} base steps")
        normalized[step] = extra
    return dict(sorted(normalized.items()))


def build_microstep_schedule(
    *,
    base_num_steps: int,
    extra_steps_by_step: Mapping[int | str, int] | None,
    index_base: int = 1,
    num_train_timesteps: int = 1000,
) -> MicrostepSchedule:
    """Insert evenly spaced raw flow-sigma microsteps into a base schedule.

    An entry ``{40: 5}`` with ``index_base=1`` adds five extra model evaluations
    between the original step-40 sigma and the next lower sigma, splitting that
    one original interval into six smaller intervals.
    """

    if num_train_timesteps <= 0:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")
    extras = normalize_extra_steps_by_step(
        extra_steps_by_step,
        base_num_steps=base_num_steps,
        index_base=index_base,
    )

    min_sigma = 1.0 / float(num_train_timesteps)
    if base_num_steps == 1:
        base_sigmas = [1.0]
    else:
        step = (1.0 - min_sigma) / float(base_num_steps)
        base_sigmas = [1.0 - step * i for i in range(base_num_steps)]

    refined: list[float] = []
    base_to_refined_start: dict[int, int] = {}
    for base_index, high in enumerate(base_sigmas):
        user_step = base_index + index_base
        base_to_refined_start[user_step] = len(refined)
        refined.append(float(high))

        extra = extras.get(user_step, 0)
        if extra <= 0:
            continue

        low = base_sigmas[base_index + 1] if base_index + 1 < len(base_sigmas) else 0.0
        for j in range(1, extra + 1):
            frac = j / float(extra + 1)
            refined.append(float(high + (low - high) * frac))

    for prev, cur in zip(refined, refined[1:]):
        if not prev > cur:
            raise ValueError("microstep sigmas must be strictly decreasing")

    return MicrostepSchedule(
        base_num_steps=base_num_steps,
        effective_num_steps=len(refined),
        index_base=index_base,
        extra_steps_by_step=extras,
        base_to_refined_start=base_to_refined_start,
        sigmas=refined,
    )
