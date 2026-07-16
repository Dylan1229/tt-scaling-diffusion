from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _as_float(value: int | float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("timestep tensors must be scalar")
        return float(value.detach().to("cpu").item())
    return float(value)


@dataclass(frozen=True)
class RenoiseMicrostepWindow:
    """A local restart window followed by extra replay microsteps.

    User-facing step indices follow ``index_base``. With ``index_base=1``,
    ``trigger_step=20, rollback_to_step=18`` means:

    1. Run the normal denoising step 20.
    2. Re-noise the post-step latent back to step-18 noise level.
    3. Replay the step-18..step-20 window, adding ``extra_microsteps`` extra
       model calls spread uniformly over that rollback interval.
    4. Continue from the normal post-step-20 resume level.
    """

    trigger_step: int
    rollback_to_step: int
    extra_microsteps: int = 5
    noise_scale: float = 1.0
    index_base: int = 1

    def __post_init__(self) -> None:
        if self.index_base not in (0, 1):
            raise ValueError("index_base must be 0 or 1")
        if self.extra_microsteps < 0:
            raise ValueError("extra_microsteps must be non-negative")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if self.rollback_to_step >= self.trigger_step:
            raise ValueError("rollback_to_step must be earlier than trigger_step")

    @property
    def trigger_index(self) -> int:
        return self.trigger_step - self.index_base

    @property
    def rollback_index(self) -> int:
        return self.rollback_to_step - self.index_base


@dataclass(frozen=True)
class RenoiseReplaySegment:
    rollback_index: int
    trigger_index: int
    resume_index: int
    rollback_timestep: float
    resume_timestep: float
    replay_timesteps: list[float]

    @property
    def base_replay_calls(self) -> int:
        return self.trigger_index - self.rollback_index + 1

    @property
    def extra_microsteps(self) -> int:
        return len(self.replay_timesteps) - self.base_replay_calls

    @property
    def extra_nfe(self) -> int:
        return len(self.replay_timesteps)


def build_renoise_replay_segment(
    base_timesteps: Sequence[int | float | torch.Tensor],
    window: RenoiseMicrostepWindow,
) -> RenoiseReplaySegment:
    """Build replay timesteps for a rollback+microstep intervention.

    The returned ``replay_timesteps`` are model-call timesteps. They include the
    rollback endpoint and exclude the resume endpoint, because the last replay
    call advances the latent to ``resume_timestep``.
    """

    if len(base_timesteps) < 2:
        raise ValueError("base_timesteps must contain at least two steps")

    rollback_index = window.rollback_index
    trigger_index = window.trigger_index
    resume_index = trigger_index + 1

    if rollback_index < 0:
        raise ValueError("rollback_to_step is before the first denoising step")
    if trigger_index >= len(base_timesteps):
        raise ValueError("trigger_step is after the final denoising step")

    rollback_timestep = _as_float(base_timesteps[rollback_index])
    # The final denoising step resumes at x0 (sigma/timestep 0). Supporting
    # this endpoint lets a checkpoint grid include step 50 just like the
    # existing AddSteps sweep.
    resume_timestep = (
        _as_float(base_timesteps[resume_index])
        if resume_index < len(base_timesteps)
        else 0.0
    )
    if rollback_timestep <= resume_timestep:
        raise ValueError(
            "base_timesteps must be descending over the rollback window "
            f"({rollback_timestep} <= {resume_timestep})"
        )

    base_replay_calls = trigger_index - rollback_index + 1
    total_replay_calls = base_replay_calls + window.extra_microsteps
    span = resume_timestep - rollback_timestep
    replay_timesteps = [
        float(rollback_timestep + span * (i / total_replay_calls))
        for i in range(total_replay_calls)
    ]

    return RenoiseReplaySegment(
        rollback_index=rollback_index,
        trigger_index=trigger_index,
        resume_index=resume_index,
        rollback_timestep=rollback_timestep,
        resume_timestep=resume_timestep,
        replay_timesteps=replay_timesteps,
    )
