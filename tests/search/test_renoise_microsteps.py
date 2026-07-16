from __future__ import annotations

import pytest

from ttsd.search import RenoiseMicrostepWindow, build_renoise_replay_segment


def test_renoise_replay_segment_replays_original_window_without_extra_steps() -> None:
    base_timesteps = [1000, 900, 800, 700, 600, 500]
    window = RenoiseMicrostepWindow(
        trigger_step=4,
        rollback_to_step=2,
        extra_microsteps=0,
        index_base=1,
    )

    segment = build_renoise_replay_segment(base_timesteps, window)

    assert segment.rollback_index == 1
    assert segment.trigger_index == 3
    assert segment.resume_index == 4
    assert segment.rollback_timestep == pytest.approx(900)
    assert segment.resume_timestep == pytest.approx(600)
    assert segment.base_replay_calls == 3
    assert segment.extra_microsteps == 0
    assert segment.extra_nfe == 3
    assert segment.replay_timesteps == pytest.approx([900, 800, 700])


def test_renoise_replay_segment_adds_uniform_microsteps_over_rollback_span() -> None:
    base_timesteps = [1000, 900, 800, 700, 600, 500]
    window = RenoiseMicrostepWindow(
        trigger_step=4,
        rollback_to_step=2,
        extra_microsteps=3,
        index_base=1,
    )

    segment = build_renoise_replay_segment(base_timesteps, window)

    assert segment.base_replay_calls == 3
    assert segment.extra_microsteps == 3
    assert segment.extra_nfe == 6
    assert segment.replay_timesteps == pytest.approx([900, 850, 800, 750, 700, 650])


def test_renoise_window_requires_earlier_rollback_step() -> None:
    with pytest.raises(ValueError, match="earlier than trigger"):
        RenoiseMicrostepWindow(trigger_step=4, rollback_to_step=4)


def test_renoise_replay_segment_rejects_final_trigger_step() -> None:
    base_timesteps = [1000, 900, 800]
    window = RenoiseMicrostepWindow(
        trigger_step=3,
        rollback_to_step=1,
        index_base=1,
    )

    with pytest.raises(ValueError, match="final denoising step"):
        build_renoise_replay_segment(base_timesteps, window)
