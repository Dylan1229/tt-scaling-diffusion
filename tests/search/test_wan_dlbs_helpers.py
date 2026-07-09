from __future__ import annotations

import pytest
import torch

from ttsd.models.wan22_dlbs import (
    WanDLBSConfig,
    _branch_noise_std,
    _flow_step_between,
    _reward_video_tensor,
)


class DummyConfig:
    num_train_timesteps = 1000


class DummyScheduler:
    config = DummyConfig()
    sigmas = torch.tensor([1.0, 0.5, 0.0])


def test_flow_step_between_matches_flow_x0_identity() -> None:
    sample = torch.tensor([2.0])
    model_output = torch.tensor([3.0])
    scheduler = DummyScheduler()

    actual = _flow_step_between(
        sample,
        model_output,
        scheduler,
        torch.tensor(500.0),
        torch.tensor(0.0),
    )

    assert actual.item() == pytest.approx(0.5)


def test_branch_noise_std_uses_sigma_gap() -> None:
    config = WanDLBSConfig(branch_noise_scale=2.0)

    std = _branch_noise_std(DummyScheduler(), 0, torch.tensor(1000.0), config)

    assert std == pytest.approx((1.0 - 0.25) ** 0.5 * 2.0)


def test_reward_video_tensor_converts_decoded_video_to_uint8_frames() -> None:
    decoded = torch.zeros(1, 3, 6, 2, 2)
    config = WanDLBSConfig(reward_frame_stride=2, reward_max_frames=2)

    video = _reward_video_tensor(decoded, config)

    assert video.shape == (2, 3, 2, 2)
    assert video.dtype == torch.uint8
    assert video.unique().tolist() == [127]
