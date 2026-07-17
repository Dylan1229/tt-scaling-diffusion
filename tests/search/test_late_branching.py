from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ttsd.models.wan22_adapter import Wan22Adapter
from ttsd.search.late_branching import (
    LateBranchConfig,
    denoising_step_equivalents,
    fork_latents,
    sigma_after_step,
)


def test_config_uses_one_based_branch_step() -> None:
    config = LateBranchConfig(branch_step=35, num_noise_branches=4)
    config.validate(50)
    assert config.total_branches == 5
    with pytest.raises(ValueError, match="num_inference_steps - 1"):
        LateBranchConfig(branch_step=50).validate(50)


def test_fork_keeps_unperturbed_branch_and_is_deterministic() -> None:
    latents = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    config = LateBranchConfig(
        branch_step=3,
        num_noise_branches=2,
        perturbation_scale=0.25,
        noise_seed_offset=100,
    )

    forked_a, specs_a = fork_latents(latents, root_seed=7, sigma=0.4, config=config)
    forked_b, specs_b = fork_latents(latents, root_seed=7, sigma=0.4, config=config)

    assert forked_a.shape == (3, 2, 2, 2)
    assert torch.equal(forked_a[0], latents[0])
    assert not torch.equal(forked_a[1], forked_a[2])
    assert torch.equal(forked_a, forked_b)
    assert specs_a == specs_b
    assert [spec.kind for spec in specs_a] == ["batched_control", "noise", "noise"]
    assert [spec.perturbation_seed for spec in specs_a] == [None, 114, 115]
    assert specs_a[1].perturbation_std == pytest.approx(0.1)


def test_sigma_and_compute_accounting() -> None:
    scheduler = SimpleNamespace(sigmas=torch.tensor([1.0, 0.8, 0.5, 0.0]))
    assert sigma_after_step(scheduler, 1) == pytest.approx(0.5)
    assert denoising_step_equivalents(50, branch_step=35, total_branches=5) == 110


class _FakeWanPipe:
    def __init__(self) -> None:
        self.scheduler = SimpleNamespace(sigmas=torch.linspace(1.0, 0.0, 6))
        self.batch_sizes: list[int] = []
        self.prompt_batch_sizes: list[int] = []

    def __call__(self, **kwargs):
        latents = torch.zeros((1, 1, 1, 1, 1))
        prompt_embeds = torch.ones((1, 2, 3))
        negative_prompt_embeds = -prompt_embeds
        callback = kwargs["callback_on_step_end"]
        for step_idx in range(kwargs["num_inference_steps"]):
            callback_outputs = callback(
                self,
                step_idx,
                torch.tensor(step_idx),
                {
                    "latents": latents,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                },
            )
            latents = callback_outputs["latents"]
            prompt_embeds = callback_outputs["prompt_embeds"]
            negative_prompt_embeds = callback_outputs["negative_prompt_embeds"]
            self.batch_sizes.append(latents.shape[0])
            self.prompt_batch_sizes.append(prompt_embeds.shape[0])
        frames = torch.zeros((latents.shape[0], 2, 2, 2, 3))
        return SimpleNamespace(frames=frames)


def test_adapter_expands_latents_and_prompt_embeddings_together() -> None:
    adapter = Wan22Adapter(device="cpu")
    fake_pipe = _FakeWanPipe()
    adapter._pipe = fake_pipe
    result = adapter.generate_with_late_branches(
        prompt="test",
        seed=3,
        branch_config=LateBranchConfig(branch_step=2, num_noise_branches=2),
        num_inference_steps=5,
    )

    assert fake_pipe.batch_sizes == [1, 3, 3, 3, 3]
    assert fake_pipe.prompt_batch_sizes == fake_pipe.batch_sizes
    assert len(result.frames_by_branch) == 3
    assert result.branch_step == 2
    assert result.denoising_step_equivalents == 11
