from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ttsd.models.wan22_adapter import Wan22Adapter
from ttsd.runners.generate.late_branching import _filter_work_by_pairs
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


def test_fixed_noise_stride_makes_best_of_m_directions_nested() -> None:
    latents = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    common = {
        "branch_step": 3,
        "perturbation_scale": 0.25,
        "noise_seed_offset": 100,
        "noise_seed_stride": 1000,
    }
    forked_m4, specs_m4 = fork_latents(
        latents,
        root_seed=7,
        sigma=0.4,
        config=LateBranchConfig(num_noise_branches=4, **common),
    )
    forked_m8, specs_m8 = fork_latents(
        latents,
        root_seed=7,
        sigma=0.4,
        config=LateBranchConfig(num_noise_branches=8, **common),
    )

    assert torch.equal(forked_m4, forked_m8[:5])
    assert specs_m4 == specs_m8[:5]
    assert [spec.perturbation_seed for spec in specs_m4[1:]] == [
        7100,
        7101,
        7102,
        7103,
    ]


def test_sigma_and_compute_accounting() -> None:
    scheduler = SimpleNamespace(sigmas=torch.tensor([1.0, 0.8, 0.5, 0.0]))
    assert sigma_after_step(scheduler, 1) == pytest.approx(0.5)
    assert denoising_step_equivalents(50, branch_step=35, total_branches=5) == 110


class _FakeScheduler:
    def __init__(self) -> None:
        self.sigmas = torch.linspace(1.0, 0.0, 6)
        self.step_index = 0

    def convert_model_output(self, model_output, sample):
        return sample - model_output

    def step(self, model_output, timestep, sample):
        del model_output, timestep
        return SimpleNamespace(prev_sample=sample)


class _FakeWanPipe:
    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()
        self.batch_sizes: list[int] = []
        self.prompt_batch_sizes: list[int] = []

    def __call__(self, **kwargs):
        latents = torch.zeros((1, 1, 1, 1, 1))
        prompt_embeds = torch.ones((1, 2, 3))
        negative_prompt_embeds = -prompt_embeds
        callback = kwargs["callback_on_step_end"]
        for step_idx in range(kwargs["num_inference_steps"]):
            latents = self.scheduler.step(
                torch.full_like(latents, 0.25),
                torch.tensor(step_idx),
                latents,
            ).prev_sample
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


def test_adapter_captures_candidate_posterior_means_after_fork() -> None:
    adapter = Wan22Adapter(device="cpu")
    adapter._pipe = _FakeWanPipe()
    result = adapter.generate_with_late_branches(
        prompt="test",
        seed=3,
        branch_config=LateBranchConfig(branch_step=2, num_noise_branches=2),
        num_inference_steps=5,
        posterior_mean_offsets=[1, 2],
    )

    assert sorted(result.posterior_means_by_step) == [2, 3]
    assert result.posterior_means_by_step[2].shape[0] == 3


def test_filter_work_by_explicit_prompt_seed_pairs_preserves_pair_order() -> None:
    p01 = {"id": "p01"}
    p02 = {"id": "p02"}
    work = [(p01, 0), (p01, 1), (p02, 0), (p02, 1)]

    selected = _filter_work_by_pairs(work, [("p02", 1), ("p01", 0)])

    assert selected == [(p02, 1), (p01, 0)]
