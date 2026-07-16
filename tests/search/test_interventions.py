from __future__ import annotations

import math

import pytest
import torch

from ttsd.search import (
    DLBSBranchingConfig,
    DLBSLookaheadConfig,
    DLBSLookaheadIntervention,
    DiffusionResamplingConfig,
    DiffusionResamplingIntervention,
    InterventionAction,
    InterventionContext,
    multinomial_resample,
    renoise_branch,
    renoise_with_sigma_gap,
    scalar_score,
)
from ttsd.verifiers.base import Verifier, VerifierOutput


class SumVerifier(Verifier):
    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        step: int,
        total_steps: int,
    ) -> VerifierOutput:
        del prompt, step, total_steps
        return VerifierOutput(score={"reward": float(latent.sum().item())})


def test_dlbs_lookahead_intervention_selects_best_candidate() -> None:
    def branch_fn(
        latent: torch.Tensor,
        ctx: InterventionContext,
        num_candidates: int,
        generator: torch.Generator | None,
    ) -> list[torch.Tensor]:
        del ctx, generator
        return [latent + float(i) for i in range(num_candidates)]

    def lookahead_fn(
        latent: torch.Tensor,
        ctx: InterventionContext,
        num_steps: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del ctx, generator
        return latent * float(num_steps)

    intervention = DLBSLookaheadIntervention(
        verifier=SumVerifier(),
        branch_fn=branch_fn,
        lookahead_fn=lookahead_fn,
        config=DLBSLookaheadConfig(num_candidates=4, lookahead_steps=2),
    )
    ctx = InterventionContext(
        latent=torch.zeros(1),
        prompt="a test prompt",
        step=4,
        total_steps=10,
    )

    result = intervention.intervene(ctx)

    assert result.action is InterventionAction.BRANCH_SELECT
    assert result.selected_index == 3
    assert result.latent.item() == pytest.approx(3.0)
    assert [candidate.score for candidate in result.candidates] == pytest.approx(
        [6.0, 4.0, 2.0, 0.0]
    )


def test_renoise_branch_uses_sigma_gap_for_candidates() -> None:
    latent = torch.zeros(2, 3)
    ctx = InterventionContext(
        latent=latent,
        prompt="a test prompt",
        step=4,
        total_steps=10,
        sigma=1.0,
    )
    config = DLBSBranchingConfig(sigma_multiplier=2.0, include_current=True)
    expected_std = math.sqrt(3.0)

    gen_actual = torch.Generator().manual_seed(123)
    gen_expected = torch.Generator().manual_seed(123)
    candidates = renoise_branch(
        latent,
        ctx,
        num_candidates=3,
        generator=gen_actual,
        config=config,
    )
    expected_first = expected_std * torch.randn(latent.shape, generator=gen_expected)
    expected_second = expected_std * torch.randn(latent.shape, generator=gen_expected)

    assert torch.equal(candidates[0], latent)
    assert torch.allclose(candidates[1], expected_first)
    assert torch.allclose(candidates[2], expected_second)


def test_dlbs_lookahead_can_use_default_renoise_branch() -> None:
    intervention = DLBSLookaheadIntervention(
        verifier=SumVerifier(),
        branch_config=DLBSBranchingConfig(noise_std=0.0),
        config=DLBSLookaheadConfig(num_candidates=2, lookahead_steps=0),
    )
    ctx = InterventionContext(
        latent=torch.ones(1),
        prompt="a test prompt",
        step=4,
        total_steps=10,
        sigma=1.0,
    )

    result = intervention.intervene(ctx)

    assert result.action is InterventionAction.BRANCH_SELECT
    assert result.selected_index == 0
    assert result.latent.item() == pytest.approx(1.0)


def test_scalar_score_requires_key_when_output_is_ambiguous() -> None:
    output = VerifierOutput(score={"a": 1.0, "b": 2.0})

    with pytest.raises(ValueError, match="multiple score keys"):
        scalar_score(output)

    assert scalar_score(output, key="b") == pytest.approx(2.0)


def test_renoise_with_sigma_gap_uses_restart_variance() -> None:
    latent = torch.zeros(2, 3)
    current_sigma = 1.0
    target_sigma = 2.0
    expected_std = math.sqrt((target_sigma * target_sigma) - (current_sigma * current_sigma))

    gen_actual = torch.Generator().manual_seed(123)
    gen_expected = torch.Generator().manual_seed(123)
    actual = renoise_with_sigma_gap(
        latent,
        current_sigma=current_sigma,
        target_sigma=target_sigma,
        generator=gen_actual,
    )
    expected_noise = torch.randn(latent.shape, generator=gen_expected)

    assert torch.allclose(actual, expected_std * expected_noise)


def test_diffusion_resampling_intervention_renoises_from_context_sigma() -> None:
    latent = torch.ones(2, 2)
    intervention = DiffusionResamplingIntervention(
        DiffusionResamplingConfig(sigma_multiplier=1.5, noise_scale=0.0)
    )
    ctx = InterventionContext(
        latent=latent,
        prompt="prompt",
        step=3,
        total_steps=10,
        sigma=2.0,
    )

    result = intervention.intervene(ctx)

    assert result.action is InterventionAction.RENOISE
    assert torch.equal(result.latent, latent)
    assert result.metadata["current_sigma"] == pytest.approx(2.0)
    assert result.metadata["target_sigma"] == pytest.approx(3.0)


def test_multinomial_resample_returns_weighted_particles() -> None:
    latents = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    log_weights = torch.tensor([-100.0, -100.0, 10.0, -100.0])
    generator = torch.Generator().manual_seed(0)

    result = multinomial_resample(latents, log_weights, num_samples=6, generator=generator)

    assert result.indices.tolist() == [2, 2, 2, 2, 2, 2]
    assert result.latents.squeeze(-1).tolist() == [2.0] * 6
    assert result.normalized_weights.argmax().item() == 2
