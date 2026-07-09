from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ttsd.search.common import (
    InterventionAction,
    InterventionCandidate,
    InterventionContext,
    InterventionResult,
    scalar_score,
)
from ttsd.verifiers.base import Verifier


BranchFn = Callable[
    [torch.Tensor, InterventionContext, int, torch.Generator | None],
    Sequence[torch.Tensor] | torch.Tensor,
]
LookaheadFn = Callable[
    [torch.Tensor, InterventionContext, int, torch.Generator | None],
    torch.Tensor,
]


def _split_candidate_latents(
    latents: Sequence[torch.Tensor] | torch.Tensor,
    *,
    reference: torch.Tensor,
    expected_count: int,
) -> list[torch.Tensor]:
    if isinstance(latents, torch.Tensor):
        if expected_count == 1 and latents.shape == reference.shape:
            return [latents]
        return [candidate for candidate in latents.unbind(0)]
    return list(latents)


def _identity_lookahead(
    latent: torch.Tensor,
    ctx: InterventionContext,
    num_steps: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    del ctx, num_steps, generator
    return latent


@dataclass(frozen=True)
class DLBSBranchingConfig:
    """Noise settings for the default DLBS local branching primitive."""

    sigma_multiplier: float = 1.05
    noise_scale: float = 1.0
    noise_std: float | None = None
    include_current: bool = False

    def __post_init__(self) -> None:
        if self.sigma_multiplier <= 1.0:
            raise ValueError("sigma_multiplier must be greater than 1")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if self.noise_std is not None and self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


def _ctx_sigma(ctx: InterventionContext) -> float:
    if ctx.sigma is None:
        raise ValueError("InterventionContext.sigma is required for default DLBS branching")
    if isinstance(ctx.sigma, torch.Tensor):
        return float(ctx.sigma.detach().to("cpu").item())
    return float(ctx.sigma)


def _branch_noise_std(ctx: InterventionContext, config: DLBSBranchingConfig) -> float:
    if config.noise_std is not None:
        return config.noise_std * config.noise_scale

    current_sigma = _ctx_sigma(ctx)
    target_sigma = current_sigma * config.sigma_multiplier
    return math.sqrt((target_sigma * target_sigma) - (current_sigma * current_sigma)) * config.noise_scale


def renoise_branch(
    latent: torch.Tensor,
    ctx: InterventionContext,
    num_candidates: int,
    generator: torch.Generator | None = None,
    config: DLBSBranchingConfig | None = None,
) -> list[torch.Tensor]:
    """Create K local DLBS branches by injecting fresh Gaussian noise."""

    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    cfg = config or DLBSBranchingConfig()
    noise_std = _branch_noise_std(ctx, cfg)
    candidates: list[torch.Tensor] = []
    if cfg.include_current:
        candidates.append(latent.clone())

    while len(candidates) < num_candidates:
        noise = torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=latent.dtype,
        )
        candidates.append(latent + noise_std * noise)

    return candidates


def make_renoise_branch_fn(config: DLBSBranchingConfig | None = None) -> BranchFn:
    def _branch(
        latent: torch.Tensor,
        ctx: InterventionContext,
        num_candidates: int,
        generator: torch.Generator | None,
    ) -> list[torch.Tensor]:
        return renoise_branch(
            latent,
            ctx,
            num_candidates,
            generator=generator,
            config=config,
        )

    return _branch


@dataclass(frozen=True)
class DLBSLookaheadConfig:
    num_candidates: int = 4
    lookahead_steps: int = 1
    score_key: str | None = None
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.lookahead_steps < 0:
            raise ValueError("lookahead_steps must be non-negative")


class DLBSLookaheadIntervention:
    """Local DLBS-style repair: branch, look ahead, score, keep the best latent."""

    def __init__(
        self,
        *,
        verifier: Verifier,
        branch_fn: BranchFn | None = None,
        lookahead_fn: LookaheadFn | None = None,
        branch_config: DLBSBranchingConfig | None = None,
        config: DLBSLookaheadConfig | None = None,
    ) -> None:
        self.verifier = verifier
        self.branch_fn = branch_fn or make_renoise_branch_fn(branch_config)
        self.lookahead_fn = lookahead_fn or _identity_lookahead
        self.config = config or DLBSLookaheadConfig()

    def intervene(
        self,
        ctx: InterventionContext,
        *,
        generator: torch.Generator | None = None,
    ) -> InterventionResult:
        raw_candidates = self.branch_fn(
            ctx.latent,
            ctx,
            self.config.num_candidates,
            generator,
        )
        branch_latents = _split_candidate_latents(
            raw_candidates,
            reference=ctx.latent,
            expected_count=self.config.num_candidates,
        )
        if not branch_latents:
            raise ValueError("branch_fn produced no candidates")

        scored: list[InterventionCandidate] = []
        score_step = min(ctx.step + self.config.lookahead_steps, ctx.total_steps - 1)
        for index, candidate_latent in enumerate(branch_latents):
            preview = self.lookahead_fn(
                candidate_latent,
                ctx,
                self.config.lookahead_steps,
                generator,
            )
            verifier_output = self.verifier.score(
                preview,
                ctx.prompt,
                score_step,
                ctx.total_steps,
            )
            scored.append(
                InterventionCandidate(
                    latent=candidate_latent,
                    preview_latent=preview,
                    verifier_output=verifier_output,
                    score=scalar_score(verifier_output, self.config.score_key),
                    metadata={"candidate_index": index},
                )
            )

        ranked = sorted(
            scored,
            key=lambda candidate: candidate.score,
            reverse=self.config.higher_is_better,
        )
        selected = ranked[0]
        selected_index = int(selected.metadata["candidate_index"])
        return InterventionResult(
            latent=selected.latent,
            action=InterventionAction.BRANCH_SELECT,
            selected_index=selected_index,
            candidates=ranked,
            extra_nfe=len(branch_latents) * max(1, self.config.lookahead_steps),
            metadata={
                "num_candidates": len(branch_latents),
                "lookahead_steps": self.config.lookahead_steps,
                "selected_score": selected.score,
            },
        )
