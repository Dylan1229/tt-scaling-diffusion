"""Helpers for branching a diffusion trajectory late in denoising."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LateBranchConfig:
    """Configuration for a single late-stage latent fork.

    ``branch_step`` is 1-based and means "fork after this denoising step".
    ``num_noise_branches`` does not include the optional unperturbed branch.
    That branch still runs in the expanded batch and is not a true batch-one
    baseline; it exists to measure batch-dependent numerical drift.
    """

    branch_step: int = 35
    num_noise_branches: int = 4
    perturbation_scale: float = 0.10
    include_batched_control: bool = True
    noise_seed_offset: int = 10_000_000

    def validate(self, num_inference_steps: int) -> None:
        if not 1 <= self.branch_step < num_inference_steps:
            raise ValueError(
                "branch_step must be in [1, num_inference_steps - 1]; "
                f"got {self.branch_step} for {num_inference_steps} steps"
            )
        if self.num_noise_branches < 1:
            raise ValueError("num_noise_branches must be at least 1")
        if self.perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be positive")
        if self.noise_seed_offset < 0:
            raise ValueError("noise_seed_offset must be non-negative")

    @property
    def total_branches(self) -> int:
        return self.num_noise_branches + int(self.include_batched_control)


@dataclass(frozen=True)
class BranchSpec:
    index: int
    kind: str
    perturbation_seed: int | None
    perturbation_std: float


def sigma_after_step(scheduler, step_index: int) -> float:
    """Return the sigma of the post-step latent for a 0-based step index."""

    sigmas = getattr(scheduler, "sigmas", None)
    sigma_index = step_index + 1
    if sigmas is None or sigma_index >= len(sigmas):
        raise ValueError(
            "scheduler must expose a sigmas sequence with one terminal entry"
        )
    return float(sigmas[sigma_index].detach().to("cpu"))


def fork_latents(
    latents: torch.Tensor,
    *,
    root_seed: int,
    sigma: float,
    config: LateBranchConfig,
) -> tuple[torch.Tensor, list[BranchSpec]]:
    """Fork a batch-one latent into an unperturbed branch and noisy candidates."""

    if latents.shape[0] != 1:
        raise ValueError(f"late branching expects batch size 1, got {latents.shape[0]}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    perturbation_std = sigma * config.perturbation_scale
    branches: list[torch.Tensor] = []
    specs: list[BranchSpec] = []

    if config.include_batched_control:
        branches.append(latents)
        specs.append(
            BranchSpec(
                index=0,
                kind="batched_control",
                perturbation_seed=None,
                perturbation_std=0.0,
            )
        )

    for noise_index in range(config.num_noise_branches):
        perturbation_seed = (
            config.noise_seed_offset
            + root_seed * config.num_noise_branches
            + noise_index
        )
        generator = torch.Generator(device=latents.device).manual_seed(perturbation_seed)
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        branch_index = len(branches)
        branches.append(latents + perturbation_std * noise)
        specs.append(
            BranchSpec(
                index=branch_index,
                kind="noise",
                perturbation_seed=perturbation_seed,
                perturbation_std=perturbation_std,
            )
        )

    return torch.cat(branches, dim=0), specs


def denoising_step_equivalents(
    num_inference_steps: int,
    branch_step: int,
    total_branches: int,
) -> int:
    """Model the compute as one shared prefix plus one suffix per branch."""

    return branch_step + (num_inference_steps - branch_step) * total_branches
