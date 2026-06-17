"""Action ABC + the v1 action repertoire.

Each Action is one unit of intervention. Built lazily from an ActionSpec
(declarative handle) by the orchestrator. New actions added under
ttsd/pipeline/actions/ in later phases register themselves with
@register_action("name").

P1 ships only `Continue` and `Noop` (synonyms). Real actions land in P2-P4:
  - StopAndFail / StopAndAccept              (P2)
  - SingleFrameAnchorInject (EFD&I Trial 1)  (P3)
  - RefinePromptVLM         (EFD&I Trial 2)  (P3)
  - RolloutBoNCandidates / KeepBestCandidate (P4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ttsd.pipeline.core import ActionResult, CostEstimate, TrajectoryState
from ttsd.pipeline.registry import ACTIONS, register_action


@dataclass
class ApplyContext:
    """Read-only context handed to Action.apply()."""
    adapter: object                                # ModelAdapter; typed as object to avoid circular
    budget: object                                 # Budget
    logger: object                                 # JsonlLogger
    prompt: str
    seed: int


class Action(ABC):
    """Pluggable intervention primitive."""

    @abstractmethod
    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult: ...

    @property
    def estimated_cost(self) -> CostEstimate:
        return CostEstimate()


def build_action(spec) -> Action:
    """Resolve an ActionSpec (kind + params) to a concrete Action instance."""
    cls = ACTIONS.get(spec.kind)
    return cls(**spec.params)


@register_action("continue")
@register_action("noop")
class Continue(Action):
    """No-op: let the trajectory keep going. The default 'do nothing' action.

    A DecisionPolicy can return ActionSpec(kind='continue') explicitly, or
    return None — both mean the same thing to the orchestrator. We support
    both because some policies want to *log* their no-op decisions and others
    want to stay quiet.
    """

    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult:
        return ActionResult(status="continue", reason="noop")


@register_action("stop_and_fail")
class StopAndFail(Action):
    """Abort the current trajectory immediately. The strategy can then
    escalate to the next trial (EFD&I-style) or return failure to the caller.

    Triggers an AbortTrajectory through the model adapter's StepDirective.
    Cost: only the work already spent up to this step. The point.
    """

    def __init__(self, reason: str = "policy_triggered"):
        self.reason = reason

    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult:
        return ActionResult(status="abort", reason=self.reason)


@register_action("stop_and_accept")
class StopAndAccept(Action):
    """Commit to the trajectory as-is — verifier says it's good enough."""

    def __init__(self, reason: str = "policy_accepted"):
        self.reason = reason

    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult:
        # The strategy treats trial completion-without-abort as acceptance,
        # so this action records the explicit accept but doesn't change the
        # trajectory. Useful for logging when policy is sure mid-trial.
        return ActionResult(status="accept", reason=self.reason)


# ── EFD&I Trial 1 ─────────────────────────────────────────────────────────────

@register_action("single_frame_anchor_inject")
class SingleFrameAnchorInject(Action):
    """EFD&I Trial 1 — generate a short side-rollout, capture its x0_hat at
    an early step, broadcast it across the video temporal axis, and emit
    `initial_latent` so the next trial starts from a semantically-anchored
    latent rather than fresh noise.

    Faithful-ish to the paper: EFD&I uses num_frames=1 (true single-image).
    We default to a short multi-frame rollout because Wan's VAE temporal
    compression makes 1-frame edge-casey; with `anchor_num_frames=5` the
    latent has temporal length ~2 and broadcasts cleanly. The single-frame
    path is available via `anchor_num_frames=1` if your backbone supports it.

    The semantic-anchor gate (˜s_img ≥ ˜s_0 + δ) from the paper is NOT
    implemented in v1 — the strategy unconditionally runs this trial when it
    appears in the trial sequence. Add the gate later by computing the
    anchor's verifier score and skipping if not improved.

    Params:
        k_img: step index at which to capture the anchor x0_hat (single-frame
               rollout aborts after this step).
        anchor_num_frames: frames in the side-rollout (5 = safe default).
        anchor_steps: total denoising steps in the side-rollout
                      (lower = cheaper; 20 is a reasonable cap).
        blend_alpha: 0..1 mix between fresh noise and broadcast anchor used
                     as initial_latent. 1.0 = pure anchor (strongest bias,
                     hardest to interpret); 0.5 = even mix; 0.0 = no anchor
                     (no-op — equivalent to vanilla generation).
    """

    def __init__(
        self,
        k_img: int = 8,
        anchor_num_frames: int = 5,
        anchor_steps: int = 20,
        blend_alpha: float = 0.5,
    ):
        self.k_img = int(k_img)
        self.anchor_num_frames = int(anchor_num_frames)
        self.anchor_steps = int(anchor_steps)
        self.blend_alpha = float(blend_alpha)

    @property
    def estimated_cost(self) -> CostEstimate:
        # Crude estimate: anchor_num_frames is much smaller than full video,
        # so wall-clock scales roughly with k_img (we abort after step k_img).
        return CostEstimate(wall_clock_s=self.k_img * 1.2, gpu_seconds=self.k_img * 1.2)

    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult:
        from ttsd.pipeline.core import StepDirective, StepState
        from ttsd.pipeline.model_adapter import GenerationRequest

        captured: dict[str, torch.Tensor] = {}

        def _capture(s: StepState) -> StepDirective | None:
            if s.step == self.k_img and s.posterior_mean is not None:
                captured["pm"] = s.posterior_mean.detach().clone()
                # Abort the side-rollout — we have what we need.
                return StepDirective(abort=True)
            return None

        # Reuse the orchestrator's main video resolution so the broadcast
        # latent shape will be compatible with the next trial's generation.
        # (Width and height come from the original request, threaded via
        # state.metadata if the strategy stashed them; else fall back to Wan
        # 480p defaults.)
        height = state.metadata.get("height", 480)
        width = state.metadata.get("width", 832)
        anchor_seed = state.seed + 9999

        side_request = GenerationRequest(
            prompt=ctx.prompt,
            seed=anchor_seed,
            num_frames=self.anchor_num_frames,
            height=height,
            width=width,
            num_inference_steps=self.anchor_steps,
            guidance_scale=5.0,
        )
        side_out = ctx.adapter.generate(side_request, on_step=_capture, capture={"posterior_mean"})
        if "pm" not in captured:
            return ActionResult(
                status="skip",
                reason="anchor side-rollout did not capture posterior_mean",
            )

        anchor = captured["pm"]    # shape (1, C, F_lat_anchor, H_lat, W_lat)
        # Broadcast across the temporal axis: collapse to (1, C, 1, H, W),
        # then we leave the final tile-up to the strategy / model adapter
        # (which knows the next trial's full latent shape).
        broadcast = anchor.mean(dim=2, keepdim=True)

        return ActionResult(
            status="continue",
            cost_spent=CostEstimate(wall_clock_s=self.k_img * 1.2, gpu_seconds=self.k_img * 1.2),
            reason=f"anchor captured at step {self.k_img}, alpha={self.blend_alpha}",
            metadata={
                "overrides": {
                    "_anchor_broadcast_latent": broadcast,
                    "_anchor_blend_alpha": self.blend_alpha,
                    "seed_offset": 1,    # next trial uses a fresh seed too
                },
            },
        )


# ── EFD&I Trial 2 ─────────────────────────────────────────────────────────────

@register_action("refine_prompt_vlm")
class RefinePromptVLM(Action):
    """EFD&I Trial 2 — invoke a VLM with the original prompt (and optionally
    the failed video preview) to produce a refined, intent-preserving prompt.
    The refined prompt overrides the trial's prompt.

    The VLM client is pluggable via `vlm: {kind: '...', params: {...}}`.
    Default `kind: noop` returns the original prompt unchanged (verifies
    plumbing without any API call). `kind: quality_modifier_stub` appends
    common quality boosters (still no network). A real client (Anthropic /
    OpenAI / local LLaVA) can be registered later via `@register_vlm_client`.

    v1 does NOT pass the decoded video preview to the VLM — text-only
    refinement. The frames-aware path lands when we wire decoded-preview
    capture in P4 or beyond.
    """

    def __init__(self, vlm: dict | None = None):
        from ttsd.pipeline.vlm import build_vlm_client
        self._client = build_vlm_client(vlm or {"kind": "noop"})

    def apply(self, state: TrajectoryState, ctx: ApplyContext) -> ActionResult:
        refined = self._client.refine_prompt(ctx.prompt, frames=None)
        if refined == ctx.prompt:
            reason = "vlm returned unchanged prompt (likely NoOp)"
        else:
            reason = "prompt refined by VLM"
        return ActionResult(
            status="continue",
            reason=reason,
            metadata={"overrides": {"prompt": refined}},
        )
