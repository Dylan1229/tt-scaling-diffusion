from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_T2V_SEARCH_ROOT = Path("/data/datasets/peihao/T2V-Diffusion-Search")


@dataclass(frozen=True)
class VBenchRewardWeights:
    subject_consistency: float = 0.0
    motion_smoothness: float = 0.0
    dynamic_degree: float = 0.0
    aesthetic: float = 0.25
    overall_consistency: float = 1.0

    @classmethod
    def from_sequence(cls, weights: list[float] | tuple[float, ...]) -> "VBenchRewardWeights":
        if len(weights) != 5:
            raise ValueError("VBench reward weights must have 5 values")
        return cls(
            subject_consistency=float(weights[0]),
            motion_smoothness=float(weights[1]),
            dynamic_degree=float(weights[2]),
            aesthetic=float(weights[3]),
            overall_consistency=float(weights[4]),
        )


@dataclass(frozen=True)
class VBenchRewardConfig:
    weights: VBenchRewardWeights = VBenchRewardWeights()
    t2v_search_root: Path | None = None
    vbench_root: Path | None = None
    pretrained_root: Path | None = None
    device: str = "cuda"

    @classmethod
    def from_weight_sequence(
        cls,
        weights: list[float] | tuple[float, ...],
        **kwargs: Any,
    ) -> "VBenchRewardConfig":
        return cls(weights=VBenchRewardWeights.from_sequence(weights), **kwargs)


@dataclass(frozen=True)
class VBenchRewardOutput:
    reward: float
    details: dict[str, float]


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _resolve_t2v_search_root(path: Path | None) -> Path:
    if path is not None:
        return path
    env_path = os.environ.get("TTSD_T2V_SEARCH_ROOT")
    if env_path:
        return Path(env_path)
    return DEFAULT_T2V_SEARCH_ROOT


def _resolve_vbench_root(t2v_search_root: Path, vbench_root: Path | None) -> Path:
    if vbench_root is not None:
        return vbench_root
    for candidate in (
        t2v_search_root / "CogVideoX" / "verifiers" / "VBench",
        t2v_search_root / "verifiers" / "VBench",
        Path("external") / "VBench",
        Path("external") / "t2v-search" / "CogVideoX" / "verifiers" / "VBench",
    ):
        if candidate.exists():
            return candidate.resolve()
    return t2v_search_root / "CogVideoX" / "verifiers" / "VBench"


def _resolve_pretrained_root(t2v_search_root: Path, pretrained_root: Path | None) -> Path:
    if pretrained_root is not None:
        return pretrained_root
    env_path = os.environ.get("TTSD_DLBS_PRETRAINED_ROOT")
    if env_path:
        return Path(env_path)
    for candidate in (
        t2v_search_root / "CogVideoX" / "pretrained",
        t2v_search_root / "pretrained",
        Path("external") / "t2v-search" / "CogVideoX" / "pretrained",
    ):
        if candidate.exists():
            return candidate.resolve()
    return t2v_search_root / "CogVideoX" / "pretrained"


def _dynamic_degree_mapping(value: float) -> float:
    return float(np.log(value) / 16.0)


class VBenchWeightedReward:
    """DLBS-style weighted reward over decoded preview videos.

    This ports the reward logic used by T2V-Diffusion-Search. It keeps the heavy
    VBench models lazy so importing ttsd remains cheap.
    """

    def __init__(self, config: VBenchRewardConfig | None = None) -> None:
        self.config = config or VBenchRewardConfig()
        self.t2v_search_root = _resolve_t2v_search_root(self.config.t2v_search_root).resolve()
        self.vbench_root = _resolve_vbench_root(self.t2v_search_root, self.config.vbench_root).resolve()
        self.pretrained_root = _resolve_pretrained_root(
            self.t2v_search_root,
            self.config.pretrained_root,
        ).resolve()
        self.device = self.config.device
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        _prepend_sys_path(self.t2v_search_root)
        if self.vbench_root.parent.name == "verifiers":
            _prepend_sys_path(self.vbench_root.parent.parent)
        _prepend_sys_path(self.vbench_root)

        import clip  # type: ignore
        from easydict import EasyDict as edict  # type: ignore
        from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer  # type: ignore
        from vbench.third_party.ViCLIP.viclip import ViCLIP  # type: ignore
        from verifiers.VBench.my_rewards.aesthetic import (  # type: ignore
            get_aesthetic_model,
            laion_aesthetic,
        )
        from verifiers.VBench.my_rewards.dino_similarity import subject_consistency  # type: ignore
        from verifiers.VBench.my_rewards.motion_prior import (  # type: ignore
            MotionSmoothness,
            motion_smoothness,
        )
        from verifiers.VBench.my_rewards.optical_flow import (  # type: ignore
            DynamicDegree,
            dynamic_degree,
        )
        from verifiers.VBench.my_rewards.viclip_similarity import overall_consistency  # type: ignore

        self._laion_aesthetic = laion_aesthetic
        self._subject_consistency = subject_consistency
        self._dynamic_degree = dynamic_degree
        self._overall_consistency = overall_consistency
        self._motion_smoothness = motion_smoothness

        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.aesthetic_model = get_aesthetic_model(
            str(self.pretrained_root / "aesthetic_model" / "emb_reader")
        ).to(self.device)

        self.dino_model = torch.hub.load(
            repo_or_dir="facebookresearch/dino:main",
            source="github",
            model="dino_vitb16",
            read_frame=False,
        ).to(self.device)

        raft_args = edict(
            {
                "model": str(self.pretrained_root / "raft_model" / "models" / "raft-things.pth"),
                "small": False,
                "mixed_precision": False,
                "alternate_corr": False,
            }
        )
        self.dynamic = DynamicDegree(raft_args, self.device)

        self.tokenizer = SimpleTokenizer(str(self.pretrained_root / "ViCLIP" / "bpe_simple_vocab_16e6.txt.gz"))
        self.viclip = ViCLIP(
            tokenizer=self.tokenizer,
            pretrain=str(self.pretrained_root / "ViCLIP" / "ViClip-InternVid-10M-FLT.pth"),
        ).to(self.device)

        self.motion = MotionSmoothness(
            str(self.vbench_root / "vbench" / "third_party" / "amt" / "cfgs" / "AMT-S.yaml"),
            str(self.pretrained_root / "amt_model" / "amt-s.pth"),
            self.device,
        )
        self._loaded = True

    @torch.no_grad()
    def __call__(
        self,
        video_tensor: torch.Tensor,
        prompt: str,
        *,
        image_reward: bool = False,
    ) -> tuple[float, dict[str, float]]:
        output = self.score(video_tensor, prompt, image_reward=image_reward)
        return output.reward, output.details

    @torch.no_grad()
    def score(
        self,
        video_tensor: torch.Tensor,
        prompt: str,
        *,
        image_reward: bool = False,
    ) -> VBenchRewardOutput:
        self._load()
        video_tensor = video_tensor.to(self.device)
        weights = self.config.weights

        aesthetic_score = float(
            self._laion_aesthetic(self.aesthetic_model, self.clip_model, video_tensor, self.device)
        )
        subject_score = (
            float(self._subject_consistency(self.dino_model, video_tensor, self.device))
            if not image_reward
            else 0.0
        )
        dynamic_score = (
            _dynamic_degree_mapping(
                float(self._dynamic_degree(self.dynamic, video_tensor.to(torch.float)))
            )
            if not image_reward
            else 0.0
        )
        overall_score = (
            float(self._overall_consistency(self.viclip, video_tensor, prompt, self.tokenizer, self.device))
            if not image_reward
            else 0.0
        )
        motion_score = (
            float(self._motion_smoothness(self.motion, video_tensor))
            if not image_reward
            else 0.0
        )

        reward = (
            weights.aesthetic * aesthetic_score
            + weights.subject_consistency * subject_score
            + weights.dynamic_degree * dynamic_score
            + weights.overall_consistency * overall_score
            + weights.motion_smoothness * motion_score
        )
        details = {
            "reward": float(reward),
            "subject_consistency": subject_score,
            "motion_smoothness": motion_score,
            "dynamic_degree": dynamic_score,
            "aesthetic": aesthetic_score,
            "overall_consistency": overall_score,
        }
        return VBenchRewardOutput(reward=float(reward), details=details)
