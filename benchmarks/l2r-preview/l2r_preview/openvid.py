"""Manifest-backed real-video training pairs in native Wan VAE space."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from .latents import WanLatentSpec


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(cache, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    records = [json.loads(line) for line in (Path(cache) / f"{split}.jsonl").read_text().splitlines()]
    if not records or any(r["split"] != split for r in records):
        raise ValueError("Empty or mixed split")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate clip IDs")
    return records


def validate_cache(cache):
    cache = Path(cache)
    report = json.loads((cache / "validation.json").read_text())
    if report["status"] != "complete" or sha256(cache / "manifest.jsonl") != report["manifest_sha256"]:
        raise ValueError("Incomplete or changed latent cache")
    splits = {s: load_records(cache, s) for s in ("train", "validation", "test")}
    records = [r for rows in splits.values() for r in rows]
    manifest = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines()]
    if sorted(records, key=lambda r: r["id"]) != sorted(manifest, key=lambda r: r["id"]):
        raise ValueError("Split manifests disagree with complete manifest")
    if {s: len(rows) for s, rows in splits.items()} != report["split_counts"]:
        raise ValueError("Split counts disagree")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Clip leaked across splits")
    sources = {s: {r["source_id"] for r in rows} for s, rows in splits.items()}
    if any(sources[a] & sources[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Source ID leaked across splits")
    return report


def patch_plan(shape, scale, rng, core=(8, 12), target_frames=2, context=7, halo=None):
    _, frames, height, width = shape
    ch, cw = min(core[0], height), min(core[1], width)
    count = min(target_frames, frames)
    start = rng.randrange(frames - count + 1)
    y, x = rng.randrange(height - ch + 1), rng.randrange(width - cw + 1)
    # 3 low-resolution convs + upsampling stencil; then 3 high-res convs
    # and a 9x9/3x3 RGB head. Halo includes the full spatial receptive field.
    halo = 4 + math.ceil(8 / scale) if halo is None else halo
    if context < 0 or halo < 0:
        raise ValueError("Context and halo must be nonnegative")
    y0, y1 = max(0, y - halo), min(height, y + ch + halo)
    x0, x1 = max(0, x - halo), min(width, x + cw + halo)
    t0 = max(0, start - context)
    return {
        "latent_slice": (t0, start + count, y0, y1, x0, x1),
        "roi": (start - t0, count, (y-y0)*scale, (x-x0)*scale, ch*scale, cw*scale),
        "rgb_crop": (y*scale, x*scale, ch*scale, cw*scale),
        "latent_start": start,
    }


def crop_prediction(prediction, roi):
    offset, count, y, x, h, w = roi
    return prediction[:, :, offset:offset+count, y:y+h, x:x+w]


def read_rgb(path, indices, expected_hw, crop=None):
    """Sequential decoding preserves exact frame IDs; no approximate seeking."""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    frames = []
    wanted = set(indices)
    if not cap.isOpened() or len(wanted) != len(indices) or sorted(indices) != list(indices):
        cap.release()
        raise ValueError(f"Invalid video or frame indices: {path}")
    try:
        for index in range(indices[-1] + 1):
            if not cap.grab():
                raise ValueError(f"Video ended at frame {index}: {path}")
            if index in wanted:
                ok, frame = cap.retrieve()
                if not ok or tuple(frame.shape[:2]) != tuple(expected_hw):
                    raise ValueError(f"Bad frame shape: {path}")
                if crop is not None:
                    y, x, h, w = crop
                    frame = frame[y:y+h, x:x+w]
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    rgb = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).contiguous()
    return rgb.float().div_(127.5).sub_(1)


class OpenVidPatches(Dataset):
    def __init__(self, cache, split, seed=7, core=(8, 12), target_frames=2, limit=None, context=7, halo=None):
        self.cache = Path(cache).resolve()
        self.config = json.loads((self.cache / "config.json").read_text())
        self.root = Path(self.config["dataset_root"])
        self.spec = WanLatentSpec.from_vae_config(self.cache / "vae_config.json")
        self.records = load_records(self.cache, split)
        if limit is not None:
            self.records = self.records[:limit]
        self.split, self.seed, self.core, self.target_frames = split, seed, core, target_frames
        self.context, self.halo = context, halo
        self.verified = set()
        if self.config["latent_space"] != "native_vae_unnormalized":
            raise ValueError("Training expects native unnormalized latents")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        record = self.records[index]
        latent_path, clip_path = self.cache / record["latent_path"], self.root / record["clip_path"]
        if record["id"] not in self.verified:
            if sha256(latent_path) != record["latent_sha256"] or sha256(clip_path) != record["clip_sha256"]:
                raise ValueError(f"Changed training pair: {record['id']}")
            self.verified.add(record["id"])
        z = load_file(latent_path)["latent"]
        if list(z.shape) != record["shape"] or z.dtype != torch.bfloat16 or not torch.isfinite(z).all():
            raise ValueError(f"Invalid latent: {record['id']}")
        if record["latent_space"] != "native_vae_unnormalized":
            raise ValueError("Mixed latent spaces")
        expected_indices = self.spec.rgb_frame_indices(z.shape[1]).tolist()
        if record["rgb_frame_indices"] != expected_indices:
            raise ValueError("Temporal alignment changed")
        # Validation/test crops stay fixed independently of training epoch/order.
        sample_epoch = epoch if self.split == "train" else 0
        token = f"{self.seed}:{self.split}:{sample_epoch}:{record['id']}"
        rng = random.Random(int(hashlib.sha256(token.encode()).hexdigest()[:16], 16))
        plan = patch_plan(z.shape, self.spec.spatial_scale, rng, self.core, self.target_frames,
                          self.context, self.halo)
        t0, t1, y0, y1, x0, x1 = plan["latent_slice"]
        z = z[:, t0:t1, y0:y1, x0:x1].float().contiguous()
        start, count = plan["latent_start"], plan["roi"][1]
        indices = expected_indices[start:start+count]
        rgb = read_rgb(clip_path, indices,
                       (record["shape"][2]*self.spec.spatial_scale, record["shape"][3]*self.spec.spatial_scale),
                       plan["rgb_crop"])
        return {"latent": z, "rgb": rgb, "roi": plan["roi"], "id": record["id"],
                "rgb_indices": indices, "plan": plan}
