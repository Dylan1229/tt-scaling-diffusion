import json
import random

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from l2r_preview.model import L2RDecoder
from l2r_preview.openvid import OpenVidPatches, crop_prediction, patch_plan, read_rgb, sha256, validate_cache


def test_context_crop_matches_full_decoder_even_at_video_edges():
    torch.manual_seed(4)
    model = L2RDecoder(2, 2).eval()
    with torch.no_grad():
        model.upsample.stencil.conv.weight.add_(0.03*torch.randn_like(model.upsample.stencil.conv.weight))
        z = torch.randn(1, 2, 12, 22, 26)
        full = model(z)
        for seed in (1, 3, 8, 13):
            plan = patch_plan(z.shape[1:], 2, random.Random(seed), core=(4, 5))
            t0, t1, y0, y1, x0, x1 = plan["latent_slice"]
            cropped = crop_prediction(model(z[:, :, t0:t1, y0:y1, x0:x1]), plan["roi"])
            y, x, h, w = plan["rgb_crop"]
            start = plan["latent_start"]
            expected = full[:, :, start:start+2, y:y+h, x:x+w]
            torch.testing.assert_close(cropped, expected, rtol=1e-5, atol=1e-6)


@pytest.fixture
def dataset_cache(tmp_path):
    root = tmp_path/"dataset"
    cache = root/"cache"
    (root/"clips").mkdir(parents=True)
    (cache/"latents").mkdir(parents=True)
    config = {"dataset_root": str(root), "latent_space": "native_vae_unnormalized"}
    (cache/"config.json").write_text(json.dumps(config))
    (cache/"vae_config.json").write_text(json.dumps({
        "z_dim": 2, "scale_factor_spatial": 4, "scale_factor_temporal": 2,
        "latents_mean": [5, 6], "latents_std": [2, 3],
    }))
    records = []
    for i, split in enumerate(("train", "validation", "test")):
        clip = root/"clips"/f"{split}.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 16, (48, 40))
        for t in range(9):
            frame = np.zeros((40, 48, 3), dtype=np.uint8)
            frame[:, :, 0] = np.arange(48)[None, :]*3
            frame[:, :, 1] = t*20
            frame[:, :, 2] = np.arange(40)[:, None]*4
            writer.write(frame)
        writer.release()
        latent = cache/"latents"/f"{split}.safetensors"
        save_file({"latent": torch.arange(2*5*10*12).reshape(2, 5, 10, 12).to(torch.bfloat16)}, latent)
        r = {"id": split, "source_id": f"source-{i}", "split": split,
             "clip_path": f"clips/{split}.mp4", "clip_sha256": sha256(clip),
             "latent_path": f"latents/{split}.safetensors", "latent_sha256": sha256(latent),
             "shape": [2, 5, 10, 12], "latent_space": "native_vae_unnormalized",
             "rgb_frame_indices": [0, 2, 4, 6, 8]}
        (cache/f"{split}.jsonl").write_text(json.dumps(r)+"\n")
        records.append(r)
    (cache/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (cache/"validation.json").write_text(json.dumps({
        "status": "complete", "manifest_sha256": sha256(cache/"manifest.jsonl"),
        "split_counts": {"train": 1, "validation": 1, "test": 1},
    }))
    return cache, root


def test_pair_alignment_native_units_and_deterministic_validation(dataset_cache):
    cache, root = dataset_cache
    validate_cache(cache)
    dataset = OpenVidPatches(cache, "validation", core=(3, 4))
    sample = dataset[(0, 0)]
    other_epoch = dataset[(99, 0)]
    assert sample["plan"] == other_epoch["plan"]
    z = load_file(cache/"latents/validation.safetensors")["latent"].float()
    t0, t1, y0, y1, x0, x1 = sample["plan"]["latent_slice"]
    assert torch.equal(sample["latent"], z[:, t0:t1, y0:y1, x0:x1])
    full_rgb = read_rgb(root/"clips/validation.mp4", list(range(9)), (40, 48))
    y, x, h, w = sample["plan"]["rgb_crop"]
    assert torch.equal(sample["rgb"], full_rgb[:, sample["rgb_indices"], y:y+h, x:x+w])
    assert not sample["rgb"].isnan().any()
    assert sample["rgb"].min() >= -1 and sample["rgb"].max() <= 1


def test_rejects_changed_split_or_pair(dataset_cache):
    cache, root = dataset_cache
    split = cache/"train.jsonl"
    split.write_text(split.read_text().replace('"source-0"', '"source-1"'))
    with pytest.raises(ValueError, match="disagree"):
        validate_cache(cache)
    dataset = OpenVidPatches(cache, "validation")
    (root/"clips/validation.mp4").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Changed training pair"):
        dataset[0]
