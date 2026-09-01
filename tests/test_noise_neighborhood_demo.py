from __future__ import annotations

import importlib
import math

import numpy as np
import torch
from PIL import Image


def load_runner():
    return importlib.import_module("ttsd.runners.generate.noise_neighborhood_demo")


def test_neighbor_construction_is_exact_and_reproducible() -> None:
    runner = load_runner()
    parent = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(1, 2, 3, 2, 2)

    torch.testing.assert_close(runner.make_neighbor(parent, 0.0, 7), parent, rtol=0, atol=0)
    first = runner.make_neighbor(parent, 0.1, 7)
    second = runner.make_neighbor(parent, 0.1, 7)
    other = runner.make_neighbor(parent, 0.1, 8)

    generator = torch.Generator(device="cpu").manual_seed(7)
    epsilon = torch.randn(parent.shape, generator=generator, dtype=torch.float32)
    expected = math.sqrt(1.0 - 0.1**2) * parent + 0.1 * epsilon

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other)
    assert first.shape == parent.shape
    assert first.dtype == torch.float32

    metrics = runner.noise_metrics(parent, first)
    assert set(metrics) == {"rms_distance", "cosine_similarity", "norm_ratio"}
    assert metrics["rms_distance"] > 0
    assert -1 <= metrics["cosine_similarity"] <= 1
    assert metrics["norm_ratio"] > 0


def test_specs_are_32_unique_reproducible_neighbors_partitioned_once() -> None:
    runner = load_runner()
    specs = runner.neighbor_specs()
    assert runner.ALPHAS == (0.02, 0.05, 0.10, 0.20)
    assert len(specs) == 32
    assert len({spec["sample_id"] for spec in specs}) == 32
    assert [sum(spec["alpha"] == alpha for spec in specs) for alpha in runner.ALPHAS] == [8, 8, 8, 8]

    shards = [runner.specs_for_shard(i, 4) for i in range(4)]
    assert all(len(shard) == 8 for shard in shards)
    assert sorted(spec["index"] for shard in shards for spec in shard) == list(range(32))


def test_contact_sheet_contains_every_frame(tmp_path) -> None:
    runner = load_runner()
    frames = [np.full((4, 6, 3), value / 4, dtype=np.float32) for value in range(5)]
    output = tmp_path / "sheet.jpg"

    runner.save_contact_sheet(frames, output, columns=3, thumb_size=(6, 4))

    with Image.open(output) as sheet:
        assert sheet.size == (18, 8)
