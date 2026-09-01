from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
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


def _write_complete_preparation_bundle(
    runner,
    output_root,
    *,
    manifest_overrides: dict[str, object] | None = None,
    parent_meta_overrides: dict[str, object] | None = None,
) -> None:
    parent_noise = torch.zeros((1,), dtype=torch.float32)
    runner._atomic_save_tensor(output_root / "parent_noise.pt", parent_noise)
    manifest = runner._prepare_neighbors(output_root, parent_noise)
    if manifest_overrides:
        manifest.update(manifest_overrides)
        runner._atomic_write_json(output_root / "manifest.json", manifest)

    control_dir = output_root / "parent_control"
    runner._atomic_write_bytes(control_dir / "video.mp4", b"video")
    runner._atomic_write_bytes(control_dir / "all_frames.jpg", b"sheet")
    parent_meta = {
        "kind": "parent_control",
        "seed": runner.PARENT_SEED,
        "prompt": runner.PROMPT,
        "input_path": str(runner.INPUT),
        "input_sha256": runner.INPUT_SHA256,
        "model_path": str(runner.MODEL),
        "scheduler_class": "UniPCMultistepScheduler",
        "height": runner.HEIGHT,
        "width": runner.WIDTH,
        "num_frames": runner.NUM_FRAMES,
        "num_inference_steps": runner.STEPS,
        "guidance_scale": runner.GUIDANCE_SCALE,
        "fps": runner.FPS,
        "elapsed_seconds": 0.0,
        "peak_gpu_memory_mb": 0.0,
    }
    if parent_meta_overrides:
        parent_meta.update(parent_meta_overrides)
    runner._atomic_write_json(control_dir / "meta.json", parent_meta)
    runner._atomic_touch(control_dir / "DONE")
    runner._atomic_touch(runner._prepare_done_path(output_root))


@pytest.mark.parametrize(
    ("manifest_overrides", "parent_meta_overrides", "expected"),
    [
        ({"prompt": "stale prompt"}, None, r"manifest\.prompt"),
        (None, {"seed": -1}, r"parent_control\.seed"),
    ],
)
def test_complete_bundle_rejects_stale_fixed_settings(
    tmp_path,
    monkeypatch,
    manifest_overrides,
    parent_meta_overrides,
    expected,
) -> None:
    runner = load_runner()
    _write_complete_preparation_bundle(
        runner,
        tmp_path,
        manifest_overrides=manifest_overrides,
        parent_meta_overrides=parent_meta_overrides,
    )
    monkeypatch.setattr(runner, "_load_image", lambda path: pytest.fail("stale bundle should not rebuild"))
    monkeypatch.setattr(runner, "load_pipeline", lambda: pytest.fail("stale bundle should not rebuild"))

    with pytest.raises(RuntimeError, match=expected):
        runner._ensure_prepared(tmp_path, auto_prepare=True)


def test_auto_prepare_recovers_from_interrupted_preparation(tmp_path, monkeypatch) -> None:
    runner = load_runner()
    runner._atomic_save_tensor(tmp_path / "parent_noise.pt", torch.zeros((1,), dtype=torch.float32))
    calls: list[str] = []
    new_parent_noise = torch.ones((1,), dtype=torch.float32)
    original_prepare_neighbors = runner._prepare_neighbors

    def fake_load_image(path):
        calls.append("load_image")
        return Image.new("RGB", (4, 4))

    def fake_load_pipeline():
        calls.append("load_pipeline")
        return object()

    def fake_capture_parent_noise(pipe, image, output_root):
        calls.append("capture_parent_noise")
        runner._atomic_save_tensor(output_root / "parent_noise.pt", new_parent_noise)
        control_dir = output_root / "parent_control"
        runner._atomic_write_bytes(control_dir / "video.mp4", b"video")
        runner._atomic_write_bytes(control_dir / "all_frames.jpg", b"sheet")
        runner._atomic_write_json(
            control_dir / "meta.json",
            {
                "kind": "parent_control",
                "seed": runner.PARENT_SEED,
                "prompt": runner.PROMPT,
                "input_path": str(runner.INPUT),
                "input_sha256": runner.INPUT_SHA256,
                "model_path": str(runner.MODEL),
                "scheduler_class": "UniPCMultistepScheduler",
                "height": runner.HEIGHT,
                "width": runner.WIDTH,
                "num_frames": runner.NUM_FRAMES,
                "num_inference_steps": runner.STEPS,
                "guidance_scale": runner.GUIDANCE_SCALE,
                "fps": runner.FPS,
                "elapsed_seconds": 0.0,
                "peak_gpu_memory_mb": 0.0,
            },
        )
        runner._atomic_touch(control_dir / "DONE")
        return new_parent_noise, np.zeros((1, 1, 1, 3), dtype=np.uint8), {"fps": runner.FPS}

    def wrapped_prepare_neighbors(output_root, parent_noise):
        calls.append("prepare_neighbors")
        assert not runner._prepare_done_path(output_root).exists()
        return original_prepare_neighbors(output_root, parent_noise)

    monkeypatch.setattr(runner, "_load_image", fake_load_image)
    monkeypatch.setattr(runner, "load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(runner, "_capture_parent_noise", fake_capture_parent_noise)
    monkeypatch.setattr(runner, "_prepare_neighbors", wrapped_prepare_neighbors)

    manifest = runner._ensure_prepared(tmp_path, auto_prepare=True)

    assert calls == ["load_image", "load_pipeline", "capture_parent_noise", "prepare_neighbors"]
    torch.testing.assert_close(torch.load(tmp_path / "parent_noise.pt"), new_parent_noise)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "parent_control" / "DONE").exists()
    assert runner._prepare_done_path(tmp_path).exists()
    assert len(manifest["neighbors"]) == len(runner.neighbor_specs())


def test_prepare_requires_complete_bundle_and_done_marker(tmp_path) -> None:
    runner = load_runner()
    noise_path = tmp_path / "noise" / "n00_a002.pt"
    runner._atomic_save_tensor(tmp_path / "parent_noise.pt", torch.zeros((1,), dtype=torch.float32))
    runner._atomic_save_tensor(noise_path, torch.zeros((1,), dtype=torch.float32))
    runner._atomic_write_json(
        tmp_path / "manifest.json",
        {
            "neighbors": [
                {
                    "index": 0,
                    "alpha": 0.02,
                    "perturb_seed": 10_000,
                    "sample_id": "n00_a002",
                    "noise_path": str(noise_path.relative_to(tmp_path)),
                    "metrics": {"rms_distance": 1.0, "cosine_similarity": 0.0, "norm_ratio": 1.0},
                }
            ]
        },
    )

    with pytest.raises(RuntimeError):
        runner._ensure_prepared(tmp_path, auto_prepare=False)


def test_metadata_records_fixed_fps_everywhere(tmp_path, monkeypatch) -> None:
    runner = load_runner()

    class FakePipe:
        def __init__(self) -> None:
            self.scheduler = type("Scheduler", (), {})()

        def prepare_latents(self, *args, **kwargs):
            return (torch.zeros((1,), dtype=torch.float32),)

    frames = [np.full((4, 4, 3), 0.5, dtype=np.float32) for _ in range(3)]

    def fake_run_pipeline(pipe, image, *, seed=None, latents=None):
        if latents is None:
            pipe.prepare_latents()
        return frames

    monkeypatch.setattr(runner, "run_pipeline", fake_run_pipeline)

    pipe = FakePipe()
    parent_latents, _, parent_meta = runner._capture_parent_noise(pipe, Image.new("RGB", (4, 4)), tmp_path)
    assert parent_meta["fps"] == runner.FPS

    manifest = runner._prepare_neighbors(tmp_path, parent_latents)
    assert manifest["fps"] == runner.FPS

    entry = manifest["neighbors"][0]
    monkeypatch.setattr(runner, "_pipe_device", lambda pipe: torch.device("cpu"))
    sample_meta = runner._generate_sample(pipe, Image.new("RGB", (4, 4)), tmp_path, entry)
    assert sample_meta["fps"] == runner.FPS
