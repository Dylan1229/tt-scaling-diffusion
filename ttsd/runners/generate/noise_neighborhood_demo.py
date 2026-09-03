from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL = Path(
    "/data/datasets/fanjiang/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/"
    "b8fff7315c768468a5333511427288870b2e9635"
)
INPUT = Path("runs/toy_red_ball_i2v_v2/input.png")
INPUT_SHA256 = "b1aad4e150009199e5a59c2f7867e32d9ae229d5f923dc53ffc857f14f95a8c9"
OUTPUT_ROOT = Path("runs/toy_red_ball_i2v_v2/noise_neighborhood_v1")
PROMPT = (
    "Static camera. A red ball moves in a straight horizontal line from left to right, "
    "enters through the open left side of a stationary blue box, and stops inside. "
    "The ball stays at the same height. The box does not move or change shape."
)
ALPHAS = (0.02, 0.05, 0.10, 0.20)
NEIGHBORS_PER_ALPHA = 8
PARENT_SEED = 0
HEIGHT, WIDTH, NUM_FRAMES, STEPS = 480, 832, 81, 50
GUIDANCE_SCALE, FPS = 5.0, 24
SCHEDULER_CLASS = "UniPCMultistepScheduler"


def neighbor_specs() -> list[dict[str, int | float | str]]:
    return [
        {
            "index": index,
            "alpha": alpha,
            "perturb_seed": 10_000 + index,
            "sample_id": f"n{index:02d}_a{round(alpha * 100):03d}",
        }
        for alpha_index, alpha in enumerate(ALPHAS)
        for local_index in range(NEIGHBORS_PER_ALPHA)
        for index in [alpha_index * NEIGHBORS_PER_ALPHA + local_index]
    ]


def specs_for_shard(shard_index: int, num_shards: int) -> list[dict[str, int | float | str]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    return [spec for spec in neighbor_specs() if int(spec["index"]) % num_shards == shard_index]


def make_neighbor(parent: torch.Tensor, alpha: float, perturb_seed: int) -> torch.Tensor:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must satisfy 0 <= alpha <= 1")
    parent = parent.detach().to(device="cpu", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(int(perturb_seed))
    epsilon = torch.randn(parent.shape, generator=generator, dtype=torch.float32)
    return math.sqrt(1.0 - alpha**2) * parent + alpha * epsilon


def noise_metrics(parent: torch.Tensor, neighbor: torch.Tensor) -> dict[str, float]:
    parent = parent.float().flatten()
    neighbor = neighbor.float().flatten()
    delta = neighbor - parent
    return {
        "rms_distance": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(parent, neighbor, dim=0)),
        "norm_ratio": float(neighbor.norm() / parent.norm()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temp_path(path: Path, suffix: str = ".tmp") -> Path:
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=suffix)
    os.close(fd)
    return Path(name)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path, suffix=".pt")
    try:
        torch.save(tensor, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_touch(path: Path) -> None:
    _atomic_write_bytes(path, b"")


def _frame_to_array(frame) -> np.ndarray:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    elif torch.is_tensor(frame):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB frame, got shape {arr.shape}")
    return arr


def _frames_to_array(frames) -> np.ndarray:
    if torch.is_tensor(frames):
        arr = frames.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(0, 2, 3, 1)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"expected video tensor with shape (T,H,W,3) or (T,3,H,W), got {arr.shape}")
        return arr
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(0, 2, 3, 1)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"expected video array with shape (T,H,W,3) or (T,3,H,W), got {arr.shape}")
        return arr
    return np.stack([_frame_to_array(frame) for frame in frames], axis=0)


def _frames_to_uint8_array(frames) -> np.ndarray:
    arr = _frames_to_array(frames)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and np.nanmax(arr) <= 1.0 + 1e-6 and np.nanmin(arr) >= -1e-6:
        arr = (arr.clip(0.0, 1.0) * 255.0).round()
    else:
        arr = arr.clip(0.0, 255.0).round()
    return arr.astype(np.uint8)


def _save_video_atomic(frames, path: Path, fps: int = FPS) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path, suffix=".mp4")
    try:
        iio.imwrite(tmp, _frames_to_uint8_array(frames), fps=fps, codec="libx264")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def save_contact_sheet(
    frames,
    path: Path,
    columns: int = 9,
    thumb_size: tuple[int, int] = (208, 120),
) -> None:
    if columns < 1:
        raise ValueError("columns must be positive")
    images = _frames_to_uint8_array(frames)
    if images.shape[0] == 0:
        raise ValueError("cannot build a contact sheet from zero frames")
    rows = math.ceil(images.shape[0] / columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), color=(16, 16, 16))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(images):
        row, col = divmod(index, columns)
        tile = Image.fromarray(frame, mode="RGB")
        tile = ImageOps.fit(tile, thumb_size, method=Image.Resampling.LANCZOS)
        x = col * thumb_size[0]
        y = row * thumb_size[1]
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + 4), str(index), fill=(255, 255, 255), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path, suffix=".jpg")
    try:
        sheet.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_pipeline():
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(
        MODEL, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    if type(pipe.scheduler).__name__ != SCHEDULER_CLASS:
        raise RuntimeError(f"expected {SCHEDULER_CLASS}, got {type(pipe.scheduler).__name__}")
    return pipe


def run_pipeline(
    pipe,
    image: Image.Image,
    *,
    seed: int | None = None,
    latents: torch.Tensor | None = None,
):
    kwargs = {
        "image": image,
        "prompt": PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }
    if latents is None:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(seed))
    else:
        kwargs["latents"] = latents.clone()
    return pipe(**kwargs).frames[0]


def _pipe_device(pipe) -> torch.device:
    device = getattr(pipe, "_execution_device", None)
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"missing input image: {path}")
    if _sha256(path) != INPUT_SHA256:
        raise RuntimeError(f"input checksum mismatch for {path}")
    return Image.open(path).convert("RGB")


def _complete_sample_dir(sample_dir: Path) -> bool:
    return all((sample_dir / name).exists() for name in ("video.mp4", "all_frames.jpg", "meta.json", "DONE"))


def _prepare_done_path(output_root: Path) -> Path:
    return output_root / "prepare" / "DONE"


def _expected_prepare_fields() -> dict[str, object]:
    return {
        "prompt": PROMPT,
        "input_path": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "model_path": str(MODEL),
        "scheduler_class": SCHEDULER_CLASS,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "fps": FPS,
    }


def _require_expected_fields(source: str, payload: dict[str, object], expected_fields: dict[str, object]) -> None:
    for key, expected in expected_fields.items():
        if key not in payload:
            raise RuntimeError(f"invalid preparation bundle: missing {source}.{key}")
        actual = payload[key]
        if actual != expected:
            raise RuntimeError(f"stale preparation bundle: {source}.{key}={actual!r} != {expected!r}")


def _capture_parent_noise(pipe, image: Image.Image, output_root: Path) -> tuple[torch.Tensor, np.ndarray, dict[str, object]]:
    captured: dict[str, torch.Tensor] = {}
    original = pipe.prepare_latents

    def wrapped_prepare_latents(*args, **kwargs):
        outputs = original(*args, **kwargs)
        latent = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        captured["latents"] = latent.detach().clone().to(device="cpu", dtype=torch.float32)
        return outputs

    pipe.prepare_latents = wrapped_prepare_latents
    started = time.perf_counter()
    try:
        parent_frames = run_pipeline(pipe, image, seed=PARENT_SEED)
    finally:
        pipe.prepare_latents = original
    elapsed = time.perf_counter() - started
    latents = captured.get("latents")
    if latents is None:
        raise RuntimeError("failed to capture parent latents")
    _atomic_save_tensor(output_root / "parent_noise.pt", latents)
    explicit_frames = run_pipeline(pipe, image, latents=latents)
    parent_array = _frames_to_array(parent_frames)
    explicit_array = _frames_to_array(explicit_frames)
    if not np.allclose(parent_array, explicit_array, atol=1e-5, rtol=0):
        raise RuntimeError("explicit parent reinjection diverged from the captured run")
    control_dir = output_root / "parent_control"
    _save_video_atomic(explicit_frames, control_dir / "video.mp4", fps=FPS)
    save_contact_sheet(explicit_frames, control_dir / "all_frames.jpg")
    meta = {
        "kind": "parent_control",
        "seed": PARENT_SEED,
        "prompt": PROMPT,
        "input_path": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "model_path": str(MODEL),
        "scheduler_class": SCHEDULER_CLASS,
        "diffusers_version": metadata.version("diffusers"),
        "torch_version": torch.__version__,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "fps": FPS,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
    }
    _atomic_write_json(control_dir / "meta.json", meta)
    _atomic_touch(control_dir / "DONE")
    return latents, explicit_array, meta


def _prepare_neighbors(output_root: Path, parent_noise: torch.Tensor) -> dict[str, object]:
    noise_dir = output_root / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    neighbors = []
    for spec in neighbor_specs():
        noise = make_neighbor(parent_noise, float(spec["alpha"]), int(spec["perturb_seed"]))
        noise_path = noise_dir / f"{spec['sample_id']}.pt"
        _atomic_save_tensor(noise_path, noise)
        metrics = noise_metrics(parent_noise, noise)
        neighbors.append({**spec, "noise_path": str(noise_path.relative_to(output_root)), "metrics": metrics})
    manifest = {
        "prompt": PROMPT,
        "input_path": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "model_path": str(MODEL),
        "scheduler_class": SCHEDULER_CLASS,
        "diffusers_version": metadata.version("diffusers"),
        "torch_version": torch.__version__,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "fps": FPS,
        "parent_noise_path": "parent_noise.pt",
        "neighbors": neighbors,
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def _validate_preparation(output_root: Path) -> dict[str, object]:
    manifest_path = output_root / "manifest.json"
    done_path = _prepare_done_path(output_root)
    control_dir = output_root / "parent_control"
    required = [
        output_root / "parent_noise.pt",
        manifest_path,
        control_dir / "video.mp4",
        control_dir / "all_frames.jpg",
        control_dir / "meta.json",
        control_dir / "DONE",
        done_path,
    ]
    missing = [str(path.relative_to(output_root)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"incomplete preparation bundle: {missing}")
    manifest = json.loads(manifest_path.read_text())
    _require_expected_fields("manifest", manifest, _expected_prepare_fields())
    parent_control_meta = json.loads((control_dir / "meta.json").read_text())
    _require_expected_fields(
        "parent_control",
        parent_control_meta,
        {**_expected_prepare_fields(), "kind": "parent_control", "seed": PARENT_SEED},
    )
    neighbors = manifest.get("neighbors")
    if not isinstance(neighbors, list) or len(neighbors) != len(neighbor_specs()):
        raise RuntimeError("incomplete preparation manifest")
    expected = {spec["sample_id"]: spec for spec in neighbor_specs()}
    seen: set[str] = set()
    for entry in neighbors:
        sample_id = entry.get("sample_id")
        if sample_id not in expected:
            raise RuntimeError(f"unexpected manifest sample: {sample_id}")
        if sample_id in seen:
            raise RuntimeError(f"duplicate manifest sample: {sample_id}")
        seen.add(sample_id)
        for key in ("index", "alpha", "perturb_seed", "noise_path", "metrics"):
            if key not in entry:
                raise RuntimeError(f"manifest entry missing {key}: {sample_id}")
        if not (output_root / str(entry["noise_path"])).exists():
            raise RuntimeError(f"missing noise tensor for {sample_id}")
    if len(seen) != len(expected):
        raise RuntimeError("incomplete preparation manifest")
    if int(manifest.get("fps", FPS)) != FPS:
        raise RuntimeError("incomplete preparation manifest: fps mismatch")
    return manifest


def _ensure_prepared(output_root: Path, auto_prepare: bool) -> dict[str, object]:
    if _prepare_done_path(output_root).exists():
        return _validate_preparation(output_root)
    if not auto_prepare:
        raise RuntimeError("preparation must finish before generation")
    image = _load_image(INPUT)
    pipe = load_pipeline()
    parent_noise, _, _ = _capture_parent_noise(pipe, image, output_root)
    manifest = _prepare_neighbors(output_root, parent_noise)
    _atomic_touch(_prepare_done_path(output_root))
    return manifest


def _selected_specs(manifest: dict[str, object], shard_index: int, num_shards: int, indices: list[int] | None) -> list[dict[str, object]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    raw_neighbors = manifest["neighbors"]
    if not isinstance(raw_neighbors, list):
        raise RuntimeError("invalid manifest neighbors")
    selected = [entry for entry in raw_neighbors if int(entry["index"]) % num_shards == shard_index]
    if indices:
        index_set = set(indices)
        invalid = [index for index in sorted(index_set) if index < 0 or index >= len(neighbor_specs())]
        if invalid:
            raise ValueError(f"indices out of range: {invalid}")
        selected = [entry for entry in selected if int(entry["index"]) in index_set]
    return sorted(selected, key=lambda entry: int(entry["index"]))


def _sample_complete(sample_dir: Path) -> bool:
    return _complete_sample_dir(sample_dir)


def _generate_sample(pipe, image: Image.Image, output_root: Path, entry: dict[str, object]) -> dict[str, object]:
    sample_id = str(entry["sample_id"])
    sample_dir = output_root / "neighbors" / sample_id
    if _sample_complete(sample_dir):
        return json.loads((sample_dir / "meta.json").read_text())
    latents = torch.load(output_root / str(entry["noise_path"]), map_location="cpu")
    started = time.perf_counter()
    frames = run_pipeline(pipe, image, latents=latents.to(device=_pipe_device(pipe), dtype=torch.float32))
    elapsed = time.perf_counter() - started
    video_path = sample_dir / "video.mp4"
    sheet_path = sample_dir / "all_frames.jpg"
    meta_path = sample_dir / "meta.json"
    _save_video_atomic(frames, video_path, fps=FPS)
    save_contact_sheet(frames, sheet_path)
    meta = {
        **{k: entry[k] for k in ("index", "alpha", "perturb_seed", "sample_id")},
        "sample_spec": {
            "index": entry["index"],
            "alpha": entry["alpha"],
            "perturb_seed": entry["perturb_seed"],
            "sample_id": entry["sample_id"],
        },
        "noise_path": str(entry["noise_path"]),
        "noise_metrics": entry["metrics"],
        "prompt": PROMPT,
        "input_path": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "model_path": str(MODEL),
        "scheduler_class": type(pipe.scheduler).__name__,
        "diffusers_version": metadata.version("diffusers"),
        "torch_version": torch.__version__,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "fps": FPS,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
    }
    _atomic_write_json(meta_path, meta)
    _atomic_touch(sample_dir / "DONE")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = args.output_root
    if args.prepare_only:
        _ensure_prepared(output_root, auto_prepare=True)
        return
    manifest = _ensure_prepared(output_root, auto_prepare=args.num_shards == 1)
    image = _load_image(INPUT)
    pipe = load_pipeline()
    selected = _selected_specs(
        manifest,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        indices=args.indices,
    )
    for entry in selected:
        meta = _generate_sample(pipe, image, output_root, entry)
        print(json.dumps(meta, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
