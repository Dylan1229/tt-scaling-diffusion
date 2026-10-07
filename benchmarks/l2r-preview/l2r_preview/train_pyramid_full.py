"""Bounded, resumable OpenVid-5K training for experimental pyramid decoders."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .io import load_checkpoint, save_checkpoint
from .openvid import OpenVidPatches, crop_prediction, sha256, validate_cache
from .pyramid import PyramidPreviewDecoder


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def resource_guard(gpu_uuid, output):
    if shutil.disk_usage(output).free < 50 * 1024**3:
        raise RuntimeError("Less than 50 GiB free disk; stopping safely")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[0] == gpu_uuid and int(fields[1]) != os.getpid():
            raise RuntimeError(f"Another process appeared on selected GPU: {line}")
    return {
        "free_disk_bytes": shutil.disk_usage(output).free,
        "shm_used_bytes": shutil.disk_usage("/dev/shm").used,
        "cpu_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(dataset, keys, workers):
    kwargs = {"prefetch_factor": 2} if workers else {}
    return DataLoader(dataset, sampler=keys, batch_size=None, num_workers=workers,
                      pin_memory=True, worker_init_fn=worker_init, **kwargs)


def predict_patch(model, sample, device):
    z = sample["latent"].unsqueeze(0).to(device, non_blocking=True)
    target = sample["rgb"].unsqueeze(0).to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        offset, count, y, x, h, w = sample["roi"]
        prediction = crop_prediction(model(z, output_start=offset, output_frames=count),
                                     (0, count, y, x, h, w))
    if prediction.shape != target.shape:
        raise RuntimeError(f"Training pair shape mismatch: {prediction.shape}, {target.shape}")
    return prediction.float(), target


def learning_rate(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    fraction = min(1.0, (step-warmup) / max(1, total-warmup-1))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi*fraction)))


@torch.inference_mode()
def evaluate(model, samples, device, guard=None):
    model.eval()
    total, clipped, count = 0.0, 0.0, 0
    began = time.monotonic()
    for index, sample in enumerate(samples):
        prediction, target = predict_patch(model, sample, device)
        mse = F.mse_loss(prediction, target)
        if not torch.isfinite(mse):
            raise RuntimeError("Non-finite validation prediction")
        total += float(mse)
        clipped += float(F.mse_loss(prediction.clamp(-1, 1), target))
        count += 1
        if guard is not None and index % 25 == 0:
            guard()
    if not count:
        raise RuntimeError("No evaluation examples")
    return {"mse": total/count, "clipped_mse": clipped/count,
            "psnr_db": 10*math.log10(4/max(clipped/count, 1e-12)), "clips": count,
            "seconds": time.monotonic()-began,
            "scope": "one fixed spatial crop and two aligned frames per clip; not full-video quality"}


def snapshot(model, spec, optimizer, config_hash, step, best, training):
    return {
        "format_version": 1, "architecture": model.architecture,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "latent_spec": spec.to_dict(), "training": training,
        "optimizer": optimizer.state_dict(), "step": step,
        "best_validation_mse": best, "config_sha256": config_hash,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "train.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal the explicitly selected GPU UUID")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.35)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required")
    report = validate_cache(args.cache)
    resource_guard(args.gpu_uuid, args.output)
    model = PyramidPreviewDecoder.from_variant(args.variant)
    datasets = {s: OpenVidPatches(args.cache, s, args.seed, tuple(args.core) if s == "train" else (8, 12),
                                 args.target_frames if s == "train" else 2,
                                 context=model.temporal_context, halo=model.spatial_halo)
                for s in ("train", "validation", "test")}
    train_data, spec = datasets["train"], datasets["train"].spec
    if any(ds.spec != spec for ds in datasets.values()):
        raise ValueError("Inconsistent normalization")
    if args.train_limit:
        train_data.records = train_data.records[:args.train_limit]
    if args.validation_limit:
        datasets["validation"].records = datasets["validation"].records[:args.validation_limit]
    total = args.steps
    source_files = ("model.py", "pyramid.py", "latents.py", "io.py", "openvid.py", "train_pyramid_full.py")
    code = {name: sha256(Path(__file__).parent/name) for name in source_files}
    config = {
        "schema": 1, "dataset_cache": str(args.cache.resolve()), "dataset": report,
        "split_hashes": {s: sha256(args.cache/f"{s}.jsonl") for s in datasets},
        "source_sha256": code, "seed": args.seed, "steps": total,
        "train_clips": len(train_data), "validation_clips": len(datasets["validation"]),
        "test_clips": len(datasets["test"]), "core_latent_hw": args.core, "validation_core_latent_hw": [8, 12],
        "validation_target_frames": 2,
        "target_frames": args.target_frames, "temporal_context": model.temporal_context,
        "spatial_halo": model.spatial_halo, "architecture": model.architecture,
        "variant": args.variant, "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "init_checkpoint_sha256": sha256(args.init_checkpoint) if args.init_checkpoint else None,
        "latent_space": "native_vae_unnormalized", "target": "original source RGB at frames 0,4,...,80",
        "optimizer": "AdamW", "loss": "unclipped RGB MSE in [-1,1] units; detail mode adds 0.05 times mean x/y RGB-gradient L1",
        "learning_rate": args.learning_rate, "weight_decay": 1e-4,
        "warmup": args.warmup, "schedule": "linear warmup, cosine decay to 0.1 * peak",
        "precision": "float32 parameters / BF16 autocast", "gradient_clip": 1.0,
        "gpu_uuid": args.gpu_uuid, "cuda_memory_fraction": 0.35,
        "workers": args.workers, "validation_every": args.validate_every,
        "checkpoint_every": args.checkpoint_every, "torch": str(torch.__version__),
        "validation_scope": "250 fixed aligned spatial/temporal patches; full videos require separate preview review",
        "initialization": "fresh random model" if not args.init_checkpoint else "warm start; optimizer reset",
        "loss_mode": args.loss_mode, "detail_weight": 0.05 if args.loss_mode == "detail" else 0.0,
        "data_epoch_offset": args.data_epoch_offset,
    }
    config = json.loads(json.dumps(config))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = args.output/"config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Existing run needs --resume and exactly matching configuration/source")
    else:
        if args.resume:
            raise ValueError("No run to resume")
        atomic_json(config_path, config)
        sources = args.output/"source"
        sources.mkdir(exist_ok=True)
        for name in source_files:
            shutil.copy2(Path(__file__).parent/name, sources/name)
    if args.init_checkpoint:
        initial, initial_spec, _ = load_checkpoint(args.init_checkpoint)
        if initial.architecture != model.architecture or initial_spec != spec:
            raise ValueError("Initialization checkpoint architecture/spec mismatch")
        model.load_state_dict(initial.state_dict())
    model = model.to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True)
    step, best, last_metrics = 0, None, None
    if args.resume:
        state = torch.load(args.output/"last.pt", weights_only=True, map_location="cpu")
        if state["config_sha256"] != config_hash:
            raise ValueError("Resume checkpoint does not match run configuration")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = state["step"], state["best_validation_mse"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    started = time.monotonic()
    start_step = step
    last_guard = 0.0
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda number, frame: stop.append(f"signal {number}"))
    log = (args.output/"metrics.jsonl").open("a", buffering=1)
    previous_seconds = 0.0
    progress_path = args.output/"progress.json"
    if args.resume and progress_path.exists():
        previous_seconds = json.loads(progress_path.read_text()).get("total_elapsed_seconds", 0)

    def guard(force=False):
        nonlocal last_guard
        if force or time.monotonic()-last_guard > 60:
            resource_guard(args.gpu_uuid, args.output)
            last_guard = time.monotonic()

    def metadata(status, **extra):
        elapsed = time.monotonic()-started
        mean_seconds = elapsed / max(1, step-start_step)
        return {
            "status": status, "updated_at_utc": now(), "pid": os.getpid(),
            "process_group": os.getpgrp(), "gpu_uuid": args.gpu_uuid,
            "step": step, "total_steps": total, "epoch": step/len(train_data),
            "train_clips": len(train_data), "parameters": model.num_parameters,
            "invocation_seconds": elapsed, "total_elapsed_seconds": previous_seconds+elapsed,
            "estimated_remaining_seconds": mean_seconds*(total-step) if step > start_step else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "best_validation_mse": best, "last_validation": last_metrics,
            "config_sha256": config_hash, **extra,
        }

    def checkpoint(status):
        info = metadata(status)
        atomic_torch(args.output/"last.pt", snapshot(model, spec, optimizer, config_hash, step, best, info))
        atomic_json(progress_path, info)

    def emit(record):
        log.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    torch.cuda.reset_peak_memory_stats()
    atomic_json(progress_path, metadata("preparing_validation"))
    emit({"event": "start", **metadata("starting"), "resumed": args.resume})
    try:
        # Bounded CPU cache: ~350 MB; all validation IDs remain held out.
        val_samples = list(make_loader(datasets["validation"], list(range(len(datasets["validation"]))), args.workers))
        guard(force=True)
        if step == 0:
            last_metrics = evaluate(model, val_samples, "cuda:0", guard)
            emit({"event": "initial_validation", "step": step, **last_metrics})
        recent = []
        while step < total:
            epoch, offset = divmod(step, len(train_data))
            generator = torch.Generator().manual_seed(args.seed+epoch)
            order = torch.randperm(len(train_data), generator=generator).tolist()
            keys = [(epoch+args.data_epoch_offset, index) for index in order[offset:]]
            loader = make_loader(train_data, keys, args.workers)
            model.train()
            for sample in loader:
                guard()
                if stop or time.monotonic()-started > args.max_seconds:
                    checkpoint("interrupted" if stop else "time_limit")
                    return
                lr = learning_rate(step, total, args.learning_rate, args.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                prediction, target = predict_patch(model, sample, "cuda:0")
                loss = F.mse_loss(prediction, target)
                if args.loss_mode == "detail":
                    dx = F.l1_loss(torch.diff(prediction, dim=-1), torch.diff(target, dim=-1))
                    dy = F.l1_loss(torch.diff(prediction, dim=-2), torch.diff(target, dim=-2))
                    loss = loss + 0.05 * (dx+dy)/2
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                recent.append(float(loss.detach()))
                if step == 1 or step % 100 == 0:
                    info = metadata("training", loss_mean=sum(recent)/len(recent),
                                    learning_rate=lr, gradient_norm=float(norm))
                    atomic_json(progress_path, info)
                    emit({"event": "train", **info})
                    recent.clear()
                if step % args.checkpoint_every == 0:
                    checkpoint("training")
                if step % args.validate_every == 0 or step == total:
                    atomic_json(progress_path, metadata("validating"))
                    last_metrics = evaluate(model, val_samples, "cuda:0", guard)
                    if best is None or last_metrics["mse"] < best:
                        best = last_metrics["mse"]
                        temporary = args.output/"best.tmp.pt"
                        save_checkpoint(temporary, model, spec, metadata("best_validation"))
                        os.replace(temporary, args.output/"best.pt")
                    temporary = args.output/"epoch.tmp.pt"
                    save_checkpoint(temporary, model, spec, metadata("validation_checkpoint"))
                    os.replace(temporary, args.output/f"decoder_step_{step:06d}.pt")
                    emit({"event": "validation", "step": step, **last_metrics})
                    checkpoint("training")
                    model.train()
                if step >= total:
                    break
        checkpoint("training_complete")
        del val_samples
        if not args.skip_test:
            # Test is opened only after training and validation selection finish.
            best_model, _, best_metadata = load_checkpoint(args.output/"best.pt", "cuda:0")
            test_loader = make_loader(datasets["test"], list(range(len(datasets["test"]))), args.workers)
            test_metrics = evaluate(best_model, test_loader, "cuda:0", guard)
            atomic_json(args.output/"test.json", {
                **test_metrics, "checkpoint": "best.pt", "checkpoint_sha256": sha256(args.output/"best.pt"),
                "selected_step": best_metadata["step"], "updated_at_utc": now(),
            })
            emit({"event": "test", **test_metrics})
        checkpoint("complete")
    except Exception as exc:
        checkpoint("failed")
        atomic_json(progress_path, metadata("failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-mode", choices=("mse", "detail"), default="mse")
    parser.add_argument("--data-epoch-offset", type=int, default=20)
    parser.add_argument("--variant", choices=("compact", "balanced", "quality", "wide"), required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--steps", type=int, default=90000)
    parser.add_argument("--max-seconds", type=float, default=28000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--core", nargs=2, type=int, default=[30, 52])
    parser.add_argument("--target-frames", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validate-every", type=int, default=4500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.max_seconds, args.learning_rate, args.target_frames, *args.core,
           args.validate_every, args.checkpoint_every) <= 0 or args.workers not in range(5) or args.warmup < 0:
        parser.error("Invalid training/resource bounds")
    run(args)


if __name__ == "__main__":
    main()
