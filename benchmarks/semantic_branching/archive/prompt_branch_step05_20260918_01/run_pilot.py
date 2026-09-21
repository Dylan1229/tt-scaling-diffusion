from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import diffusers
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parent
MODEL = "/data/datasets/fanjiang/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635"
A = "A realistic orange tabby cat walks slowly from left to right across a green grassy meadow. Full body side view, static camera, natural daylight, smooth continuous motion."
B = "A realistic golden retriever dog walks slowly from left to right across a green grassy meadow. Full body side view, static camera, natural daylight, smooth continuous motion."
STEPS, SWITCH, SEED, FPS = 50, 5, 0, 24

def digest(x):
    data = x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()

def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original_cat", "switch_step05", "target_dog"], required=True)
    args = parser.parse_args()
    out = ROOT / args.mode
    out.mkdir(exist_ok=False)
    started = time.time()
    meta = {
        "mode": args.mode, "prompt_A": A, "prompt_B": B, "negative_prompt": "",
        "model": MODEL, "seed": SEED, "steps": STEPS, "switch_after_completed_steps": SWITCH,
        "height": 480, "width": 832, "num_frames": 81, "fps": FPS,
        "guidance_scale": 5.0, "max_sequence_length": 512,
        "diffusers": diffusers.__version__, "torch": torch.__version__,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(0),
        "method": "Replay identical A prefix; replace positive conditioning after native step 5; retain all UniPC history; no re-noising or extra denoising steps.",
        "steps_log": [], "status": "loading",
    }
    write_json(out / "metadata.json", meta)
    print(f"START {args.mode}", flush=True)
    torch.manual_seed(SEED)
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    assert type(pipe.scheduler).__name__ == "UniPCMultistepScheduler"
    encode_kw = dict(negative_prompt="", do_classifier_free_guidance=True, device="cuda",
                     num_videos_per_prompt=1, max_sequence_length=512)
    a, negative = pipe.encode_prompt(prompt=A, **encode_kw)
    b, negative_b = pipe.encode_prompt(prompt=B, **encode_kw)
    a, b = a.to(pipe.transformer.dtype), b.to(pipe.transformer.dtype)
    negative = negative.to(pipe.transformer.dtype)
    assert torch.equal(negative, negative_b.to(pipe.transformer.dtype))
    meta.update(status="generating", load_encode_seconds=time.time()-started,
                scheduler=type(pipe.scheduler).__name__,
                scheduler_config=dict(pipe.scheduler.config),
                prompt_A_embedding_sha256=digest(a), prompt_B_embedding_sha256=digest(b),
                negative_embedding_sha256=digest(negative))
    write_json(out / "metadata.json", meta)
    generation_start = time.time()
    switched = False

    def callback(pipeline, index, timestep, values):
        nonlocal switched
        completed = index + 1
        active = "B" if args.mode == "target_dog" or (args.mode == "switch_step05" and completed > SWITCH) else "A"
        assert torch.equal(values["prompt_embeds"], b if active == "B" else a)
        latents = values["latents"]
        event = {"completed_steps": completed, "model_timestep": float(timestep),
                 "sigma_after": float(pipeline.scheduler.sigmas[completed]),
                 "conditioning_used": active, "elapsed_seconds": time.time()-generation_start}
        if completed <= SWITCH + 1 or completed == STEPS:
            event["latent_sha256"] = digest(latents)
        if completed == SWITCH:
            torch.save(latents.detach().cpu(), out / "latent_after_step05.pt")
            event["scheduler_step_index"] = pipeline.scheduler.step_index
            event["scheduler_lower_order_nums"] = pipeline.scheduler.lower_order_nums
            event["scheduler_history_sha256"] = [None if t is None else digest(t) for t in pipeline.scheduler.model_outputs]
        if args.mode == "switch_step05" and completed == SWITCH:
            values["prompt_embeds"] = b
            switched = True
            event["switch_prompt_after_this_step"] = True
        meta["steps_log"].append(event)
        write_json(out / "metadata.json", meta)
        if completed <= SWITCH+1 or completed % 5 == 0:
            print(json.dumps(event), flush=True)
        return values

    result = pipe(prompt_embeds=b if args.mode=="target_dog" else a,
                  negative_prompt_embeds=negative,
                  height=480, width=832, num_frames=81,
                  num_inference_steps=STEPS, guidance_scale=5.0,
                  generator=torch.Generator(device="cuda").manual_seed(SEED),
                  callback_on_step_end=callback,
                  callback_on_step_end_tensor_inputs=["latents","prompt_embeds"],
                  max_sequence_length=512, output_type="np")
    assert len(meta["steps_log"]) == STEPS
    assert switched == (args.mode == "switch_step05")
    frames = result.frames[0]
    assert frames.shape == (81,480,832,3), frames.shape
    assert np.isfinite(frames).all()
    video = out / "video.mp4"
    export_to_video(frames, str(video), fps=FPS)
    np.save(out / "frames_uint8.npy", (frames.clip(0,1)*255).round().astype(np.uint8))
    meta.update(status="complete", generation_and_decode_seconds=time.time()-generation_start,
                total_seconds=time.time()-started, frame_shape=list(frames.shape),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    write_json(out / "metadata.json", meta)
    (out / "DONE").touch()
    print(f"DONE {args.mode} {meta['total_seconds']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
