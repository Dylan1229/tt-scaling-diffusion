from __future__ import annotations
import hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import diffusers
from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PREV = Path("/data/datasets/peihao/tt-scaling-diffusion/runs/prompt_branch_boundary_20260918_01")
PILOT = Path("/data/datasets/peihao/tt-scaling-diffusion/runs/prompt_branch_step05_20260918_01")
BASE = json.loads((PREV / "step_040/metadata.json").read_text())
NOISE_SEED = 20260923
RESTART_AFTER = 4

def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def write_json(p, d):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2) + "\n")
    tmp.replace(p)

class RestartUniPC(UniPCMultistepScheduler):
    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        assert num_inference_steps == 50 and sigmas is None
        super().set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        self.timesteps = self.timesteps[RESTART_AFTER:].clone()
        self.sigmas = self.sigmas[RESTART_AFTER:].clone()
        self.num_inference_steps = len(self.timesteps)
        self.timestep_list = [None] * self.config.solver_order
        assert len(self.timesteps) == 46 and len(self.sigmas) == 47
        assert self.lower_order_nums == 0 and self.last_sample is None
        assert all(x is None for x in self.model_outputs)

def verify_scheduler_restart():
    full = UniPCMultistepScheduler.from_config(BASE["scheduler_config"])
    sliced = RestartUniPC.from_config(BASE["scheduler_config"])
    full.set_timesteps(50)
    full.set_begin_index(4)
    sliced.set_timesteps(50)
    sliced.set_begin_index(0)
    assert torch.equal(full.timesteps[4:], sliced.timesteps)
    assert torch.equal(full.sigmas[4:], sliced.sigmas)
    x = torch.randn((1, 4, 2, 4, 4), generator=torch.Generator().manual_seed(91))
    y = x.clone()
    max_error = 0.
    for i, t in enumerate(sliced.timesteps):
        vx = 0.125 * x + float(t) * 0.0001
        vy = 0.125 * y + float(t) * 0.0001
        x = full.step(vx, t, x, return_dict=False)[0]
        y = sliced.step(vy, t, y, return_dict=False)[0]
        err = float((x-y).abs().max())
        max_error = max(max_error, err)
        assert torch.equal(x,y), (i,err)
    return {"compared_steps": 46, "max_abs_error": max_error, "bit_identical": True}

@torch.inference_mode()
def main():
    start = time.time()
    scheduler_check = verify_scheduler_restart()
    source_path = PREV / "step_040/branch_latent.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    assert source.dtype == torch.float32
    source_hash = digest(source)
    assert source_hash == BASE["steps_log"][39]["latent_sha256"]
    original_meta = json.loads((PREV / "original_cat/metadata.json").read_text())
    assert source_hash == original_meta["steps_log"][39]["latent_sha256"]
    sched = UniPCMultistepScheduler.from_config(BASE["scheduler_config"])
    sched.set_timesteps(50)
    sigma_source, sigma_target = float(sched.sigmas[40]), float(sched.sigmas[4])
    assert sigma_source == BASE["steps_log"][39]["sigma_after"]
    a = (1-sigma_target)/(1-sigma_source)
    b = math.sqrt(sigma_target**2-(a*sigma_source)**2)
    assert abs(a*(1-sigma_source)-(1-sigma_target)) < 1e-12
    assert abs((a*sigma_source)**2+b*b-sigma_target**2) < 1e-12
    meta = {k:BASE[k] for k in ["model","prompt_A","prompt_B","negative_prompt","seed","steps","height","width","num_frames","fps","guidance_scale","max_sequence_length"]}
    meta.update(status="loading", source_latent=str(source_path), source_latent_sha256=source_hash,
        source_completed_steps=40, target_completed_steps=4, restart_model_timestep=int(sched.timesteps[4]),
        sigma_source=sigma_source, sigma_target=sigma_target, noise_seed=NOISE_SEED,
        source_coefficient=a, fresh_noise_coefficient=b, remaining_denoising_steps=46,
        method="Marginal-matched flow re-noise: x_new = a*x40+b*z_fresh; a=(1-sigma4)/(1-sigma40); b=sqrt(sigma4^2-a^2*sigma40^2). z_fresh is independent unit Gaussian. No old step4 latent/noise is loaded. Then B conditioning throughout a fresh UniPC restart using the exact original schedule suffix.",
        assumptions="Matches nominal flow marginal signal/noise coefficients; finite-solver conditional x40 need not have an exact recoverable x0/noise decomposition.",
        scheduler_restart_check=scheduler_check, source_matches_original_cat_at_step40=True,
        scheduler_config=BASE["scheduler_config"], diffusers=diffusers.__version__, torch=torch.__version__,
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), steps_log=[])
    write_json(ROOT/"metadata.json",meta)
    print("PREFLIGHT",json.dumps({k:meta[k] for k in ["sigma_source","sigma_target","source_coefficient","fresh_noise_coefficient","scheduler_restart_check"]}),flush=True)
    vae = AutoencoderKLWan.from_pretrained(BASE["model"],subfolder="vae",torch_dtype=torch.float32,local_files_only=True)
    pipe = WanPipeline.from_pretrained(BASE["model"],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
    assert pipe.config.boundary_ratio is None and pipe.config.expand_timesteps
    pipe.scheduler = RestartUniPC.from_config(BASE["scheduler_config"])
    prompt, negative = pipe.encode_prompt(prompt=BASE["prompt_B"],negative_prompt="",do_classifier_free_guidance=True,
        device="cuda",num_videos_per_prompt=1,max_sequence_length=512)
    prompt,negative = prompt.to(pipe.transformer.dtype),negative.to(pipe.transformer.dtype)
    assert digest(prompt) == BASE["prompt_B_embedding_sha256"]
    assert digest(negative) == BASE["negative_embedding_sha256"]
    noise = torch.randn(source.shape,device="cuda",dtype=torch.float32,
        generator=torch.Generator(device="cuda").manual_seed(NOISE_SEED))
    latents = source.to("cuda")*a + noise*b
    assert torch.isfinite(latents).all()
    meta.update(status="generating", fresh_noise_sha256=digest(noise), renoised_latent_sha256=digest(latents),
        fresh_noise_mean=float(noise.mean()),fresh_noise_std=float(noise.std()),
        prompt_B_embedding_sha256=digest(prompt),negative_embedding_sha256=digest(negative),
        source_latent_shape=list(source.shape),load_seconds=time.time()-start)
    torch.save(noise.cpu(),ROOT/"fresh_random_noise.pt")
    torch.save(latents.cpu(),ROOT/"renoised_latent.pt")
    write_json(ROOT/"metadata.json",meta)
    print("GENERATING",meta["load_seconds"],flush=True)
    gen_start=time.time()
    def callback(p,i,t,values):
        assert int(t) == int(sched.timesteps[i+4])
        assert torch.equal(values["prompt_embeds"],prompt)
        assert p.scheduler.step_index == i+1
        event={"restart_iteration":i+1,"original_schedule_completed_steps":i+5,
            "model_timestep":int(t),"sigma_after":float(p.scheduler.sigmas[i+1]),
            "conditioning":"B","solver_order":int(p.scheduler.this_order),
            "elapsed_seconds":time.time()-gen_start}
        if i==0 or i==45:
            event["latent_sha256"]=digest(values["latents"])
        meta["steps_log"].append(event)
        write_json(ROOT/"metadata.json",meta)
        if i==0 or (i+1)%10==0 or i==45: print("STEP",json.dumps(event),flush=True)
        return values
    frames=pipe(prompt_embeds=prompt,negative_prompt_embeds=negative,height=480,width=832,num_frames=81,
        num_inference_steps=50,guidance_scale=5.0,latents=latents,
        callback_on_step_end=callback,callback_on_step_end_tensor_inputs=["latents","prompt_embeds"],
        max_sequence_length=512,output_type="np").frames[0]
    assert frames.shape==(81,480,832,3) and np.isfinite(frames).all()
    assert len(meta["steps_log"])==46
    assert meta["steps_log"][0]["solver_order"]==1
    assert meta["steps_log"][-1]["original_schedule_completed_steps"]==50
    assert meta["steps_log"][-1]["sigma_after"]==0.0
    export_to_video(frames,str(ROOT/"video.mp4"),fps=24)
    np.save(ROOT/"frames_uint8.npy",(frames.clip(0,1)*255).round().astype(np.uint8))
    meta.update(status="complete",generation_and_decode_seconds=time.time()-gen_start,
        total_seconds=time.time()-start,frame_shape=list(frames.shape),
        video_sha256=hashlib.sha256((ROOT/"video.mp4").read_bytes()).hexdigest())
    write_json(ROOT/"metadata.json",meta)
    (ROOT/"DONE").touch()
    print("DONE",meta["total_seconds"],flush=True)

if __name__=="__main__":
    main()
