from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,diffusers
from diffusers import WanPipeline,AutoencoderKLWan,UniPCMultistepScheduler
from diffusers.utils import export_to_video
from run_case import digest,write_json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"full_dog_fresh_noise"
OLD=json.loads((ROOT/"metadata.json").read_text())
@torch.inference_mode()
def main():
 start=time.time()
 noise=torch.load(ROOT/"fresh_random_noise.pt",map_location="cpu",weights_only=True)
 assert noise.dtype==torch.float32 and digest(noise)==OLD["fresh_noise_sha256"]
 meta={k:OLD[k] for k in ["model","prompt_B","negative_prompt","height","width","num_frames","fps","guidance_scale","max_sequence_length","scheduler_config"]}
 meta.update(mode="dog_prompt_from_start",seed=OLD["noise_seed"],steps=50,status="loading",
   initial_noise_path=str(ROOT/"fresh_random_noise.pt"),initial_noise_sha256=digest(noise),
   initial_noise_identical_to_renoise_fresh_noise=True,
   method="Use the exact saved fresh standard Gaussian from the preceding re-noise run as the initial latent; dog prompt for all50 native WanPipeline steps, fresh full50 UniPC history, no cat conditioning or cat latent.",
   diffusers=diffusers.__version__,torch=torch.__version__,physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),steps_log=[])
 write_json(OUT/"metadata.json",meta)
 vae=AutoencoderKLWan.from_pretrained(meta["model"],subfolder="vae",torch_dtype=torch.float32,local_files_only=True)
 pipe=WanPipeline.from_pretrained(meta["model"],vae=vae,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
 pipe.scheduler=UniPCMultistepScheduler.from_config(meta["scheduler_config"])
 prompt,negative=pipe.encode_prompt(prompt=meta["prompt_B"],negative_prompt="",do_classifier_free_guidance=True,
  device="cuda",num_videos_per_prompt=1,max_sequence_length=512)
 prompt,negative=prompt.to(pipe.transformer.dtype),negative.to(pipe.transformer.dtype)
 assert digest(prompt)==OLD["prompt_B_embedding_sha256"]
 assert digest(negative)==OLD["negative_embedding_sha256"]
 initial=noise.to("cuda")
 assert digest(initial)==meta["initial_noise_sha256"]
 meta.update(status="generating",load_seconds=time.time()-start,
  prompt_B_embedding_sha256=digest(prompt),negative_embedding_sha256=digest(negative))
 write_json(OUT/"metadata.json",meta)
 gen_start=time.time()
 def callback(p,i,t,values):
  assert torch.equal(values["prompt_embeds"],prompt)
  assert p.scheduler.step_index==i+1
  event={"completed_steps":i+1,"model_timestep":int(t),"sigma_after":float(p.scheduler.sigmas[i+1]),
   "conditioning":"B","solver_order":int(p.scheduler.this_order),"elapsed_seconds":time.time()-gen_start}
  if i==0 or i==49: event["latent_sha256"]=digest(values["latents"])
  meta["steps_log"].append(event)
  write_json(OUT/"metadata.json",meta)
  if i==0 or (i+1)%10==0: print("STEP",json.dumps(event),flush=True)
  return values
 frames=pipe(prompt_embeds=prompt,negative_prompt_embeds=negative,height=480,width=832,num_frames=81,
  num_inference_steps=50,guidance_scale=5.0,latents=initial,
  callback_on_step_end=callback,callback_on_step_end_tensor_inputs=["latents","prompt_embeds"],
  max_sequence_length=512,output_type="np").frames[0]
 assert frames.shape==(81,480,832,3) and np.isfinite(frames).all()
 assert len(meta["steps_log"])==50 and meta["steps_log"][-1]["sigma_after"]==0.0
 export_to_video(frames,str(OUT/"video.mp4"),fps=24)
 np.save(OUT/"frames_uint8.npy",(frames.clip(0,1)*255).round().astype(np.uint8))
 meta.update(status="complete",generation_and_decode_seconds=time.time()-gen_start,total_seconds=time.time()-start,
  frame_shape=list(frames.shape),video_sha256=hashlib.sha256((OUT/"video.mp4").read_bytes()).hexdigest())
 write_json(OUT/"metadata.json",meta)
 (OUT/"DONE").touch()
 print("DONE",meta["total_seconds"],flush=True)
if __name__=="__main__":main()
