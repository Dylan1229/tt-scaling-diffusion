from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import diffusers
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/"prompt_branch_step05_20260918_01"
BASE=json.loads((PILOT/"switch_step05"/"metadata.json").read_text())
A,B,MODEL=BASE["prompt_A"],BASE["prompt_B"],BASE["model"]

def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def write_json(p,d):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d,indent=2)+"\n")
    tmp.replace(p)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",nargs="+",type=int,required=True)
    args=parser.parse_args()
    assert all(5<=k<=50 for k in args.steps)
    torch.manual_seed(0)
    t0=time.time()
    vae=AutoencoderKLWan.from_pretrained(MODEL,subfolder="vae",torch_dtype=torch.float32,local_files_only=True)
    pipe=WanPipeline.from_pretrained(MODEL,vae=vae,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
    assert type(pipe.scheduler).__name__=="UniPCMultistepScheduler"
    kw=dict(negative_prompt="",do_classifier_free_guidance=True,device="cuda",num_videos_per_prompt=1,max_sequence_length=512)
    a,negative=pipe.encode_prompt(prompt=A,**kw)
    b,negative_b=pipe.encode_prompt(prompt=B,**kw)
    a,b,negative=a.to(pipe.transformer.dtype),b.to(pipe.transformer.dtype),negative.to(pipe.transformer.dtype)
    assert torch.equal(negative,negative_b.to(pipe.transformer.dtype))
    embeddings={"prompt_A_embedding_sha256":digest(a),"prompt_B_embedding_sha256":digest(b),"negative_embedding_sha256":digest(negative)}
    for key,value in embeddings.items(): assert value==BASE[key],key
    print("LOADED",time.time()-t0,flush=True)
    for switch in args.steps:
        out=ROOT/("original_cat" if switch==50 else f"step_{switch:03d}")
        out.mkdir(exist_ok=False)
        start=time.time()
        meta={key:BASE[key] for key in ["prompt_A","prompt_B","negative_prompt","model","seed","steps","height","width","num_frames","fps","guidance_scale","max_sequence_length"]}
        meta.update(embeddings)
        meta.update(mode="original_cat" if switch==50 else "prompt_switch",switch_after_completed_steps=switch,
                    scheduler=type(pipe.scheduler).__name__,scheduler_config=dict(pipe.scheduler.config),
                    diffusers=diffusers.__version__,torch=torch.__version__,
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),steps_log=[],status="generating",
                    method="Native prompt switch after k completed steps; original float32 latent and UniPC history retained. Same-seed prefix replay; no re-noise.")
        write_json(out/"metadata.json",meta)
        switched=False
        def callback(p,i,t,values):
            nonlocal switched
            done=i+1
            active="B" if done>switch else "A"
            assert torch.equal(values["prompt_embeds"],b if active=="B" else a)
            event={"completed_steps":done,"model_timestep":float(t),"sigma_after":float(p.scheduler.sigmas[done]),
                   "conditioning_used":active,"elapsed_seconds":time.time()-start}
            if done<=switch+1 or done==50:
                event["latent_sha256"]=digest(values["latents"])
            if switch==50 or done==switch:
                event.update(scheduler_step_index=p.scheduler.step_index,
                             scheduler_lower_order_nums=p.scheduler.lower_order_nums,
                             scheduler_history_sha256=[None if x is None else digest(x) for x in p.scheduler.model_outputs])
            if done==switch and switch<50:
                torch.save(values["latents"].detach().cpu(),out/"branch_latent.pt")
                values["prompt_embeds"]=b
                switched=True
                event["switch_prompt_after_this_step"]=True
            meta["steps_log"].append(event)
            write_json(out/"metadata.json",meta)
            if done==switch or done%10==0:
                print(json.dumps({"switch":switch,"step":done,"active":active,"elapsed":time.time()-start}),flush=True)
            return values
        frames=pipe(prompt_embeds=a,negative_prompt_embeds=negative,height=480,width=832,num_frames=81,
                    num_inference_steps=50,guidance_scale=5.0,generator=torch.Generator(device="cuda").manual_seed(0),
                    callback_on_step_end=callback,callback_on_step_end_tensor_inputs=["latents","prompt_embeds"],
                    max_sequence_length=512,output_type="np").frames[0]
        assert len(meta["steps_log"])==50
        assert switched==(switch<50)
        assert frames.shape==(81,480,832,3) and np.isfinite(frames).all()
        export_to_video(frames,str(out/"video.mp4"),fps=24)
        np.save(out/"frames_uint8.npy",(frames.clip(0,1)*255).round().astype(np.uint8))
        meta.update(status="complete",generation_and_decode_seconds=time.time()-start,frame_shape=list(frames.shape),
                    video_sha256=hashlib.sha256((out/"video.mp4").read_bytes()).hexdigest())
        write_json(out/"metadata.json",meta)
        (out/"DONE").touch()
        print("DONE",switch,time.time()-start,flush=True)
if __name__=="__main__":main()
