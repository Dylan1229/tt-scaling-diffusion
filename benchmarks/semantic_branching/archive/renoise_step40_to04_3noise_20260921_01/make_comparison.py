from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from diffusers.utils import export_to_video
ROOT=Path(__file__).resolve().parent
SEEDS=[20260922,20260923,20260924]
OUTS=[ROOT/f"seed_{s}" for s in SEEDS]
assert all((p/"DONE").exists() for p in OUTS)
metas=[json.loads((p/"metadata.json").read_text()) for p in OUTS]
fixed=["model","prompt_A","prompt_B","negative_prompt","seed","steps","height","width","num_frames","fps","guidance_scale","max_sequence_length","source_latent_sha256","source_completed_steps","target_completed_steps","sigma_source","sigma_target","source_coefficient","fresh_noise_coefficient","remaining_denoising_steps","scheduler_config","prompt_B_embedding_sha256","negative_embedding_sha256"]
for k in fixed:assert all(m[k]==metas[0][k] for m in metas),k
assert [m["noise_seed"] for m in metas]==SEEDS
assert len({m["fresh_noise_sha256"] for m in metas})==3
assert len({m["renoised_latent_sha256"] for m in metas})==3
assert len({m["video_sha256"] for m in metas})==3
for m in metas:
 assert m["status"]=="complete" and len(m["steps_log"])==46
 assert all(e["conditioning"]=="B" for e in m["steps_log"])
 assert m["steps_log"][-1]["sigma_after"]==0
data=[np.load(p/"frames_uint8.npy",mmap_mode="r") for p in OUTS]
font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def font(n):return ImageFont.truetype(font_path,n)
labels=[f"Noise {i+1} | seed {s}" for i,s in enumerate(SEEDS)]
video=[]
for i in range(81):
 im=Image.new("RGB",(1872,448),"#f5f6f8");d=ImageDraw.Draw(im)
 d.text((16,10),"Same step 40 cat latent -> step 4 noise level -> same dog prompt | only fresh noise changes",font=font(27),fill="#182235")
 for j,label in enumerate(labels):
  d.text((j*624+12,48),label,font=font(24),fill="#182235")
  im.paste(Image.fromarray(data[j][i]).resize((624,360),Image.Resampling.LANCZOS),(j*624,76))
 video.append(im)
export_to_video(video,str(ROOT/"comparison.mp4"),fps=24)
indices=[0,16,32,48,64,80]
sheet=Image.new("RGB",(2496,3188),"#f5f6f8");d=ImageDraw.Draw(sheet)
d.text((16,15),"Three fresh-noise samples | fixed step 40 cat latent -> step 4 noise level -> golden retriever prompt",font=font(34),fill="#182235")
d.text((16,64),"Wan2.2 TI2V-5B | identical source, re-noising strength and 46-step restart | source seed 0",font=font(26),fill="#4b5568")
for j,label in enumerate(labels):d.text((j*832+16,108),label,font=font(30),fill="#182235")
for row,i in enumerate(indices):
 y=152+row*506
 d.text((15,y),f"Frame {i:02d} / 80 | {i/24:.2f} s",font=font(21),fill="#4b5568")
 for j in range(3):sheet.paste(Image.fromarray(data[j][i]),(j*832,y+26))
sheet.save(ROOT/"contact_sheet.jpg",quality=95,subsampling=0)
for j,out in enumerate(OUTS):
 audit=Image.new("RGB",(2493,1730),"#f5f6f8");d=ImageDraw.Draw(audit)
 d.text((15,14),f"NOISE {j+1} | seed {SEEDS[j]} | all81 frames, chronological order",font=font(30),fill="#182235")
 for i in range(81):
  x=(i%9)*277;y=65+(i//9)*185
  d.text((x+5,y),f"{i:02d}  {i/24:.2f}s",font=font(17),fill="#4b5568")
  audit.paste(Image.fromarray(data[j][i]).resize((277,160),Image.Resampling.LANCZOS),(x,y+25))
 audit.save(out/"all_frames.jpg",quality=92,subsampling=0)
 for i in [0,40,80]:Image.fromarray(data[j][i]).save(out/f"frame_{i:02d}.jpg",quality=97,subsampling=0)
validation={"noise_seeds":SEEDS,"fixed_fields_checked":fixed,"same_source_latent":True,
 "source_latent_sha256":metas[0]["source_latent_sha256"],"distinct_noise_hashes":True,
 "distinct_renoised_latent_hashes":True,"distinct_video_hashes":True,"all_three_complete_46_dog_steps":True,
 "sigma_source":metas[0]["sigma_source"],"sigma_target":metas[0]["sigma_target"],
 "source_coefficient":metas[0]["source_coefficient"],"fresh_noise_coefficient":metas[0]["fresh_noise_coefficient"]}
(ROOT/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
(ROOT/"README.md").write_text(
 "# Three fresh-noise re-noising samples\n\n"
 "All three use the exact saved cat latent after40 completed denoising steps, from source seed0. Each adds independently seeded fresh standard Gaussian noise to the nominal noise level after step4, then applies the identical golden retriever prompt for46 restarted UniPC updates. Only the fresh-noise seed changes:20260922,20260923,20260924.\n\n"
 "Flow re-noising: x_restart=a*x40+b*z; a=(1-sigma4)/(1-sigma40), b=sqrt(sigma4^2-a^2*sigma40^2). "
 f"a={validation['source_coefficient']}, b={validation['fresh_noise_coefficient']}.\n\n"
 "Model Wan2.2 TI2V-5B T2V; CFG5;832x480;81frames;24fps; original50-step schedule suffix with solver history reset. "
 "Source, prompt embeddings, noise strength and schedule are checked identical. Noise tensors, re-noised latents and videos have distinct hashes. See validation.json and per-seed metadata.\n\n"
 "comparison.mp4 synchronizes the three clips. contact_sheet.jpg samples frames0/16/32/48/64/80. "
 "Each seed directory contains its video, full81-frame sheet and metadata. Tensor artifacts remain on yukelab.\n"
)
manifest={"files":{}}
names=["comparison.mp4","contact_sheet.jpg","validation.json","README.md"]
for s in SEEDS:names.extend([f"seed_{s}/video.mp4",f"seed_{s}/all_frames.jpg",f"seed_{s}/metadata.json"])
for rel in names:
 p=ROOT/rel
 manifest["files"][rel]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"validation":validation,"artifacts":manifest},indent=2))
