from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from diffusers.utils import export_to_video
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"full_dog_fresh_noise"
assert (OUT/"DONE").exists()
old=json.loads((ROOT/"metadata.json").read_text())
new=json.loads((OUT/"metadata.json").read_text())
assert old["fresh_noise_sha256"]==new["initial_noise_sha256"]
for k in ["model","prompt_B","negative_prompt","height","width","num_frames","fps","guidance_scale","max_sequence_length","prompt_B_embedding_sha256","negative_embedding_sha256"]:
 assert old[k]==new[k],k
panels=[("Re-noise step 40 -> 4; dog (46 steps)",ROOT/"frames_uint8.npy"),("Dog prompt from start (50 steps)",OUT/"frames_uint8.npy")]
data=[np.load(p,mmap_mode="r") for _,p in panels]
font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def font(n):return ImageFont.truetype(font_path,n)
video=[]
for i in range(81):
 im=Image.new("RGB",(1664,576),"#f5f6f8");d=ImageDraw.Draw(im)
 d.text((16,12),"Same fresh Gaussian noise: seed 20260921 | Wan2.2 TI2V-5B",font=font(30),fill="#182235")
 for j,(label,_) in enumerate(panels):
  d.text((j*832+16,60),label,font=font(28),fill="#182235")
  im.paste(Image.fromarray(data[j][i]),(j*832,96))
 video.append(im)
export_to_video(video,str(ROOT/"renoise_vs_full_dog.mp4"),fps=24)
indices=[0,16,32,48,64,80]
sheet=Image.new("RGB",(1664,3188),"#f5f6f8");d=ImageDraw.Draw(sheet)
d.text((16,14),"Re-noised cat state vs dog generation from the start",font=font(36),fill="#182235")
d.text((16,65),"Same fresh Gaussian z (seed 20260921); identical model, prompt, CFG and output size",font=font(26),fill="#4b5568")
for j,(label,_) in enumerate(panels):
 d.text((j*832+16,110),label,font=font(28),fill="#182235")
for row,i in enumerate(indices):
 y=152+row*506
 d.text((15,y),f"Frame {i:02d} / 80 | {i/24:.2f} s",font=font(21),fill="#4b5568")
 for j in range(2):sheet.paste(Image.fromarray(data[j][i]),(j*832,y+26))
sheet.save(ROOT/"renoise_vs_full_dog.jpg",quality=95,subsampling=0)
audit=Image.new("RGB",(2493,1730),"#f5f6f8");d=ImageDraw.Draw(audit)
d.text((15,14),"DOG FROM START | all 81 frames | seed 20260921",font=font(30),fill="#182235")
for i in range(81):
 x=(i%9)*277;y=65+(i//9)*185
 d.text((x+5,y),f"{i:02d}  {i/24:.2f}s",font=font(17),fill="#4b5568")
 audit.paste(Image.fromarray(data[1][i]).resize((277,160),Image.Resampling.LANCZOS),(x,y+25))
audit.save(OUT/"all_frames.jpg",quality=94,subsampling=0)
for i in [0,40,80]:Image.fromarray(data[1][i]).save(OUT/f"frame_{i:02d}.jpg",quality=97,subsampling=0)
manifest={"same_noise_sha256":new["initial_noise_sha256"],"sampled_frames":indices,"files":{}}
for rel in ["renoise_vs_full_dog.mp4","renoise_vs_full_dog.jpg","full_dog_fresh_noise/video.mp4","full_dog_fresh_noise/metadata.json","full_dog_fresh_noise/all_frames.jpg"]:
 p=ROOT/rel
 manifest["files"][rel]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/"full_dog_comparison_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
(OUT/"README.md").write_text(
 "# Full dog prompt control\n\n"
 "Dog prompt throughout all50 original denoising steps. Initial float32 latent is the exact saved fresh Gaussian noise from the preceding step40-to4 re-noising run (seed20260921), verified by SHA256. No cat latent or cat conditioning is used.\n\n"
 "Fresh full50 UniPC scheduler; same model snapshot, CFG5, 832x480,81 frames,24fps and empty negative prompt. All50 steps checked for dog conditioning and terminal sigma0.\n\n"
 "This paired comparison holds the Gaussian realization fixed. The full-start run also includes four additional denoising updates, so output differences cannot be attributed solely to the retained cat latent.\n"
)
print(json.dumps(manifest,indent=2))
