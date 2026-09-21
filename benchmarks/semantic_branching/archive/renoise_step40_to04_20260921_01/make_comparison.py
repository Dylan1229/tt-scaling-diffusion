import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from diffusers.utils import export_to_video
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"prompt_branch_boundary_20260918_01"
assert (ROOT/"DONE").exists()
font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def font(size): return ImageFont.truetype(font_path,size)
panels=[
 ("Original cat (A throughout)",PREV/"original_cat/frames_uint8.npy"),
 ("Prompt switch at step 40",PREV/"step_040/frames_uint8.npy"),
 ("Step 40 + fresh noise -> step 4; dog",ROOT/"frames_uint8.npy")]
data=[np.load(p,mmap_mode="r") for _,p in panels]
assert all(x.shape==(81,480,832,3) for x in data)
# Synchronized side-by-side video, all 81 frames.
video=[]
for i in range(81):
 im=Image.new("RGB",(1872,448),"#f5f6f8")
 d=ImageDraw.Draw(im)
 d.text((18,10),"Cat -> dog | Re-noise feasibility | source seed 0, fresh noise seed 20260921",font=font(24),fill="#182235")
 for j,(label,_) in enumerate(panels):
  d.text((j*624+12,48),label,font=font(20),fill="#182235")
  im.paste(Image.fromarray(data[j][i]).resize((624,360),Image.Resampling.LANCZOS),(j*624,76))
 video.append(im)
export_to_video(video,str(ROOT/"comparison.mp4"),fps=24)
# Large contact sheet: same six frame indices across all conditions.
indices=[0,16,32,48,64,80]
sheet=Image.new("RGB",(2496,3188),"#f5f6f8")
d=ImageDraw.Draw(sheet)
d.text((20,15),"Step 40 latent + fresh random noise -> step 4 noise level -> dog prompt",font=font(34),fill="#182235")
d.text((20,62),"Wan2.2 TI2V-5B | original 50-step schedule | 46 restarted denoising steps | single case",font=font(25),fill="#4b5568")
for j,(label,_) in enumerate(panels):
 d.text((j*832+15,105),label,font=font(28),fill="#182235")
for row,i in enumerate(indices):
 y=152+row*506
 d.text((15,y),f"Frame {i:02d} / 80   |   {i/24:.2f} s",font=font(21),fill="#4b5568")
 for j in range(3):
  sheet.paste(Image.fromarray(data[j][i]),(j*832,y+26))
sheet.save(ROOT/"comparison.jpg",quality=95,subsampling=0)
# Full-video visual audit at all 81 frames.
thumb_w,thumb_h=277,160
audit=Image.new("RGB",(9*thumb_w,9*185+65),"#f5f6f8")
d=ImageDraw.Draw(audit)
d.text((15,14),"NEW RESULT | all 81 frames, chronological order",font=font(30),fill="#182235")
for i in range(81):
 x=(i%9)*thumb_w;y=65+(i//9)*185
 d.text((x+5,y),f"{i:02d}  {i/24:.2f}s",font=font(17),fill="#4b5568")
 audit.paste(Image.fromarray(data[2][i]).resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y+25))
audit.save(ROOT/"all_frames.jpg",quality=95,subsampling=0)
for i in [0,16,40,64,80]:
 Image.fromarray(data[2][i]).save(ROOT/f"frame_{i:02d}.jpg",quality=97,subsampling=0)
manifest={"panels":[{"label":label,"frames":str(p)} for label,p in panels],"sampled_frame_indices":indices,
 "files":{}}
for name in ["video.mp4","comparison.mp4","comparison.jpg","all_frames.jpg","metadata.json"]:
 p=ROOT/name
 manifest["files"][name]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
