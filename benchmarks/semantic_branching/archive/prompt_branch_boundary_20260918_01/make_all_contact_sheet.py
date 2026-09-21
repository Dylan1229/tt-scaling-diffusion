from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/"prompt_branch_step05_20260918_01"
FRAME=16
COLS=6
TW,TH=832,480
LABEL=48
GAP=14
MARGIN=28
HEADER=144
ROWS=9
W=2*MARGIN+COLS*TW+(COLS-1)*GAP
H=HEADER+ROWS*(TH+LABEL)+(ROWS-1)*GAP+MARGIN
fontpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
titlefont=ImageFont.truetype(fontpath,52)
subfont=ImageFont.truetype(fontpath,28)
labelfont=ImageFont.truetype(fontpath,29)
smallfont=ImageFont.truetype(fontpath,25)
sheet=Image.new("RGB",(W,H),(16,22,31))
d=ImageDraw.Draw(sheet)
d.text((MARGIN,22),"CAT → DOG PROMPT BRANCHING · ALL TESTED STEPS",font=titlefont,fill=(245,248,252))
d.text((MARGIN,88),"Same case / seed 0 / 50 denoising steps / frame 16 of 80 (0.67 s) / read left-to-right, top-to-bottom",font=subfont,fill=(178,191,211))
manifest=[]
for k in range(51):
    path=PILOT/"target_dog" if k==0 else PILOT/"switch_step05" if k==5 else ROOT/"original_cat" if k==50 else ROOT/f"step_{k:03d}"
    meta=json.loads((path/"metadata.json").read_text())
    assert meta["status"]=="complete",k
    assert len(meta["steps_log"])==50,k
    arr=np.load(path/"frames_uint8.npy",mmap_mode="r")
    assert arr.shape==(81,480,832,3) and arr.dtype==np.uint8,(k,arr.shape,arr.dtype)
    expected=["B"]*50 if k==0 else ["A"]*k+["B"]*(50-k)
    assert [x["conditioning_used"] for x in meta["steps_log"]]==expected,k
    row,col=divmod(k,COLS)
    x=MARGIN+col*(TW+GAP)
    y=HEADER+row*(TH+LABEL+GAP)
    d.rectangle((x,y,x+TW-1,y+LABEL-1),fill=(39,55,74) if k in (0,50) else (30,39,53))
    label="00  ·  ALL DOG — control" if k==0 else "50  ·  ALL CAT — control" if k==50 else f"{k:02d}  ·  switch after {k} steps  |  {k} A + {50-k} B"
    d.text((x+12,y+8),label,font=labelfont,fill=(247,249,252))
    sheet.paste(Image.fromarray(arr[FRAME]),(x,y+LABEL))
    manifest.append({"k":k,"kind":"dog_control" if k==0 else "cat_control" if k==50 else "prompt_switch",
                     "source":str(path),"frame_index":FRAME,"row":row,"col":col,
                     "frame_sha256":hashlib.sha256(arr[FRAME].tobytes()).hexdigest()})
# Remaining three cells form an explanatory legend.
x=MARGIN+3*(TW+GAP);y=HEADER+8*(TH+LABEL+GAP)
legend=[
    "A = orange tabby cat walking across a meadow",
    "B = golden retriever dog walking across a meadow",
    "",
    "01–49: switch conditioning after k completed denoising steps.",
    "00: all-dog reference.  50: all-cat reference.",
    "No quality labels: inspect the outputs directly.",
    "",
    "One identical video-frame index per output; original 832 × 480 pixels.",
    "All 49 switch positions + 2 controls. No tested position omitted.",
]
for i,line in enumerate(legend):
    d.text((x+16,y+18+i*46),line,font=smallfont,fill=(181,194,214))
out=ROOT/"prompt_branching_all_steps_00-50_seed0_frame16_2026-09-18.jpg"
sheet.save(out,quality=95,subsampling=0,optimize=True)
with Image.open(out) as im: im.verify()
assert len(manifest)==51 and [x["k"] for x in manifest]==list(range(51))
manifest_doc={"image":str(out),"size":[W,H],"tiles":manifest,"image_sha256":hashlib.sha256(out.read_bytes()).hexdigest()}
(ROOT/"all_steps_contact_sheet_manifest.json").write_text(json.dumps(manifest_doc,indent=2)+"\n")
preview=sheet.copy()
preview.thumbnail((1536,1536),Image.Resampling.LANCZOS)
preview.save(ROOT/"all_steps_contact_sheet_preview.jpg",quality=90)
print(json.dumps({"output":str(out),"pixels":[W,H],"bytes":out.stat().st_size,"tiles":len(manifest),"sha256":manifest_doc["image_sha256"]}))
