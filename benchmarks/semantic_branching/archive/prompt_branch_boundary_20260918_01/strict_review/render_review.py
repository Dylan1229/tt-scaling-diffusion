from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
SWEEP=ROOT.parent
PILOT=SWEEP.parent/"prompt_branch_step05_20260918_01"
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
def path(k):
    return PILOT/"target_dog" if k==0 else PILOT/"switch_step05" if k==5 else SWEEP/f"step_{k:03d}"
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--keys',nargs='+',type=int,default=[0,5,6,7,8,9,10]);keys=parser.parse_args().keys
for k in keys:
    assert (path(k)/'DONE').exists(), k
    arr=np.load(path(k)/"frames_uint8.npy",mmap_mode="r")
    # Full-resolution frames 0/40/80, stacked, for anatomy review.
    sheet=Image.new("RGB",(832,3*514),(18,23,31));d=ImageDraw.Draw(sheet)
    for row,t in enumerate([0,40,80]):
        d.text((10,row*514+6),f"{'All B dog reference' if k==0 else 'Switch after '+str(k)} | frame {t}/80",font=font,fill="white")
        sheet.paste(Image.fromarray(arr[t]),(0,row*514+34))
    sheet.save(ROOT/f"detail_{k:02d}.jpg",quality=96)
    # Every frame, in chronological order; no frame omitted.
    sheet=Image.new("RGB",(9*208,9*138+34),(18,23,31));d=ImageDraw.Draw(sheet)
    d.text((8,7),f"{'All B dog reference' if k==0 else 'Switch after '+str(k)} | all 81 frames (left-to-right, top-to-bottom)",font=font,fill="white")
    for t in range(81):
        x=(t%9)*208;y=34+(t//9)*138
        d.text((x+5,y+1),str(t),font=small,fill="white")
        sheet.paste(Image.fromarray(arr[t]).resize((208,120),Image.Resampling.LANCZOS),(x,y+18))
    sheet.save(ROOT/f"allframes_{k:02d}.jpg",quality=88)
    print(k,flush=True)
