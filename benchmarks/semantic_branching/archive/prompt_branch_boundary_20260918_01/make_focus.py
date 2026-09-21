import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument("--steps",nargs="+",type=int,required=True);a.add_argument("--name",required=True);args=a.parse_args()
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
for start in range(0,len(args.steps),6):
    ks=args.steps[start:start+6]
    sheet=Image.new("RGB",(832,len(ks)*276),(18,23,31));d=ImageDraw.Draw(sheet)
    for row,k in enumerate(ks):
        frames=np.load(ROOT/f"step_{k:03d}"/"frames_uint8.npy",mmap_mode="r")
        d.text((8,row*276+7),f"Switch after step {k:02d} | frame 0 (left), frame 16 (right)",font=font,fill="white")
        for col,i in enumerate([0,16]):
            sheet.paste(Image.fromarray(frames[i]).resize((416,240),Image.Resampling.LANCZOS),(col*416,row*276+36))
    path=ROOT/f"{args.name}_{start//6+1:02d}.jpg";sheet.save(path,quality=94);print(path)
