import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v3 as iio
from make_sheets import data_dir
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument("--steps",nargs="+",type=int,required=True);p.add_argument("--name",required=True);args=p.parse_args()
frames=[np.load(data_dir(k)/"frames_uint8.npy",mmap_mode="r") for k in args.steps]
cols=3;rows=(len(frames)+cols-1)//cols;w,h,bar=416,240,40
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
output=[]
for t in range(81):
    canvas=Image.new("RGB",(cols*w,rows*(h+bar)),(18,23,31));d=ImageDraw.Draw(canvas)
    for j,(k,video) in enumerate(zip(args.steps,frames)):
        x=(j%cols)*w;y=(j//cols)*(h+bar)
        label="All A: original cat" if k==50 else "All B: dog reference" if k==0 else f"Switch after step {k:02d} / {k} A + {50-k} B"
        d.text((x+8,y+5),label,font=font,fill="white")
        d.text((x+8,y+25),"Same seed 0 | 50 denoising steps",font=small,fill=(170,185,205))
        canvas.paste(Image.fromarray(video[t]).resize((w,h),Image.Resampling.LANCZOS),(x,y+bar))
    output.append(np.asarray(canvas))
    if t==16:canvas.save(ROOT/f"{args.name}.jpg",quality=90)
iio.imwrite(ROOT/f"{args.name}.mp4",np.stack(output),fps=24,codec="libx264")
print(ROOT/f"{args.name}.mp4")
