import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/"prompt_branch_step05_20260918_01"
def data_dir(k):
    return PILOT/"target_dog" if k==0 else PILOT/"switch_step05" if k==5 else ROOT/"original_cat" if k==50 else ROOT/f"step_{k:03d}"
def check(k,ref):
    p=data_dir(k);m=json.loads((p/"metadata.json").read_text())
    assert m["status"]=="complete"
    if k==0:return {"step":0,"control":"all B"}
    prefix=[m["steps_log"][i]["latent_sha256"]==ref["steps_log"][i]["latent_sha256"] for i in range(k)]
    assert all(prefix),(k,prefix)
    hist=m["steps_log"][k-1]["scheduler_history_sha256"]==ref["steps_log"][k-1]["scheduler_history_sha256"]
    assert hist,k
    active=[s["conditioning_used"] for s in m["steps_log"]]
    assert active==["A"]*k+["B"]*(50-k),(k,active)
    return {"step":k,"identical_prefix":all(prefix),"matching_UniPC_history":hist,
            "sigma_at_switch":m["steps_log"][k-1]["sigma_after"],
            "different_latent_next_step":None if k==50 else m["steps_log"][k]["latent_sha256"]!=ref["steps_log"][k]["latent_sha256"]}
def main():
    a=argparse.ArgumentParser();a.add_argument("--steps",nargs="+",type=int,required=True);a.add_argument("--name",required=True);args=a.parse_args()
    ref=json.loads((ROOT/"original_cat/metadata.json").read_text())
    old=json.loads((PILOT/"original_cat/metadata.json").read_text())
    assert ref["status"]=="complete"
    assert ref["steps_log"][-1]["latent_sha256"]==old["steps_log"][-1]["latent_sha256"],"baseline drift"
    checks=[check(k,ref) for k in args.steps]
    (ROOT/f"{args.name}_validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    ids=[0,16,32,48,64,80]
    for start in range(0,len(args.steps),8):
        ks=args.steps[start:start+8]
        sheet=Image.new("RGB",(1248,len(ks)*152),(18,23,31));d=ImageDraw.Draw(sheet)
        for row,k in enumerate(ks):
            frames=np.load(data_dir(k)/"frames_uint8.npy",mmap_mode="r")
            label="All B: dog reference" if k==0 else "All A: original cat" if k==50 else f"Switch after step {k:02d}: {k} A + {50-k} B"
            d.text((8,row*152+7),label,font=font,fill="white")
            for col,i in enumerate(ids):
                sheet.paste(Image.fromarray(frames[i]).resize((208,120),Image.Resampling.LANCZOS),(col*208,row*152+32))
                d.rectangle((col*208+2,row*152+34,col*208+47,row*152+52),fill=(12,16,23))
                d.text((col*208+5,row*152+35),f"f{i:02d}",font=small,fill="white")
        path=ROOT/f"{args.name}_{start//8+1:02d}.jpg"
        sheet.save(path,quality=86)
        print(path)
    print(json.dumps(checks))
if __name__=="__main__":main()
