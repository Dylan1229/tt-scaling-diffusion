import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v3 as iio

ROOT = Path(__file__).resolve().parent
MODES = ["original_cat", "switch_step05", "target_dog"]
LABELS = ["A: cat / all 50 steps", "A -> B: switch after step 5", "B: dog / all 50 steps"]
metas = [json.loads((ROOT/m/"metadata.json").read_text()) for m in MODES]
for mode,meta in zip(MODES,metas):
    assert meta["status"] == "complete", mode
    assert len(meta["steps_log"]) == 50, mode
for meta in metas:
    meta["scheduler_config"]["_use_default_values"] = sorted(meta["scheduler_config"].get("_use_default_values", []))
for key in ["prompt_A_embedding_sha256", "prompt_B_embedding_sha256", "negative_embedding_sha256", "scheduler_config"]:
    assert metas[0][key] == metas[1][key] == metas[2][key], key
prefix_equal = [metas[0]["steps_log"][i]["latent_sha256"] == metas[1]["steps_log"][i]["latent_sha256"] for i in range(5)]
assert all(prefix_equal), prefix_equal
assert metas[0]["steps_log"][4]["scheduler_history_sha256"] == metas[1]["steps_log"][4]["scheduler_history_sha256"]
assert metas[0]["steps_log"][5]["latent_sha256"] != metas[1]["steps_log"][5]["latent_sha256"]
assert [s["conditioning_used"] for s in metas[1]["steps_log"]] == ["A"]*5+["B"]*45
validation = {
    "same_seed": all(m["seed"]==0 for m in metas),
    "identical_A_and_branch_prefix_steps_1_to_5": prefix_equal,
    "identical_UniPC_model_output_history_at_branch": True,
    "different_latent_after_step_6": True,
    "branch_conditioning": "5 steps A, 45 steps B",
    "sigma_at_branch": metas[1]["steps_log"][4]["sigma_after"],
    "num_cases": 1, "num_seeds": 1,
    "interpretation": "Execution and intervention checks only. Semantic effect requires visual inspection.",
    "compute": "Three 50-step runs (150 steps total); A prefix replayed for exact-state verification.",
}
(ROOT/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
frames = [np.load(ROOT/m/"frames_uint8.npy", mmap_mode="r") for m in MODES]
font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_file,18)
small = ImageFont.truetype(font_file,14)
indices = [0,16,32,48,64,80]
w,h,bar = 208,120,38
sheet = Image.new("RGB",(w*6,3*(h+bar)),(20,24,32))
draw = ImageDraw.Draw(sheet)
for row,(video,label) in enumerate(zip(frames,LABELS)):
    top=row*(h+bar)
    draw.text((10,top+9),label,fill=(240,244,250),font=font)
    for col,k in enumerate(indices):
        tile=Image.fromarray(video[k]).resize((w,h),Image.Resampling.LANCZOS)
        sheet.paste(tile,(col*w,top+bar))
        draw.rectangle((col*w+3,top+bar+3,col*w+65,top+bar+22),fill=(12,15,20))
        draw.text((col*w+6,top+bar+4),f"f{k:02d}",font=small,fill="white")
sheet.save(ROOT/"comparison.png")

w,h,bar=416,240,48
merged=[]
for k in range(81):
    canvas=Image.new("RGB",(w*3,h+bar),(20,24,32))
    d=ImageDraw.Draw(canvas)
    for col,(video,label) in enumerate(zip(frames,LABELS)):
        canvas.paste(Image.fromarray(video[k]).resize((w,h),Image.Resampling.LANCZOS),(col*w,bar))
        d.text((col*w+10,10),label,font=font,fill=(240,244,250))
        d.text((col*w+10,31),f"seed 0 | 50 steps | frame {k:02d}/80",font=small,fill=(168,183,201))
    merged.append(np.asarray(canvas))
iio.imwrite(ROOT/"comparison.mp4",np.stack(merged),fps=24,codec="libx264")
print(json.dumps(validation,indent=2))
print(ROOT/"comparison.mp4")
