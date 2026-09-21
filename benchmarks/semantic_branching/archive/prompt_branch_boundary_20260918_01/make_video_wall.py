from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/"all_steps_contact_sheet_manifest.json").read_text())
tiles=manifest["tiles"]
assert [x["k"] for x in tiles]==list(range(51))
COLS,ROWS,TW,TH,LABEL,GAP,MARGIN,HEADER=6,9,416,240,28,6,14,80
W=2*MARGIN+COLS*TW+(COLS-1)*GAP
H=HEADER+ROWS*(TH+LABEL)+(ROWS-1)*GAP+MARGIN
assert W%2==0 and H%2==0
fontpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
title=ImageFont.truetype(fontpath,28)
sub=ImageFont.truetype(fontpath,16)
label=ImageFont.truetype(fontpath,15)
small=ImageFont.truetype(fontpath,14)
base=Image.new("RGB",(W,H),(16,22,31));d=ImageDraw.Draw(base)
d.text((MARGIN,10),"CAT → DOG PROMPT BRANCHING · ALL 51 OUTPUTS",font=title,fill=(245,248,252))
d.text((MARGIN,48),"Same case / seed 0 / synchronized at 24 fps / read left-to-right, top-to-bottom",font=sub,fill=(178,191,211))
sources=[]
for tile in tiles:
    k=tile["k"];p=Path(tile["source"])
    meta=json.loads((p/"metadata.json").read_text())
    assert meta["status"]=="complete" and meta["fps"]==24
    arr=np.load(p/"frames_uint8.npy",mmap_mode="r")
    assert arr.shape==(81,480,832,3) and arr.dtype==np.uint8
    row,col=divmod(k,COLS);x=MARGIN+col*(TW+GAP);y=HEADER+row*(TH+LABEL+GAP)
    d.rectangle((x,y,x+TW-1,y+LABEL-1),fill=(39,55,74) if k in (0,50) else (30,39,53))
    text="00 · ALL DOG — control" if k==0 else "50 · ALL CAT — control" if k==50 else f"{k:02d} · switch after {k} steps | {k} A + {50-k} B"
    assert d.textlength(text,font=label)<TW-12
    d.text((x+6,y+5),text,font=label,fill=(247,249,252))
    sources.append((k,x,y+LABEL,arr))
lx=MARGIN+3*(TW+GAP);ly=HEADER+8*(TH+LABEL+GAP)
lines=["A = orange tabby cat","B = golden retriever dog","01–49: prompt switches after k completed denoising steps.","00 / 50: all-dog / all-cat references.","All 81 original video frames, synchronized; no frame omitted.","51 outputs / same seed / no quality labels."]
for i,line in enumerate(lines):
    d.text((lx+10,ly+12+i*29),line,font=small,fill=(181,194,214))
out=ROOT/"prompt_branching_all_steps_00-50_seed0_video_wall_2026-09-18.mp4"
tmp=out.with_suffix(".partial.mp4")
assert not out.exists()
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y",
     "-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r","24","-i","pipe:0",
     "-an","-c:v","libx264","-preset","medium","-crf","18","-profile:v","high",
     "-level:v","5.1","-pix_fmt","yuv420p","-threads","8","-movflags","+faststart",str(tmp)]
started=time.time()
with (ROOT/"video_wall_encode.log").open("wb") as error:
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=error)
    try:
        for t in range(81):
            frame=base.copy()
            for k,x,y,arr in sources:
                frame.paste(Image.fromarray(arr[t]).resize((TW,TH),Image.Resampling.LANCZOS),(x,y))
            fd=ImageDraw.Draw(frame)
            fd.text((lx+10,ly+206),f"Frame {t:02d}/80    Time {t/24:.2f} s",font=sub,fill=(234,241,251))
            fd.rectangle((lx+10,ly+239,lx+1100,ly+247),fill=(40,53,72))
            fd.rectangle((lx+10,ly+239,lx+10+int(1090*t/80),ly+247),fill=(101,179,240))
            proc.stdin.write(frame.tobytes())
            if t in (0,40,80):frame.save(ROOT/f"video_wall_preview_{t:02d}.jpg",quality=88)
            if t%20==0:print(json.dumps({"frame":t,"elapsed_seconds":time.time()-started}),flush=True)
        proc.stdin.close()
        code=proc.wait()
        assert code==0,code
    except BaseException:
        proc.kill();proc.wait()
        raise
tmp.replace(out)
count,duration=imageio_ffmpeg.count_frames_and_secs(str(out))
assert count==81 and abs(duration-81/24)<0.05,(count,duration)
result={"output":str(out),"width":W,"height":H,"frames":count,"fps":24,"duration_seconds":duration,
        "tiles":51,"tile_size":[TW,TH],"codec":"h264","pixel_format":"yuv420p",
        "bytes":out.stat().st_size,"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "render_seconds":time.time()-started}
(ROOT/"video_wall_manifest.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
