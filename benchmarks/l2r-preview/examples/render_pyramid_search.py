"""Local, synchronized review of decoder search results."""
import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from examples.compare_taehv import frames_u8, get_font, save_mp4
from l2r_preview.train_openvid import atomic_json

BASE = Path("/data/datasets/peihao/l2r-preview")
ROOT = BASE/"outputs/pyramid_search_20261006"
LABELS = {"compact": "Compact", "balanced_full": "Balanced", "quality_full": "Quality",
          "wide_full": "Wide", "taehv21": "TAEHV 21", "taehv81": "TAEHV full 81", "old_l2r": "Old L2R"}


def render(args):
    torch.set_num_threads(4)
    web = ROOT/"web"
    web.mkdir(exist_ok=True)
    test = {name: json.loads(p.read_text()) for name in LABELS if (p:=ROOT/"test"/(name+".json")).exists()}
    diffusion = {name: json.loads(p.read_text()) for name in LABELS if (p:=ROOT/"diffusion"/(name+".json")).exists()}
    winner = args.winner
    assert winner in test and winner in diffusion
    cases = []
    for r in test[winner]["visual_records"]:
        cases.append({"id":r["id"], "title":f'{r["topic"]} · {r["motion_band"]} motion',
                      "group":"test", "reference":"Original video"})
    for r in diffusion[winner]["cases"]:
        cases.append({"id":r["id"], "title":r["id"].replace("_"," ")+" · "+r["stage"],
                      "group":"diffusion", "reference":"Wan VAE reference"})
    cards, contact = [], []
    for case in cases:
        name, group = case["id"], case["group"]
        reports = test if group=="test" else diffusion
        order = [("source",case["reference"]),("taehv81","TAEHV full 81"),(winner,LABELS[winner])]
        decoded, arrays = {}, {}
        for model in ["source",*reports]:
            file = ROOT/group/"tensors"/(name+"_"+model+".pt")
            if not file.exists(): continue
            decoded[model] = torch.load(file, weights_only=True).float()
            arrays[model] = frames_u8(decoded[model])
            save_mp4(web/(name+"_"+model+".mp4"),arrays[model],4)
        scaled = {m:np.stack([np.array(Image.fromarray(f).resize((416,240),Image.Resampling.LANCZOS))
                             for f in arrays[m]]) for m,_ in order}
        combined = np.concatenate([scaled[m] for m,_ in order],axis=2)
        save_mp4(web/(name+"_comparison.mp4"),combined,4)
        Image.fromarray(combined[10]).save(web/(name+"_poster.jpg"),quality=94)
        panel = Image.new("RGB",(1248,844),"white")
        draw = ImageDraw.Draw(panel)
        draw.text((10,6),case["title"],font=get_font(15),fill="#152a3c")
        for row,frame in enumerate((0,10,20)):
            for col,(m,label) in enumerate(order):
                x,y=col*416,30+row*270
                draw.text((x+8,y),label+f" / {frame/4:g}s",font=get_font(14),fill="#405366")
                panel.paste(Image.fromarray(scaled[m][frame]),(x,y+22))
        panel.save(web/(name+"_frames.jpg"),quality=94)
        overview = Image.new("RGB",(1248,292),"white")
        overview.paste(Image.fromarray(combined[10]),(0,52))
        draw = ImageDraw.Draw(overview)
        draw.text((10,5),case["title"],font=get_font(15),fill="#152a3c")
        for col,(_,label) in enumerate(order):
            draw.text((col*416+10,29),label,font=get_font(14),fill="#405366")
        contact.append(overview)
        links = " · ".join(f'<a href="{name}_{m}.mp4">{html.escape(LABELS.get(m,case["reference"]))}</a>' for m in arrays)
        labels = "".join("<span>"+html.escape(label)+"</span>" for _,label in order)
        row = next(r for r in reports[winner]["records" if group=="test" else "cases"] if r["id"]==name)
        score = row["full_psnr_db"]
        cards.append(f'<section data-group="{group}"><h2>{html.escape(case["title"])}</h2>'
                     f'<p>{LABELS[winner]}: {score:.2f} dB vs {case["reference"].lower()} · 21 frames / 4 fps</p>'
                     f'<div class="labels">{labels}</div><video controls loop muted playsinline preload="metadata" '
                     f'src="{name}_comparison.mp4" poster="{name}_poster.jpg"></video>'
                     f'<p class="links">Full resolution: {links}</p>'
                     f'<details><summary>First / middle / last frames</summary><img loading="lazy" src="{name}_frames.jpg"></details></section>')
        print(json.dumps({"rendered":name}),flush=True)
    sheet=Image.new("RGB",(1248,sum(im.height for im in contact)),"white")
    offset=0
    for im in contact:
        sheet.paste(im,(0,offset));offset+=im.height
    sheet.save(web/"contact_sheet.jpg",quality=94)
    rows=[]
    for name,report in sorted(test.items(),key=lambda x:x[1]["median_ms"]):
        rows.append(f'<tr><td>{LABELS[name]}</td><td>{report["parameters"]/1e6:.3f}M</td>'
                    f'<td>{report["median_ms"]:.2f} ms</td><td>{report["peak_allocated_gib"]:.2f} GiB</td>'
                    f'<td>{report["full_psnr_db"]:.2f}</td><td>{report["ssim"]:.3f}</td>'
                    f'<td>{report["computed_frames"]}</td></tr>')
    validation = {p.stem:json.loads(p.read_text()) for p in (ROOT/"validation").glob("*.json")}
    selection = json.loads((ROOT/"selection.json").read_text()) if (ROOT/"selection.json").exists() else {}
    result = {"winner":winner,"selection":selection,"test":test,"diffusion":diffusion,"validation":validation,
              "notes":["Quality is raw RGB before MP4 encoding; 250 held-out clips.",
                       "PSNR aggregates full-resolution MSE over every aligned frame; SSIM uses frames 0/10/20.",
                       "Same GPU, BF16 batch1; synchronized forward only; 5 warmups and 10 repeats on each of 3 inputs.",
                       "TAEHV81 computes 81 frames before taking 21 endpoints; other decoders compute21.",
                       "Diffusion cases are a reused qualitative check; no failure detector or generation search was evaluated."]}
    atomic_json(web/"report.json",result)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fast decoder search</title><style>
body{margin:0;background:#eff3f7;color:#152a3c;font:16px/1.5 system-ui}main{max-width:1280px;margin:auto;padding:28px 20px}
h1{margin:0;font-size:32px}h2{font-size:19px;margin:0}p{color:#506174}
section{background:#fff;padding:18px;border-radius:12px;margin:24px 0}
video,img{display:block;width:100%;height:auto;border-radius:5px}.labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;font-weight:650;margin:12px 0}
table{background:white;border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #dae2ec}
.note{background:#e0ebf6;border-left:4px solid #3b70aa;padding:12px 16px}.links{font-size:13px}a{color:#235ea4}
summary{cursor:pointer}button{background:white;border:1px solid #bbc8d8;border-radius:7px;padding:8px 13px;cursor:pointer}
button.active{background:#152a3c;color:white}.scroll{overflow:auto}
</style><main><h1>Fast decoder search</h1>'''
    page+=f'<p>Four network sizes · crop and full-frame training · 4,500 training / 250 validation / 250 test clips.</p><p class="note">Featured candidate: <b>{LABELS[winner]}</b>. Selection uses validation data. The table below reports the frozen test set.</p>'
    page+='<div class="scroll"><table><tr><th>Decoder</th><th>Params</th><th>Decode</th><th>Peak GPU</th><th>Test PSNR</th><th>SSIM</th><th>Frames computed</th></tr>'+"".join(rows)+"</table></div>"
    page+='<p>All previews show the same 21 frame positions at 832×480. Timing excludes model loading, normalization, transfers and video encoding. TAEHV full 81 computes additional frames before selection.</p>'
    page+='<p>Videos are local. The three columns share one MP4 for synchronized playback; individual full-resolution outputs are linked below.</p>'
    page+='<nav><button class="active" data-filter="all">All 12</button> <button data-filter="test">Held-out real videos</button> <button data-filter="diffusion">Diffusion previews</button></nav>'
    page+="".join(cards)
    page+='<p><a href="contact_sheet.jpg">Contact sheet</a> · <a href="report.json">Full results</a> · <a href="TAEHV_LICENSE.txt">TAEHV license</a></p></main>'
    page+='''<script>
document.querySelectorAll('button[data-filter]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));
 document.querySelectorAll('section[data-group]').forEach(s=>{
  s.hidden=b.dataset.filter!=='all'&&s.dataset.group!==b.dataset.filter;
  if(s.hidden)s.querySelector('video').pause();
 });});
document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>{
 document.querySelectorAll('video').forEach(o=>{if(o!==v)o.pause()});
}));
</script></html>'''
    (web/"index.html").write_text(page)
    shutil.copy2(BASE/"outputs/taehv_20261002/upstream/LICENSE",web/"TAEHV_LICENSE.txt")
    print(json.dumps({"web":str(web),"videos":len(list(web.glob("*.mp4")))}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner",required=True)
    render(parser.parse_args())
