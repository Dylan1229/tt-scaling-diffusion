"""Standalone compiled-decoder gallery, leaving the earlier report intact."""
import json,math,statistics,subprocess,re
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from l2r_preview.openvid import sha256
from l2r_preview.train_openvid import atomic_json
from examples.evaluate_compiled_unseen import NEW,OLD,preserve_old


def aggregate(rows,name):
    x=[r[name] for r in rows]
    return {'cases':len(x),'pooled_psnr_db':-10*math.log10(statistics.mean(a['full_mse_01'] for a in x)),
      'mean_ssim':statistics.mean(a['ssim'] for a in x),'mean_temporal_delta_mae':statistics.mean(a['temporal_delta_mae'] for a in x),
      'median_case_psnr_db':statistics.median(a['full_psnr_db'] for a in x),'worst_case_psnr_db':min(a['full_psnr_db'] for a in x)}


def contact(rows,path):
    w,h=1248,276
    sheet=Image.new('RGB',(w,h*len(rows)+30),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for x,t in [(12,'Native VAE'),(428,'Original Quality / eager'),(844,'Quality / compiled')]:draw.text((x,6),t,font=font,fill='#354a5c')
    for i,r in enumerate(rows):
        y=30+i*h
        text=f"{r['id']} | {r['prompt']} | compiled {r['compiled']['full_psnr_db']:.2f}dB"
        draw.text((8,y+5),text,font=font,fill='#142c3d')
        sheet.paste(Image.open(NEW/r['media']['poster']),(0,y+32))
    sheet.save(path,quality=93)


def main():
    plan=json.loads((NEW/'plan.json').read_text());ev=json.loads((NEW/'evaluation.json').read_text())
    rows=ev['records'];assert ev['completed'] and len(rows)==len(plan['cases'])==165
    assert [r['id'] for r in rows]==[r['id'] for r in plan['cases']]
    worst=sorted(rows,key=lambda r:r['compiled']['full_psnr_db'])[:6]
    changed=sorted(rows,key=lambda r:r['compiled_minus_eager']['mean_abs_rgb'],reverse=True)[:6]
    reps=[next(r for r in rows if r['prompt_id']==p and r['seed']==5 and r['step']==10) for p in sorted({r['prompt_id'] for r in rows})]
    report={'completed':True,'cases':len(rows),'trajectories':plan['trajectories'],'prompts':plan['prompts'],
      'overall':{m:aggregate(rows,m) for m in ('eager','compiled')},
      'by_step':{str(s):{m:aggregate([r for r in rows if r['step']==s],m) for m in ('eager','compiled')} for s in plan['steps']},
      'paired':{'mean_case_psnr_change_db':statistics.mean(r['compiled']['full_psnr_db']-r['eager']['full_psnr_db'] for r in rows),
                'mean_ssim_change':statistics.mean(r['compiled']['ssim']-r['eager']['ssim'] for r in rows),
                'mean_abs_rgb_difference':statistics.mean(r['compiled_minus_eager']['mean_abs_rgb'] for r in rows),
                'max_abs_rgb_difference':max(r['compiled_minus_eager']['max_abs_rgb'] for r in rows)},
      'median_ms':ev['median_ms'],'compile_and_first_call_seconds':ev['compile_and_first_call_seconds'],
      'worst_compiled_psnr':[r['id'] for r in worst],'largest_compiled_vs_eager_difference':[r['id'] for r in changed],
      'visual_representatives':[r['id'] for r in reps],'old_report_preservation':preserve_old(),'limits':plan['limits'],
      'evaluation_sha256':sha256(NEW/'evaluation.json'),'plan_sha256':sha256(NEW/'plan.json'),'records':rows}
    atomic_json(NEW/'report.json',report)
    contact(reps,NEW/'all_prompts_step10.jpg');contact(worst,NEW/'worst6_compiled.jpg');contact(changed,NEW/'largest6_execution_difference.jpg')
    table=''.join(f'<tr><td>{label}</td><td>{report["overall"][m]["pooled_psnr_db"]:.3f}</td><td>{report["overall"][m]["mean_ssim"]:.4f}</td><td>{ev["median_ms"][m]:.2f} ms</td></tr>' for m,label in [('eager','Original Quality · eager'),('compiled','Quality · compiled')])
    steps=''.join(f'<tr><td>{s} / 50</td><td>{report["by_step"][str(s)]["eager"]["pooled_psnr_db"]:.3f} / {report["by_step"][str(s)]["eager"]["mean_ssim"]:.4f}</td><td>{report["by_step"][str(s)]["compiled"]["pooled_psnr_db"]:.3f} / {report["by_step"][str(s)]["compiled"]["mean_ssim"]:.4f}</td></tr>' for s in plan['steps'])
    data=[{'id':r['id'],'prompt':r['prompt'],'prompt_id':r['prompt_id'],'step':r['step'],'seed':r['seed'],
        'eager_psnr':r['eager']['full_psnr_db'],'compiled_psnr':r['compiled']['full_psnr_db'],'compiled_ssim':r['compiled']['ssim'],
        'difference':r['compiled_minus_eager']['mean_abs_rgb'],**r['media']} for r in rows]
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compiled Quality — new unseen report</title>
<style>body{margin:0;background:#eef3f2;color:#132d39;font:16px/1.5 system-ui}main{max-width:1280px;margin:auto;padding:24px}h1{margin:0;font-size:30px}h2{font-size:21px}section{background:#fff;padding:22px;border-radius:12px;margin:20px 0}p{color:#506773}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce5e7}.badge{display:inline-block;background:#d5eee8;color:#15554b;padding:4px 9px;border-radius:6px;font-size:14px}select,button{font:inherit;padding:7px;margin:4px;max-width:100%}video{width:100%;background:#19232c;border-radius:6px}.labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;font-size:14px;margin:12px 0}.labels span:last-child{font-weight:700;color:#13765e}a{color:#136a8b}.small{font-size:14px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cards button{display:block;background:white;border:1px solid #dce5e7;text-align:left;cursor:pointer}.cards img{width:100%}.controls{display:flex;flex-wrap:wrap;align-items:center}@media(max-width:700px){main{padding:12px}section{padding:12px}.cards{grid-template-columns:1fr}}</style>
<main><span class="badge">Separate report · newly rendered compiled outputs</span><h1>Quality compiled / 2.42 ms version</h1><p>165 previews · 33 trajectories · 11 unseen prompts · same frozen set as the original report.</p><p><a href="../l2r_unseen_20261007/index.html" target="_blank">Open the preserved original Quality report</a></p>
<section><h2>Paired reconstruction and measured timing</h2><p>Both versions receive the same predicted-clean latent. Native Wan VAE is the reconstruction reference. Same network and BF16 parameter values; different execution.</p><table><tr><th>Decoder</th><th>PSNR ↑</th><th>SSIM ↑</th><th>Fresh steady-state timing</th></tr>TABLE</table><p class="small">21×480×832 output, BF16, batch 1, RTX PRO 6000 Blackwell. Timing alternates both methods across 3 fixed inputs with 10 warmups and 30 repeats each. Cold compile/I/O/transfers/video encoding are excluded.</p><p class="small">COMPILE_NOTE</p><p class="small">DELTA_NOTE</p><table><tr><th>Denoising step</th><th>Eager PSNR / SSIM</th><th>Compiled PSNR / SSIM</th></tr>STEPS</table></section>
<section id="viewer"><h2>All 165 compiled videos</h2><div class="controls"><select id="prompt"><option value="all">All prompts</option></select><select id="step"><option value="all">All stages</option><option>5</option><option>10</option><option>20</option><option>30</option><option>50</option></select><select id="order"><option value="case">Prompt / seed / step</option><option value="worst">Lowest compiled PSNR first</option><option value="difference">Largest eager / compiled difference</option></select></div><select id="case"></select><button id="prev">Previous</button><button id="next">Next</button><p id="caption"></p><div class="labels"><span>Native Wan VAE</span><span>Original Quality / 5.37 ms version</span><span>Compiled Quality / 2.42 ms version</span></div><video id="video" controls muted loop playsinline preload="metadata"></video><p class="small">New full-resolution exports, synchronized in one MP4. 21 frames at 4 fps. Fullscreen to inspect 832×480 detail per column. Labels identify the earlier benchmark versions; fresh timing is reported above.</p></section>
<section><h2>Largest eager / compiled differences</h2><p>Ranked by raw mean absolute RGB difference, before MP4 encoding. Click to inspect.</p><div class="cards" id="differences"></div></section>
<section><h2>Lowest compiled reconstruction scores</h2><p>These expose limitations shared with or added to the original decoder.</p><div class="cards" id="worst"></div></section>
<section><h2>Scope and files</h2><p>No retraining or weight selection. Every compiled video in this page was generated in this new run. The earlier report and videos are preserved and hash-verified.</p><p>This repeats the same held-out set; it is not 165 new independent videos. Results cover one model/scheduler/resolution and 11 short prompts. Fidelity to the native VAE does not establish failure-detection accuracy or semantic correctness.</p><p><a href="report.json">Full metrics</a> · <a href="plan.json">Frozen plan</a> · <a href="all_prompts_step10.jpg">Every prompt at step 10</a> · <a href="largest6_execution_difference.jpg">Largest-difference contact sheet</a> · <a href="worst6_compiled.jpg">Lowest-PSNR contact sheet</a> · <a href="README.txt">Notes</a></p></section></main>
<script>const rows=ROWS;const $=id=>document.getElementById(id);let shown=[];for(const p of [...new Set(rows.map(r=>r.prompt_id))]){const o=document.createElement('option');o.value=p;o.textContent=p+' — '+rows.find(r=>r.prompt_id===p).prompt;$('prompt').append(o)}function load(id){const r=rows.find(x=>x.id===id);if(!r)return;$('case').value=id;$('caption').textContent=r.id+' · '+r.prompt+' · Compiled '+r.compiled_psnr.toFixed(3)+'dB / SSIM '+r.compiled_ssim.toFixed(4)+' · Eager '+r.eager_psnr.toFixed(3)+'dB';$('video').src=r.video;$('video').poster=r.poster}function filter(){shown=rows.filter(r=>($('prompt').value==='all'||r.prompt_id===$('prompt').value)&&($('step').value==='all'||r.step===Number($('step').value)));if($('order').value==='worst')shown.sort((a,b)=>a.compiled_psnr-b.compiled_psnr);if($('order').value==='difference')shown.sort((a,b)=>b.difference-a.difference);$('case').replaceChildren();for(const r of shown){const o=document.createElement('option');o.value=r.id;o.textContent=r.id+' · '+r.compiled_psnr.toFixed(2)+'dB';$('case').append(o)}if(shown.length)load(shown[0].id)}for(const id of ['prompt','step','order'])$(id).onchange=filter;$('case').onchange=()=>load($('case').value);function advance(n){const i=shown.findIndex(r=>r.id===$('case').value);load(shown[(i+n+shown.length)%shown.length].id)}$('prev').onclick=()=>advance(-1);$('next').onclick=()=>advance(1);function cards(id,items){for(const r of items){const b=document.createElement('button'),img=document.createElement('img'),label=document.createElement('div');img.src=r.poster;img.loading='lazy';label.textContent=r.id+' · Compiled '+r.compiled_psnr.toFixed(2)+'dB · RGB difference '+r.difference.toFixed(6);b.append(img,label);b.onclick=()=>{$('prompt').value='all';$('step').value='all';filter();load(r.id);$('viewer').scrollIntoView({behavior:'smooth'})};$(id).append(b)}}cards('differences',[...rows].sort((a,b)=>b.difference-a.difference).slice(0,6));cards('worst',[...rows].sort((a,b)=>a.compiled_psnr-b.compiled_psnr).slice(0,6));filter();</script></html>'''
    compile_note=f"First compile and invocation in this task's new cache took {ev['compile_and_first_call_seconds']:.2f}s. Reuse the decoder in a persistent process."
    delta_note=f"Compiled minus eager: mean per-case PSNR {report['paired']['mean_case_psnr_change_db']:+.5f}dB; SSIM {report['paired']['mean_ssim_change']:+.6f}; mean absolute RGB difference {report['paired']['mean_abs_rgb_difference']:.7f} on [0,1]. Outputs are not bit-identical."
    page=page.replace('TABLE',table).replace('STEPS',steps).replace('COMPILE_NOTE',compile_note).replace('DELTA_NOTE',delta_note).replace('ROWS',json.dumps(data).replace('</','<\\/'))
    (NEW/'index.html').write_text(page)
    (NEW/'README.txt').write_text('''Compiled Quality: separate unseen visual report — 2026-10-07

Open index.html. Every video is new to this report.
Left: native Wan VAE. Middle: original Quality eager. Right: compiled Quality.
All 3 columns use the same latent and are frame-synchronized in one MP4.
Same 165 held-out latents from 33 trajectories / 11 prompts / 3 seeds / 5 steps.
No new training, no weight selection, no replacement of the earlier report.
Original report is preserved at ../l2r_unseen_20261007/index.html on this Mac.
Version labels 5.37 ms and 2.42 ms identify prior results; see the table for fresh timing.
PSNR uses all 21 frames; SSIM samples frames 0, 10, 20. Raw RGB metrics precede video encoding.
The shared native reference is cached FP16 output from the prior native VAE decoding.
Cold compilation is separate from warmed decoding latency.
Raw native reference tensors stay in the original server directory and are reused read-only.
Source, hashes, all case-level results and preservation checks are included.
''')
    verified=[]
    for r in rows:
        p=NEW/r['media']['video']
        assert sha256(p)==r['media']['video_sha256']
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=60)
        verified.append({'id':r['id'],'fully_decoded':True,'sha256':sha256(p)})
    atomic_json(NEW/'media_validation.json',{'videos':len(verified),'all_fully_decoded':True,'records':verified})
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','by_step')},indent=2))

if __name__=='__main__':main()
