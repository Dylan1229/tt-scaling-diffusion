"""Aggregate a frozen unseen-latent evaluation and render its full local gallery."""
from __future__ import annotations
import json, math, statistics, hashlib, html, subprocess
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from l2r_preview.openvid import sha256
from l2r_preview.train_openvid import atomic_json
from examples.evaluate_unseen import DEFAULT,STEPS

MODELS=('quality_full','taehv21','taehv81')


def summary(records,model):
    rows=[r['metrics'][model] for r in records]
    ps=np.array([r['full_psnr_db'] for r in rows])
    mse=statistics.mean(r['full_mse_01'] for r in rows)
    return {'cases':len(rows),'pooled_psnr_db':-10*math.log10(mse),'mean_case_psnr_db':float(ps.mean()),
        'median_case_psnr_db':float(np.median(ps)),'p10_case_psnr_db':float(np.quantile(ps,.1)),
        'worst_case_psnr_db':float(ps.min()),'mean_ssim':statistics.mean(r['ssim'] for r in rows),
        'mean_temporal_delta_mae':statistics.mean(r['temporal_delta_mae'] for r in rows)}


def paired(records,baseline):
    prompts=sorted({r['prompt_id'] for r in records})
    out={}
    for metric in ('full_psnr_db','ssim','temporal_delta_mae'):
        values=np.array([statistics.mean(r['metrics']['quality_full'][metric]-r['metrics'][baseline][metric]
                       for r in records if r['prompt_id']==p) for p in prompts])
        rng=np.random.default_rng(20261007)
        boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
        out[metric]={'mean_quality_minus_baseline':float(values.mean()),
            'prompt_cluster_bootstrap95':[float(v) for v in np.quantile(boot,[.025,.975])],
            'clusters':len(prompts)}
    return out


def main():
    root=DEFAULT
    plan=json.loads((root/'plan.json').read_text())
    records=[]
    for case in plan['cases']:
        path=root/'records'/f"{case['id']}.json"
        row=json.loads(path.read_text())
        assert row['source_sha256']==case['sha256']
        assert row['quality_checkpoint_sha256']==plan['checkpoints']['quality_full']['sha256']
        assert set(row['metrics'])==set(MODELS)
        for m in MODELS:
            assert all(math.isfinite(row['metrics'][m][k]) for k in ('full_mse_01','full_psnr_db','ssim','temporal_delta_mae'))
        records.append(row)
    assert len(records)==165
    assert sha256(plan['checkpoints']['quality_full']['checkpoint'])==plan['checkpoints']['quality_full']['sha256']
    worst=sorted(records,key=lambda r:r['metrics']['quality_full']['full_psnr_db'])[:12]
    representatives=sorted(records,key=lambda r:hashlib.sha256(('unseen-display-33:'+r['id']).encode()).hexdigest())[:12]
    report={'plan_sha256':sha256(root/'plan.json'),'checkpoint_sha256':plan['checkpoints']['quality_full']['sha256'],
        'cases':len(records),'trajectories':len({r['trajectory'] for r in records}),'prompts':len({r['prompt_id'] for r in records}),
        'overall':{m:summary(records,m) for m in MODELS},
        'by_step':{str(s):{m:summary([r for r in records if r['step']==s],m) for m in MODELS} for s in STEPS},
        'by_prompt':{p:{m:summary([r for r in records if r['prompt_id']==p],m) for m in MODELS} for p in sorted({r['prompt_id'] for r in records})},
        'paired_quality_minus_baseline':{m:paired(records,m) for m in MODELS if m!='quality_full'},
        'worst12_by_quality_psnr':[r['id'] for r in worst],'representative12_by_hash':[r['id'] for r in representatives],
        'limits':plan['limits'],'records':records,'completed':True}
    atomic_json(root/'report.json',report)
    # Show one poster from every prompt at an unchanged seed and step.
    cards=[next(r for r in records if r['prompt_id']==p and r['seed']==5 and r['step']==10) for p in sorted({r['prompt_id'] for r in records})]
    def contact(rows,name):
        canvas=Image.new('RGB',(1248,272*len(rows)),'white')
        draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        for i,r in enumerate(rows):
            draw.text((8,272*i+5),f"{r['id']} | {r['prompt']} | Quality {r['metrics']['quality_full']['full_psnr_db']:.2f}dB",fill='#17212e',font=font)
            canvas.paste(Image.open(root/r['media']['poster']),(0,272*i+32))
        canvas.save(root/name,quality=92)
    contact(cards,'all_prompts_step10.jpg')
    contact(worst[:6],'worst6.jpg')
    table=''.join(f'<tr><td>{m}</td><td>{report["overall"][m]["pooled_psnr_db"]:.2f}</td><td>{report["overall"][m]["mean_ssim"]:.3f}</td><td>{report["overall"][m]["p10_case_psnr_db"]:.2f}</td></tr>' for m in MODELS)
    stage=''.join('<tr><td>'+str(s)+'</td>'+''.join(f'<td>{report["by_step"][str(s)][m]["pooled_psnr_db"]:.2f} / {report["by_step"][str(s)][m]["mean_ssim"]:.3f}</td>' for m in MODELS)+'</tr>' for s in STEPS)
    preview_rows=[{'id':r['id'],'prompt':r['prompt'],'prompt_id':r['prompt_id'],'seed':r['seed'],'step':r['step'],
        'psnr':r['metrics']['quality_full']['full_psnr_db'],'ssim':r['metrics']['quality_full']['ssim'],**r['media']} for r in records]
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quality decoder — unseen diffusion test</title>
<style>body{margin:0;background:#eef2f5;color:#152334;font:16px/1.5 system-ui}main{max-width:1280px;margin:auto;padding:24px}h1{margin:0;font-size:30px}h2{font-size:21px}section{background:white;padding:22px;border-radius:12px;margin:20px 0}p{color:#4f6174}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #dbe3ea;padding:8px}select,button{font:inherit;padding:7px;max-width:100%;margin:4px}video{width:100%;background:#19232c;border-radius:6px}.labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;font-size:14px;margin:10px 0}a{color:#185ea6}.small{font-size:14px}.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.cards img{width:100%}.cards button{text-align:left;border:1px solid #dbe3ea;background:white;cursor:pointer}@media(max-width:650px){.cards{grid-template-columns:1fr}main{padding:12px}section{padding:12px}}</style>
<main><h1>Quality decoder: unseen diffusion latents</h1><p>Frozen checkpoint · 11 unseen prompts × 3 seeds × 5 steps = 165 previews from 33 trajectories.</p>
<section><h2>Reconstruction against the native VAE</h2><p>All models decode the same predicted-clean latent. 21 aligned frames at 480×832; raw RGB metrics before video compression. Higher is better.</p>
<table><tr><th>Decoder</th><th>Pooled PSNR (dB)</th><th>Mean SSIM</th><th>10th-percentile PSNR</th></tr>TABLE</table>
<p class="small">PSNR uses all 21 frames; SSIM samples frames 0 / 10 / 20. TAEHV81 generates all 81 frames then selects matching endpoints. These scores measure decoder fidelity, not whether the generated scene is correct.</p>
<table><tr><th>Denoising step / 50</th><th>Quality PSNR / SSIM</th><th>TAEHV21</th><th>TAEHV81</th></tr>STAGE</table></section>
<section id="viewer"><h2>Every case</h2><div><select id="prompt"><option value="all">All prompts</option></select><select id="step"><option value="all">All steps</option><option>5</option><option>10</option><option>20</option><option>30</option><option>50</option></select><select id="order"><option value="case">Prompt / seed / step</option><option value="worst">Worst Quality PSNR first</option></select></div>
<select id="case"></select><button id="prev">Previous</button><button id="next">Next</button><p id="caption"></p><div class="labels"><span>Native Wan VAE</span><span>TAEHV81 → 21 endpoints</span><span>Quality</span></div><video id="video" controls muted loop playsinline preload="metadata"></video><p class="small">All three columns are frame-synchronized in one local MP4. 4 fps preserves the original 5-second duration. Fullscreen to inspect detail.</p></section>
<section><h2>Six worst cases by Quality PSNR</h2><p>Selected by error, so they expose limitations. Click to load the full video.</p><div id="worst" class="cards"></div></section>
<section><h2>Scope</h2><p>No checkpoint updates, no quality-based filtering, no prompt or latent overlap with the earlier six-case decoder review. Inputs come from an existing TTS cache that had not been used in this decoder experiment.</p><p>Only one generator, scheduler, resolution and 11 short prompts. This does not establish broad out-of-distribution generalization, semantic-evaluator agreement or failure-detection accuracy.</p><p><a href="report.json">Raw metrics and paired confidence intervals</a> · <a href="plan.json">Frozen plan and provenance</a> · <a href="all_prompts_step10.jpg">Every prompt at step 10</a> · <a href="worst6.jpg">Worst-case contact sheet</a></p></section></main>
<script>const rows=ROWS;const $=id=>document.getElementById(id);let shown=[];for(const p of [...new Set(rows.map(r=>r.prompt_id))]){const o=document.createElement('option');o.value=p;o.textContent=p+' — '+rows.find(r=>r.prompt_id===p).prompt;$('prompt').append(o)}function load(id){const r=rows.find(x=>x.id===id);if(!r)return;$('caption').textContent=r.id+' · '+r.prompt+' · Quality '+r.psnr.toFixed(2)+' dB / SSIM '+r.ssim.toFixed(3);$('video').src=r.video;$('video').poster=r.poster;$('case').value=r.id}function filter(){shown=rows.filter(r=>($('prompt').value==='all'||r.prompt_id===$('prompt').value)&&($('step').value==='all'||r.step===Number($('step').value)));if($('order').value==='worst')shown.sort((a,b)=>a.psnr-b.psnr);$('case').replaceChildren();for(const r of shown){const o=document.createElement('option');o.value=r.id;o.textContent=r.id+' · '+r.psnr.toFixed(2)+' dB';$('case').append(o)}if(shown.length)load(shown[0].id)}for(const id of ['prompt','step','order'])$(id).onchange=filter;$('case').onchange=()=>load($('case').value);function advance(d){const i=shown.findIndex(r=>r.id===$('case').value);load(shown[(i+d+shown.length)%shown.length].id)}$('prev').onclick=()=>advance(-1);$('next').onclick=()=>advance(1);for(const r of [...rows].sort((a,b)=>a.psnr-b.psnr).slice(0,6)){const b=document.createElement('button');const img=document.createElement('img');img.src=r.poster;img.loading='lazy';const p=document.createElement('div');p.textContent=r.id+' · '+r.psnr.toFixed(2)+' dB';b.append(img,p);b.onclick=()=>{$('prompt').value='all';$('step').value='all';filter();load(r.id);$('viewer').scrollIntoView({behavior:'smooth'})};$('worst').append(b)}filter();</script></html>'''
    text=text.replace('TABLE',table).replace('STAGE',stage).replace('ROWS',json.dumps(preview_rows).replace('</','<\\/'))
    (root/'index.html').write_text(text)
    # Decode every MP4 to verify playable content, not merely container metadata.
    checks=[]
    for r in records:
        path=root/r['media']['video']
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=60)
        checks.append({'id':r['id'],'sha256':sha256(path),'decoded':True})
    atomic_json(root/'media_validation.json',{'videos':len(checks),'all_fully_decoded':True,'records':checks})
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','by_prompt')},indent=2))

if __name__=='__main__':main()
