import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from examples.compare_taehv import frames_u8, save_mp4

root=Path("/data/datasets/peihao/l2r-preview/outputs/pyramid_search_20261006")
web=root/"web"
torch.set_num_threads(4)
scores={}
for group in ("test","diffusion"):
    report=json.loads((root/group/"quality_full.json").read_text())
    cases=report["visual_records"] if group=="test" else report["cases"]
    rows={r["id"]:r for r in report["records" if group=="test" else "cases"]}
    for case in cases:
        name=case["id"]
        arrays=[]
        for model in ("source","taehv81","quality_full"):
            rgb=torch.load(root/group/"tensors"/(name+"_"+model+".pt"),weights_only=True)
            a=frames_u8(rgb)
            arrays.append(np.stack([np.array(Image.fromarray(f).resize((416,240),Image.Resampling.LANCZOS)) for f in a]))
        combined=np.concatenate(arrays,axis=2)
        save_mp4(web/(name+"_quality_comparison.mp4"),combined,4)
        Image.fromarray(combined[10]).save(web/(name+"_quality_poster.jpg"),quality=94)
        scores[name]=rows[name]["full_psnr_db"]
page=(web/"index.html").read_text()
control='<p><label>Compare: <select id="featured"><option value="wide">Wide — highest validation PSNR</option><option value="quality">Quality — faster, near-best validation PSNR</option></select></label></p>'
page=page.replace("<nav>",control+"<nav>",1)
script="""
<script>
const qualityScores=QUALITY_SCORES;
document.querySelectorAll('section[data-group]').forEach(s=>{
 const v=s.querySelector('video');
 v.dataset.case=v.getAttribute('src').replace('_comparison.mp4','');
 s.dataset.wideScore=s.querySelector('p').textContent;
});
document.querySelector('#featured').onchange=e=>{
 const isQuality=e.target.value==='quality';
 document.querySelectorAll('section[data-group]').forEach(s=>{
  const v=s.querySelector('video'), name=v.dataset.case, t=v.currentTime, playing=!v.paused;
  v.pause();
  v.src=name+(isQuality?'_quality_comparison.mp4':'_comparison.mp4');
  v.poster=name+(isQuality?'_quality_poster.jpg':'_poster.jpg');
  v.addEventListener('loadedmetadata',()=>{v.currentTime=Math.min(t,v.duration);if(playing)v.play().catch(()=>{});},{once:true});
  s.querySelector('.labels').lastElementChild.textContent=isQuality?'Quality':'Wide';
  s.querySelector('p').textContent=isQuality?
   'Quality: '+qualityScores[name].toFixed(2)+' dB vs '+(s.dataset.group==='test'?'original video':'Wan VAE reference')+' · 21 frames / 4 fps':
   s.dataset.wideScore;
 });
};
</script>
""".replace("QUALITY_SCORES",json.dumps(scores))
page=page.replace("</html>",script+"</html>")
(web/"index.html").write_text(page)
print("Added synchronized Quality view to all12 cases",flush=True)
