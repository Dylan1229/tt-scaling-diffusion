"""Four-technique unseen comparison with a new, independent report directory."""
from __future__ import annotations
import argparse, fcntl, json, os, statistics, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from examples.evaluate_pyramid import build_model, UPSTREAM
from l2r_preview.io import load_checkpoint, load_latent
from l2r_preview.openvid import sha256
from l2r_preview.quality_metrics import full_video_metrics
from l2r_preview.runtime import make_preview_decoder
from l2r_preview.train_openvid import atomic_json, resource_guard

BASE = Path('/data/datasets/peihao/l2r-preview')
OLD = BASE/'outputs/unseen_diffusion_20261007'
COMPILED = BASE/'outputs/unseen_compiled_20261007'
NEW = BASE/'outputs/unseen_fourway_20261007'
NAMES = {'native':'Native', 'taehv21':'TAEHV 21 endpoints',
         'eager':'Eager BF16', 'compiled':'Compiled Quality + BF16 Export'}
MODE = 'max-autotune-no-cudagraphs'

def preserved():
    result = {}
    for root in (OLD, COMPILED):
        checks = json.loads((root/'bundle_checksums.json').read_text())
        for name, digest in checks.items():
            if sha256(root/name) != digest:
                raise ValueError(f'Existing report changed: {root/name}')
        result[root.name] = {
            'verified_files':len(checks),
            'checksums_sha256':sha256(root/'bundle_checksums.json'),
            'index_sha256':sha256(root/'index.html'),
            'report_sha256':sha256(root/'report.json')}
    return result

def prepare():
    if NEW.exists():
        raise FileExistsError(f'Refusing to replace {NEW}')
    old = json.loads((OLD/'plan.json').read_text())
    cp = json.loads((COMPILED/'plan.json').read_text())
    assert old['cases'] == cp['cases'] and len(old['cases']) == 165
    before = preserved()
    for name in ('source','records','media','logs'):
        (NEW/name).mkdir(parents=True, exist_ok=True)
    plan = {
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'techniques':NAMES, 'cases':old['cases'], 'trajectories':33, 'prompts':11,
        'steps':old['steps'], 'seeds':old['seeds'], 'output_shape':old['output_shape'],
        'native_model_path':old['model_path'],
        'original_checkpoint':cp['original_checkpoint'],
        'compiled_checkpoint':cp['compiled_checkpoint'],
        'compiled_checkpoint_sha256':cp['compiled_checkpoint_sha256'],
        'compile_mode':MODE,
        'taehv_revision':json.loads((UPSTREAM/'revision.json').read_text()),
        'taehv21_mode':{'decoder_time_upscale':[False,False,False],
                        'decode_video_parallel':True},
        'dtype':'bfloat16', 'batch':1,
        'timing_cases':[old['cases'][i]['id'] for i in (0,82,164)],
        'timing_contract':'Each technique: 3 fixed inputs, 10 warmups per input, 30 repeats per input in rotating order; CUDA synchronize before and after. Includes forward, RGB conversion/clamp and native endpoint selection; excludes input preparation, transfer, I/O, compilation, metrics and video encoding.',
        'comparison':'Same predicted-clean latent. Native computes 81 frames and selects 0,4,...,80. TAEHV 21 endpoints disables all temporal upsampling and computes 21 frames directly. Eager and compiled compute 21 frames directly.',
        'reference':'Hash-verified cached native FP16 RGB from the original evaluation. All candidate metrics use this identical reference; native is also freshly decoded for the three timing inputs.',
        'metrics':'Raw RGB: PSNR over all 21 frames; SSIM on frames 0,10,20; temporal-delta MAE.',
        'media':'All four-column videos are newly rendered from raw RGB in this run; no MP4 crops or TAEHV81 substitution.',
        'prior_reports_preserved_before':before,
        'limits':['Same frozen held-out set: 165 latents from 33 trajectories and 11 prompts; not 165 independent videos.',
                  'One generator, scheduler and resolution. No retraining, model selection or semantic-detector evaluation.',
                  'BF16 export reduces checkpoint storage, not parameter count or BF16 runtime weight memory.',
                  'Compiled execution may differ slightly from eager due to arithmetic rounding.'],
        'evaluation_source_sha256':sha256(__file__)}
    atomic_json(NEW/'plan.json',plan)
    files = ['examples/evaluate_fourway_unseen.py','examples/evaluate_pyramid.py',
             'l2r_preview/runtime.py','l2r_preview/pyramid.py','l2r_preview/io.py',
             'l2r_preview/latents.py','l2r_preview/quality_metrics.py']
    for name in files:
        target=NEW/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((BASE/name).read_bytes())
    (NEW/'source/taehv_revision.json').write_text(json.dumps(plan['taehv_revision'],indent=2)+'\n')
    print(json.dumps({'prepared':str(NEW),'cases':165,'preserved':before}),flush=True)

def save_media(case_id, tensors):
    arrays=[(x[0].permute(1,2,3,0).float().cpu().clamp(0,1)*255).round().byte().numpy() for x in tensors]
    pixels=np.concatenate(arrays,axis=2)
    header=Image.new('RGB',(3328,40),'#132d39')
    draw=ImageDraw.Draw(header)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    for i,label in enumerate(NAMES.values()):
        draw.text((i*832+18,5),label,font=font,fill='#91e3c8' if i==3 else 'white')
    frames=np.concatenate([np.broadcast_to(np.array(header),(21,40,3328,3)),pixels],axis=1).copy()
    path=NEW/'media'/f'{case_id}.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','3328x520','-r','4','-i','pipe:0',
        '-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
        input=frames.tobytes(),check=True,timeout=90)
    poster=NEW/'media'/f'{case_id}.jpg'
    Image.fromarray(frames[10]).resize((1664,260),Image.Resampling.LANCZOS).save(poster,quality=94)
    return {'video':f'media/{path.name}','poster':f'media/{poster.name}',
            'video_sha256':sha256(path),'poster_sha256':sha256(poster),
            'frames':21,'fps':4,'columns':list(NAMES.values()),
            'dimensions_with_label_header':[3328,520],
            'per_technique_rgb_dimensions':[832,480]}

@torch.inference_mode()
def evaluate(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('Explicit GPU pin required')
    lock=(NEW/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((NEW/'plan.json').read_text())
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.5)
    resources=resource_guard(args.gpu_uuid,NEW)
    began=time.monotonic()
    def status(phase,completed=0,**extra):
        value={'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'status':phase,
               'completed':completed,'total':165,'elapsed_seconds':time.monotonic()-began,**extra}
        atomic_json(NEW/'progress.json',value)
        print(json.dumps(value),flush=True)
    status('loading')
    assert sha256(plan['original_checkpoint']['checkpoint']) == plan['original_checkpoint']['sha256']
    assert sha256(plan['compiled_checkpoint']) == plan['compiled_checkpoint_sha256']
    original,spec,_=load_checkpoint(plan['original_checkpoint']['checkpoint'])
    exported,export_spec,_=load_checkpoint(plan['compiled_checkpoint'])
    assert spec==export_spec and original.architecture==exported.architecture
    assert all(torch.equal(a.bfloat16(),b.bfloat16()) for a,b in zip(original.parameters(),exported.parameters(),strict=True))
    original=original.cuda().bfloat16().requires_grad_(False)
    exported=exported.cuda().bfloat16().requires_grad_(False)
    eager=make_preview_decoder(original,'none')
    compiled=make_preview_decoder(exported,MODE)
    tae,tae_convert,tae_forward,frames,tae_hash,_=build_model(SimpleNamespace(model='taehv21'),spec)
    assert frames==21
    from diffusers import AutoencoderKLWan
    native=AutoencoderKLWan.from_pretrained(plan['native_model_path'],subfolder='vae',
                torch_dtype=torch.bfloat16,local_files_only=True).eval()
    native.encoder=torch.nn.Identity()
    native=native.cuda().requires_grad_(False)
    def native_forward(z):
        full=native.decode(z,return_dict=False)[0]
        assert full.shape==(1,3,81,480,832)
        return full[:,:,::4].float().add(1).mul(.5).clamp(0,1).contiguous()
    models={'native':native,'taehv21':tae,'eager':original,'compiled':exported}
    fns={'native':native_forward,'taehv21':tae_forward,'eager':eager,'compiled':compiled}
    metadata={key:{
        'name':NAMES[key],'parameters':sum(p.numel() for p in model.parameters()),
        'runtime_weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'computed_frames':81 if key=='native' else 21,'compared_frames':21}
        for key,model in models.items()}
    for key,path in [('eager',plan['original_checkpoint']['checkpoint']),
                     ('compiled',plan['compiled_checkpoint'])]:
        metadata[key].update(checkpoint=path,checkpoint_bytes=Path(path).stat().st_size,
                             checkpoint_sha256=sha256(path))
    metadata['taehv21']['checkpoint_sha256']=tae_hash
    reference_records={r['id']:r for r in json.loads((OLD/'report.json').read_text())['records']}
    z=spec.denormalize(load_latent(plan['cases'][0]['source'])).cuda().bfloat16()
    status('compiling')
    torch.cuda.synchronize()
    t=time.perf_counter()
    first=compiled(z)
    torch.cuda.synchronize()
    cold=time.perf_counter()-t
    assert tuple(first.shape)==tuple(plan['output_shape']) and torch.isfinite(first).all()
    del first,z
    rows,timings=[],[]
    for index,case in enumerate(plan['cases']):
        if index%10==0:
            resources=resource_guard(args.gpu_uuid,NEW)
        assert sha256(case['source'])==case['sha256']
        old=reference_records[case['id']]
        ref_path=Path(old['reference_path'])
        assert sha256(ref_path)==old['reference_sha256']
        latent=spec.denormalize(load_latent(case['source']))
        z=latent.cuda().bfloat16()
        inputs={key:(tae_convert(latent) if key=='taehv21' else z) for key in NAMES}
        ref=torch.load(ref_path,map_location='cpu',weights_only=True).cuda()
        if case['id'] in plan['timing_cases']:
            samples={key:[] for key in NAMES}
            for key,fn in fns.items():
                status('timing_warmup',index,case=case['id'],technique=NAMES[key])
                for _ in range(10):
                    tmp=fn(inputs[key])
                    del tmp
            order=list(NAMES)
            status('timing',index,case=case['id'])
            for repeat in range(30):
                rotated=order[repeat%4:]+order[:repeat%4]
                for key in rotated:
                    torch.cuda.synchronize()
                    t=time.perf_counter()
                    tmp=fns[key](inputs[key])
                    torch.cuda.synchronize()
                    samples[key].append((time.perf_counter()-t)*1000)
                    assert tuple(tmp.shape)==tuple(plan['output_shape'])
                    del tmp
            fresh_native=native_forward(z)
            reference_check={'mean_abs_rgb':float((fresh_native-ref.float()).abs().mean()),
                             'max_abs_rgb':float((fresh_native-ref.float()).abs().max())}
            del fresh_native
            timings.append({'case':case['id'],'samples_ms':samples,
                            'cached_reference_check':reference_check})
            atomic_json(NEW/'timing_partial.json',timings)
        outputs={key:fns[key](inputs[key]) for key in ('taehv21','eager','compiled')}
        for output in outputs.values():
            assert tuple(output.shape)==tuple(plan['output_shape']) and torch.isfinite(output).all()
        metrics={key:full_video_metrics(ref,out) for key,out in outputs.items()}
        delta=(outputs['compiled'].float()-outputs['eager'].float()).abs()
        row={**case,'metrics':metrics,'native_reference_sha256':old['reference_sha256'],
             'native_reference_path':str(ref_path),
             'compiled_minus_eager':{'mean_abs_rgb':float(delta.mean()),
                                     'max_abs_rgb':float(delta.max())}}
        row['media']=save_media(case['id'],[ref]+[outputs[key] for key in ('taehv21','eager','compiled')])
        row['completed_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(NEW/'records'/f"{case['id']}.json",row)
        rows.append(row)
        status('rendering',index+1,last_case=case['id']) if (index+1)%10==0 else None
        del z,latent,inputs,ref,outputs,delta
    result={'completed':True,'cases':len(rows),'records':rows,'techniques':metadata,
            'timing':timings,'median_ms':{key:statistics.median(t for r in timings for t in r['samples_ms'][key]) for key in NAMES},
            'cold_compile_and_first_call_seconds':cold,'weights_equal_at_bf16':True,
            'gpu_uuid':args.gpu_uuid,'gpu_name':torch.cuda.get_device_name(),
            'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'dtype':'bfloat16','batch':1,'timing_contract':plan['timing_contract'],
            'evaluation_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memory_scope':'Combined evaluation with all four decoders and metrics in memory; not isolated inference memory.',
            'resources':resources,'prior_reports_preserved_after':preserved(),
            'plan_sha256':sha256(NEW/'plan.json'),'elapsed_seconds':time.monotonic()-began}
    atomic_json(NEW/'evaluation.json',result)
    status('complete',len(rows),median_ms=result['median_ms'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','evaluate'])
    p.add_argument('--gpu-uuid')
    args=p.parse_args()
    prepare() if args.stage=='prepare' else evaluate(args)
