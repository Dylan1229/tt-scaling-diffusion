"""Separate visual evaluation of compiled Quality; never modifies the eager report."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,statistics,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import imageio_ffmpeg
from l2r_preview.io import load_checkpoint,load_latent
from l2r_preview.runtime import make_preview_decoder
from l2r_preview.openvid import sha256
from l2r_preview.quality_metrics import full_video_metrics
from l2r_preview.train_openvid import atomic_json,resource_guard

BASE=Path('/data/datasets/peihao/l2r-preview')
OLD=BASE/'outputs/unseen_diffusion_20261007'
NEW=BASE/'outputs/unseen_compiled_20261007'
EXPORT=BASE/'outputs/compression_after_unseen_20261007/checkpoints/quality_bf16.pt'
MODE='max-autotune-no-cudagraphs'


def preserve_old():
    checks=json.loads((OLD/'bundle_checksums.json').read_text())
    for name,digest in checks.items():
        if sha256(OLD/name)!=digest:raise ValueError(f'Original report changed: {name}')
    return {'verified_files':len(checks),'bundle_checksums_sha256':sha256(OLD/'bundle_checksums.json'),
        'index_sha256':sha256(OLD/'index.html'),'report_sha256':sha256(OLD/'report.json')}


def prepare():
    if NEW.exists():raise FileExistsError(f'Refusing to overwrite {NEW}')
    old=json.loads((OLD/'plan.json').read_text())
    preserved=preserve_old()
    for name in ['source','records','media','logs']: (NEW/name).mkdir(parents=True,exist_ok=True)
    selected=[old['cases'][i]['id'] for i in (0,82,164)]
    plan={'frozen_utc':datetime.now(timezone.utc).isoformat(),'original_report_preservation':preserved,
        'original_report':str(OLD),'original_plan_sha256':sha256(OLD/'plan.json'),
        'original_checkpoint':old['checkpoints']['quality_full'],'compiled_checkpoint':str(EXPORT),
        'compiled_checkpoint_sha256':sha256(EXPORT),'compile_mode':MODE,'cases':old['cases'],
        'trajectories':old['trajectories'],'prompts':old['prompts'],'seeds':old['seeds'],'steps':old['steps'],
        'output_shape':old['output_shape'],'dtype':'bfloat16','input_space':old['input_space'],
        'gallery_columns':['Native Wan VAE','Quality eager (original 5.37ms version)','Quality compiled (2.42ms version)'],
        'media':'Every compiled video is newly rendered in this run. Eager is redecoded with the original checkpoint for paired comparison.',
        'reference':'Reuse and verify the native-VAE raw reference tensors; both candidates use the identical cached FP16 reference.',
        'metric_scope':old['metrics'],'timing_cases':selected,
        'timing':'3 fixed cases,10 warmups each method,30 synchronized repeats each with alternating order; excludes cold compilation,I/O,transfer,metrics,MP4',
        'visual_selection':'Every case available. Fixed seed5 step10 covers all11 prompts; also show6 worst compiled PSNR and6 largest eager/compiled pixel differences.',
        'limits':old['limits']+['This is a second examination of the SAME held-out set,not another independent dataset. No training or checkpoint selection occurs.',
        'Version names5.37ms and2.42ms refer to prior measurements; this report includes a fresh timing measurement.',
        'Compiled BF16 arithmetic is not bit-exact to eager. Metrics precede MP4 compression.'],
        'evaluation_source_sha256':sha256(__file__)}
    atomic_json(NEW/'plan.json',plan)
    for f in ['examples/evaluate_compiled_unseen.py','l2r_preview/runtime.py','l2r_preview/pyramid.py','l2r_preview/io.py','l2r_preview/latents.py','l2r_preview/quality_metrics.py']:
        out=NEW/'source'/f;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((BASE/f).read_bytes())
    print(json.dumps({'frozen':str(NEW/'plan.json'),'cases':len(plan['cases']),'old_report_preserved':preserved}),flush=True)


def save_media(case_id,reference,eager,compiled):
    arrays=[(x[0].permute(1,2,3,0).float().cpu().clamp(0,1)*255).round().byte().numpy() for x in [reference,eager,compiled]]
    frames=np.concatenate(arrays,axis=2)
    output=NEW/'media'/f'{case_id}.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','2496x480','-r','4','-i','pipe:0',
        '-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],
        input=frames.tobytes(),check=True,timeout=90)
    Image.fromarray(frames[10]).resize((1248,240),Image.Resampling.LANCZOS).save(NEW/'media'/f'{case_id}.jpg',quality=93)
    return {'video':f'media/{case_id}.mp4','poster':f'media/{case_id}.jpg','video_sha256':sha256(output),'frames':21,'fps':4,'columns':['Native VAE','Eager Quality','Compiled Quality']}


@torch.inference_mode()
def evaluate(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Explicit GPU pin required')
    lock=(NEW/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((NEW/'plan.json').read_text())
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4)
    resources=resource_guard(args.gpu_uuid,NEW)
    assert sha256(EXPORT)==plan['compiled_checkpoint_sha256']
    assert sha256(plan['original_checkpoint']['checkpoint'])==plan['original_checkpoint']['sha256']
    original,spec,_=load_checkpoint(plan['original_checkpoint']['checkpoint'])
    exported,espec,_=load_checkpoint(EXPORT)
    assert spec==espec
    assert original.architecture==exported.architecture
    assert all(torch.equal(a.bfloat16(),b.bfloat16()) for a,b in zip(original.parameters(),exported.parameters(),strict=True))
    original=original.cuda().bfloat16().requires_grad_(False)
    exported=exported.cuda().bfloat16().requires_grad_(False)
    eager=make_preview_decoder(original,'none')
    compiled=make_preview_decoder(exported,MODE)
    old_records=json.loads((OLD/'report.json').read_text())['records']
    references={r['id']:r for r in old_records}
    started=time.monotonic()
    atomic_json(NEW/'progress.json',{'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'status':'compiling','completed':0,'total':165})
    z=spec.denormalize(load_latent(plan['cases'][0]['source'])).cuda().bfloat16()
    t0=time.monotonic();warm=compiled(z);torch.cuda.synchronize();cold=time.monotonic()-t0
    assert torch.isfinite(warm).all();del warm,z
    timing=[]
    records=[]
    for index,case in enumerate(plan['cases']):
        if index%10==0:resources=resource_guard(args.gpu_uuid,NEW)
        assert sha256(case['source'])==case['sha256']
        old=references[case['id']]
        ref_path=Path(old['reference_path'])
        assert sha256(ref_path)==old['reference_sha256']
        z=spec.denormalize(load_latent(case['source'])).cuda().bfloat16()
        if case['id'] in plan['timing_cases']:
            for fn in (eager,compiled):
                for _ in range(10):tmp=fn(z);del tmp
            samples={'eager':[],'compiled':[]}
            for repeat in range(30):
                order=[('eager',eager),('compiled',compiled)]
                if repeat%2:order.reverse()
                for name,fn in order:
                    torch.cuda.synchronize();t=time.perf_counter();tmp=fn(z);torch.cuda.synchronize()
                    samples[name].append((time.perf_counter()-t)*1000);del tmp
            timing.append({'case':case['id'],**samples})
        a,b=eager(z),compiled(z)
        ref=torch.load(ref_path,map_location='cpu',weights_only=True).cuda()
        assert a.shape==b.shape==ref.shape==tuple(plan['output_shape'])
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta=(a.float()-b.float()).abs()
        row={**case,'eager':full_video_metrics(ref,a),'compiled':full_video_metrics(ref,b),
            'compiled_minus_eager':{'max_abs_rgb':float(delta.max()),'mean_abs_rgb':float(delta.mean()),
                'mse_rgb':float(delta.square().mean()),'nonidentical_fraction':float((delta!=0).float().mean())},
            'source_sha256':case['sha256'],'native_reference_sha256':old['reference_sha256']}
        row['media']=save_media(case['id'],ref,a,b)
        row['completed_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(NEW/'records'/f"{case['id']}.json",row);records.append(row)
        atomic_json(NEW/'progress.json',{'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'status':'rendering','completed':index+1,'total':len(plan['cases']),
            'last_case':case['id'],'elapsed_seconds':time.monotonic()-started,'compile_and_first_call_seconds':cold})
        if (index+1)%10==0:print(json.dumps({'completed':index+1,'total':len(plan['cases']),'elapsed_seconds':time.monotonic()-started}),flush=True)
        del a,b,ref,z,delta
    report={'completed':True,'cases':len(records),'records':records,'timing':timing,'compile_and_first_call_seconds':cold,
        'median_ms':{name:statistics.median(x for t in timing for x in t[name]) for name in ('eager','compiled')},
        'gpu_uuid':args.gpu_uuid,'gpu_name':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'peak_allocated_bytes_combined_evaluation':torch.cuda.max_memory_allocated(),
        'memory_scope':'Both decoder instances and metric/media tensors coexist; not an isolated inference memory benchmark.',
        'resources':resources,'plan_sha256':sha256(NEW/'plan.json'),'old_report_preservation_after':preserve_old(),
        'elapsed_seconds':time.monotonic()-started,'weights_equal_at_bf16':True}
    atomic_json(NEW/'evaluation.json',report)
    atomic_json(NEW/'progress.json',{'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'status':'complete','completed':len(records),'total':len(records),'elapsed_seconds':report['elapsed_seconds']})
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','timing')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--gpu-uuid');args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:evaluate(args)
