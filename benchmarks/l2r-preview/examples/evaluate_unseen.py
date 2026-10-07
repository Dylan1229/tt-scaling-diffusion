"""Frozen-checkpoint evaluation on previously unused TTS diffusion trajectories.

Every target is the native VAE decode of the SAME predicted-clean latent.
No training or candidate selection uses this evaluation set.
"""
from __future__ import annotations
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import imageio_ffmpeg
import torch
from PIL import Image
from examples.evaluate_pyramid import BASE, CACHE, build_model
from l2r_preview.io import load_latent
from l2r_preview.latents import WanLatentSpec
from l2r_preview.openvid import sha256
from l2r_preview.quality_metrics import full_video_metrics
from l2r_preview.train_openvid import atomic_json, resource_guard

DEFAULT = BASE/'outputs/unseen_diffusion_20261007'
RUN = Path('/data/datasets/peihao/tt-scaling-diffusion/runs/baseline/all150_euler_online_001')
QUALITY = BASE/'outputs/pyramid_search_20261006/quality_full/best.pt'
STEPS = (5, 10, 20, 30, 50)
SEEDS = (2, 5, 8)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise FileExistsError('Frozen plan already exists; do not regenerate it')
    prior = json.loads((BASE/'outputs/taehv_20261002/config.json').read_text())
    previous_prompts = {r['prompt'] for r in prior['cases']}
    previous_hashes = {r['sha256'] for r in prior['cases']}
    cases = []
    for number in range(5, 16):
        prompt_id = f'p{number:02d}'
        for seed in SEEDS:
            trajectory = RUN/prompt_id/f'seed{seed:04d}'
            meta = json.loads((trajectory/'meta.json').read_text())
            assert (trajectory/'DONE').exists()
            assert meta['prompt_text'] not in previous_prompts
            assert (meta['num_frames'], meta['height'], meta['width'], meta['num_inference_steps']) == (81,480,832,50)
            for step in STEPS:
                path = trajectory/'posterior_means'/f'step_{step-1:03d}.pt'
                digest = sha256(path)
                assert digest not in previous_hashes
                z = load_latent(path)
                assert tuple(z.shape) == (1,48,21,30,52)
                cases.append({'id':f'{prompt_id}_seed{seed:04d}_step{step:02d}',
                    'trajectory':f'{prompt_id}_seed{seed:04d}', 'prompt_id':prompt_id,
                    'prompt':meta['prompt_text'], 'axis':meta['axis'], 'seed':seed, 'step':step,
                    'source':str(path), 'sha256':digest, 'meta_sha256':sha256(trajectory/'meta.json'),
                    'shape':list(z.shape), 'latent_kind':'predicted_clean',
                    'mean':float(z.mean()), 'std':float(z.std())})
    assert len(cases)==165 and len({r['sha256'] for r in cases})==165
    for name in ('references','records','media','logs','source'):
        (root/name).mkdir(exist_ok=True)
    models = {'quality_full':{'checkpoint':str(QUALITY),'sha256':sha256(QUALITY)},
              'taehv21':{}, 'taehv81':{}}
    plan = {'frozen_utc':datetime.now(timezone.utc).isoformat(), 'checkpoints':models,
        'cases':cases, 'trajectories':33, 'prompts':11, 'seeds':list(SEEDS), 'steps':list(STEPS),
        'model_path':prior['model_path'], 'generator':'Wan2.2 TI2V 5B, Euler, CFG5, 50steps',
        'input_space':'TTS normalized predicted-clean; denormalize in FP32 once, then BF16 for native and Quality',
        'comparison':'native VAE decodes81 frames, compare aligned endpoints0,4,...,80; no final-video teacher target for early steps',
        'output_shape':[1,3,21,480,832], 'dtype':'bfloat16',
        'selection':'All previously unreviewed prompts p05..p15; fixed seeds2,5,8; fixed steps5,10,20,30,50. No output-quality filtering.',
        'unseen_audit':{'no_prompt_overlap_with_previous_six':True,'no_latent_hash_overlap_with_previous_six':True,
            'quality_training':'Only4500 OpenVid real-video VAE latent/RGB pairs; no diffusion trajectories used in training',
            'source_run_config_sha256':sha256(RUN/'config.snapshot.yaml')},
        'metrics':'PSNR pools all21 full-resolution frames; SSIM on frames0,10,20; temporal-delta MAE. Paired comparisons to native, not generation correctness.',
        'aggregate':'Report per-step, per-prompt, all-case distributions; prompt-cluster bootstrap intervals for paired model differences.',
        'success_scope':'No universal pass threshold. Assess fidelity, compare fixed baselines and show worst cases before deciding compression.',
        'limits':['11 short prompts from one generator/scheduler/resolution; no cross-model or cross-dataset OOD claim.',
                  '33 trajectories are independent seeds conditional on11 prompts;165 steps are correlated, not165 independent videos.',
                  'No semantic evaluator or failure-detector validation.'],
        'evaluation_source_sha256':sha256(__file__)}
    atomic_json(root/'plan.json',plan)
    (root/'source/evaluate_unseen.py').write_text(Path(__file__).read_text())
    print(json.dumps({'frozen':str(root/'plan.json'),'cases':len(cases),'checkpoint':models['quality_full']}),flush=True)


def save_media(root, name, tensors):
    # Full spatial resolution is retained in the video; browser can show fullscreen.
    arrays = [(v[0].permute(1,2,3,0).float().cpu().clamp(0,1)*255).round().byte().numpy() for v in tensors]
    frames=np.concatenate(arrays,axis=2)
    path=root/'media'/f'{name}.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{frames.shape[2]}x{frames.shape[1]}',
         '-r','4','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    subprocess.run(cmd,input=frames.tobytes(),check=True,timeout=90)
    Image.fromarray(frames[len(frames)//2]).resize((1248,240),Image.Resampling.LANCZOS).save(root/'media'/f'{name}.jpg',quality=90)
    return {'video':f'media/{name}.mp4','poster':f'media/{name}.jpg','columns':['Native Wan VAE','TAEHV81 sampled21','Quality'],
            'video_sha256':sha256(path)}


@torch.inference_mode()
def evaluate(args):
    root=args.output
    plan=json.loads((root/'plan.json').read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu_uuid
    lock=(root/f'worker{args.shard}.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.6)
    resources=resource_guard(args.gpu_uuid,root)
    from diffusers import AutoencoderKLWan
    spec=WanLatentSpec.from_vae_config(Path(plan['model_path'])/'vae/config.json')
    native=AutoencoderKLWan.from_pretrained(plan['model_path'],subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).eval()
    native.encoder=torch.nn.Identity()
    native=native.cuda().requires_grad_(False)
    models={}
    for name,entry in plan['checkpoints'].items():
        ckpt=Path(entry['checkpoint']) if entry else None
        if ckpt: assert sha256(ckpt)==entry['sha256']
        built=build_model(SimpleNamespace(model=name,checkpoint=ckpt),spec)
        models[name]=built
    assigned=plan['cases'][args.shard::args.shards]
    if args.limit: assigned=assigned[:args.limit]
    start=time.monotonic()
    completed=0
    for case in assigned:
        destination=root/'records'/f"{case['id']}.json"
        if destination.exists():
            saved=json.loads(destination.read_text())
            assert saved['source_sha256']==case['sha256']
            assert saved['quality_checkpoint_sha256']==plan['checkpoints']['quality_full']['sha256']
            completed+=1
            continue
        resources=resource_guard(args.gpu_uuid,root)
        assert sha256(case['source'])==case['sha256']
        raw=load_latent(case['source'])
        z=spec.denormalize(raw)
        z_gpu=z.cuda().bfloat16()
        t0=time.monotonic()
        decoded=native.decode(z_gpu,return_dict=False)[0]
        assert tuple(decoded.shape)==(1,3,81,480,832)
        ref=decoded[:,:,::4].float().add(1).mul(0.5).clamp(0,1).contiguous()
        del decoded
        assert torch.isfinite(ref).all()
        ref_path=root/'references'/f"{case['id']}.pt"
        torch.save(ref.cpu().half(),ref_path)
        outputs={}
        result={}
        for name,(model,convert,forward,computed,digest,training) in models.items():
            output=forward(convert(z))
            selected=output if computed==21 else output[:,:,::4]
            assert selected.shape==ref.shape and torch.isfinite(selected).all()
            result[name]=full_video_metrics(ref,selected)
            result[name]['parameters']=sum(p.numel() for p in model.parameters())
            outputs[name]=selected.cpu().half()
            del output,selected
        torch.cuda.synchronize()
        media=save_media(root,case['id'],[ref.cpu().half(),outputs['taehv81'],outputs['quality_full']])
        record={**case,'source_sha256':case['sha256'],'quality_checkpoint_sha256':plan['checkpoints']['quality_full']['sha256'],
            'reference_path':str(ref_path),'reference_sha256':sha256(ref_path),'metrics':result,'media':media,
            'gpu_uuid':args.gpu_uuid,'pid':os.getpid(),'evaluation_source_sha256':sha256(__file__),'seconds_including_metrics_media':time.monotonic()-t0,
            'resources':resources,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'completed_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(destination,record)
        completed+=1
        status={'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'shard':args.shard,'shards':args.shards,
            'completed':completed,'assigned':len(assigned),'last_case':case['id'],'elapsed_seconds':time.monotonic()-start,
            'status':'complete' if completed==len(assigned) else 'running'}
        atomic_json(root/f'progress_{args.shard}.json',status)
        print(json.dumps(status),flush=True)
        del ref,z_gpu,z,raw,outputs
    for name,entry in plan['checkpoints'].items():
        if entry: assert sha256(entry['checkpoint'])==entry['sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.output)
    else: evaluate(args)

if __name__=='__main__': main()
