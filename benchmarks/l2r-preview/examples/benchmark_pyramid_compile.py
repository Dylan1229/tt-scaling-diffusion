"""Bounded inference-only compilation experiment with frozen weights and input checks."""
import argparse,json,os,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from l2r_preview.io import load_checkpoint,load_latent
from l2r_preview.runtime import make_preview_decoder
from l2r_preview.openvid import load_records,sha256
from l2r_preview.train_openvid import atomic_json,resource_guard
from l2r_preview.quality_metrics import full_video_metrics

CACHE=Path('/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16')
ROOT=Path('/data/datasets/peihao/l2r-preview/outputs/unseen_diffusion_20261007')

@torch.inference_mode()
def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu_uuid
    args.output.parent.mkdir(parents=True,exist_ok=True)
    resource_guard(args.gpu_uuid,args.output.parent)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.4)
    model,spec,_=load_checkpoint(args.checkpoint)
    model=model.cuda().bfloat16().requires_grad_(False)
    records=load_records(CACHE,'validation')
    eager=make_preview_decoder(model,"none")
    z=load_file(CACHE/records[0]['latent_path'])['latent'].unsqueeze(0).cuda().bfloat16()
    reference=eager(z)
    compiled=make_preview_decoder(model,args.mode)
    t0=time.monotonic()
    output=compiled(z)
    torch.cuda.synchronize()
    cold=time.monotonic()-t0
    report={'mode':args.mode,'checkpoint_sha256':sha256(args.checkpoint),'gpu':torch.cuda.get_device_name(),
      'gpu_uuid':args.gpu_uuid,'torch':str(torch.__version__),'shape':list(z.shape),'dtype':'bfloat16',
      'compile_and_first_call_seconds':cold,'initial_max_abs_difference':float((output-reference).abs().max()),
      'timing_scope':'Same process and weights, same GPU.3 validation inputs,10 warmups,30 synchronized repetitions per method/input, alternating order. Excludes transfer,load,compile.',
      'timing':[],'validation_differences':[],'unseen_differences':[],'completed':False}
    atomic_json(args.output,report)
    del output,reference
    for i in range(3):
        z=load_file(CACHE/records[i]['latent_path'])['latent'].unsqueeze(0).cuda().bfloat16()
        for f in (eager,compiled):
            for _ in range(10):f(z)
        torch.cuda.synchronize()
        times={'eager':[],'compiled':[]}
        # Alternation limits effects of temperature or clock drift.
        for repeat in range(30):
            sequence=[('eager',eager),('compiled',compiled)]
            if repeat%2:sequence.reverse()
            for name,f in sequence:
                torch.cuda.synchronize();t=time.perf_counter();out=f(z);torch.cuda.synchronize()
                times[name].append((time.perf_counter()-t)*1000);del out
        report['timing'].append({'input_id':records[i]['id'],**times})
    for i,r in enumerate(records):
        z=load_file(CACHE/r['latent_path'])['latent'].unsqueeze(0).cuda().bfloat16()
        a,b=eager(z),compiled(z)
        diff=(a-b).float()
        report['validation_differences'].append({'id':r['id'],'max_abs':float(diff.abs().max()),'mean_abs':float(diff.abs().mean())})
        del a,b,diff,z
    frozen=json.loads((ROOT/'plan.json').read_text())
    for i,r in enumerate(frozen['cases']):
        if i%30==0:resource_guard(args.gpu_uuid,args.output.parent)
        z=spec.denormalize(load_latent(r['source'])).cuda().bfloat16()
        a,b=eager(z),compiled(z)
        ref=torch.load(ROOT/'references'/f"{r['id']}.pt",map_location='cpu',weights_only=True).cuda()
        ma,mb=full_video_metrics(ref,a),full_video_metrics(ref,b)
        diff=(a-b).float()
        report['unseen_differences'].append({'id':r['id'],'max_abs':float(diff.abs().max()),'mean_abs':float(diff.abs().mean()),
            'eager':ma,'compiled':mb})
        del a,b,diff,z,ref
    report['median_ms']={name:statistics.median([x for r in report['timing'] for x in r[name]]) for name in ('eager','compiled')}
    report['speedup']=report['median_ms']['eager']/report['median_ms']['compiled']
    report['unseen_mean_case_psnr_change']=statistics.mean(r['compiled']['full_psnr_db']-r['eager']['full_psnr_db'] for r in report['unseen_differences'])
    report['unseen_mean_ssim_change']=statistics.mean(r['compiled']['ssim']-r['eager']['ssim'] for r in report['unseen_differences'])
    report['worst_unseen_max_abs_difference']=max(r['max_abs'] for r in report['unseen_differences'])
    report['completed']=True
    atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('timing','validation_differences','unseen_differences')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mode',choices=['default','max-autotune-no-cudagraphs'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    run(p.parse_args())
