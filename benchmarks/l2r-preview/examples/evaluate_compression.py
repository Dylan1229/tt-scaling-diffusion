"""Evaluate the single validation-selected compression candidate, then freeze results."""
import json,os,math,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
from examples.evaluate_pyramid import run,build_model,CACHE
from examples.evaluate_unseen import DEFAULT
from l2r_preview.io import load_latent,load_checkpoint
from l2r_preview.latents import WanLatentSpec
from l2r_preview.quality_metrics import full_video_metrics
from l2r_preview.train_openvid import atomic_json,resource_guard
from l2r_preview.openvid import sha256

ROOT=Path('/data/datasets/peihao/l2r-preview/outputs/compression_after_unseen_20261007')

def main():
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    plan=json.loads((ROOT/'plan.json').read_text())
    chosen=ROOT/'balanced_distill/best.pt'
    state=torch.load(chosen,map_location='cpu',weights_only=True)
    initial=torch.load(plan['student_initial'],map_location='cpu',weights_only=True)
    unchanged=all(torch.equal(v,initial['state_dict'][k]) for k,v in state['state_dict'].items())
    selection={'selected_step':state['training']['step'],'checkpoint_sha256':sha256(chosen),
        'weights_equal_initial':unchanged,'training_completed_steps':json.loads((ROOT/'balanced_distill/progress.json').read_text())['step'],
        'selection':'Minimum fixed250-clip validation RGB MSE, including initial checkpoint; frozen before this evaluation'}
    atomic_json(ROOT/'selection.json',selection)
    for name,ckpt in [('quality_full',Path(plan['teacher'])),('balanced_selected',chosen)]:
        run(SimpleNamespace(model=name,checkpoint=ckpt,output=ROOT/'validation',gpu_uuid=gpu,full_metrics=True,split='validation'))
    q=json.loads((ROOT/'validation/quality_full.json').read_text())
    b=json.loads((ROOT/'validation/balanced_selected.json').read_text())
    decision={'selection':selection,'validation_delta_psnr_db':b['full_psnr_db']-q['full_psnr_db'],
        'validation_delta_ssim':b['ssim']-q['ssim'],'quality_budget':{'max_psnr_drop_db':0.2,'max_ssim_drop':0.01}}
    decision['within_provisional_budget']=decision['validation_delta_psnr_db']>=-.2 and decision['validation_delta_ssim']>=-.01
    atomic_json(ROOT/'validation_decision.json',decision)
    spec=WanLatentSpec.from_vae_config(CACHE/'vae_config.json')
    model,convert,forward,_,_,_=build_model(SimpleNamespace(model='balanced_selected',checkpoint=chosen),spec)
    report={'selection':selection,'cases':[],'parameters':model.num_parameters,'scope':'One fixed selected candidate; no retraining or reselection after unseen evaluation'}
    with torch.inference_mode():
        for i,case in enumerate(json.loads((DEFAULT/'plan.json').read_text())['cases']):
            if i%25==0:resource_guard(gpu,ROOT)
            z=spec.denormalize(load_latent(case['source']))
            pred=forward(convert(z))
            ref=torch.load(DEFAULT/'references'/f"{case['id']}.pt",map_location='cpu',weights_only=True).cuda()
            report['cases'].append({'id':case['id'],'step':case['step'],'prompt_id':case['prompt_id'],**full_video_metrics(ref,pred)})
            del ref,pred,z
    report['pooled_psnr_db']=-10*math.log10(statistics.mean(r['full_mse_01'] for r in report['cases']))
    report['mean_ssim']=statistics.mean(r['ssim'] for r in report['cases'])
    report['completed']=True
    atomic_json(ROOT/'unseen_balanced_selected.json',report)
    print(json.dumps({'validation':decision,'unseen_psnr_db':report['pooled_psnr_db'],'unseen_ssim':report['mean_ssim']},indent=2))

if __name__=='__main__':main()
