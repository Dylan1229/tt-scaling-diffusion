"""Render the four-technique unseen report with method names throughout."""
import json, math, statistics, subprocess
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from examples.evaluate_fourway_unseen import NEW,NAMES,preserved
from l2r_preview.openvid import sha256
from l2r_preview.train_openvid import atomic_json

def aggregate(rows,key):
    values=[r['metrics'][key] for r in rows]
    return {'cases':len(values),
        'pooled_psnr_db':-10*math.log10(statistics.mean(r['full_mse_01'] for r in values)),
        'mean_ssim':statistics.mean(r['ssim'] for r in values),
        'mean_temporal_delta_mae':statistics.mean(r['temporal_delta_mae'] for r in values),
        'worst_case_psnr_db':min(r['full_psnr_db'] for r in values)}

def contact(rows,name):
    sheet=Image.new('RGB',(1664,296*len(rows)),'white')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for i,row in enumerate(rows):
        text=f"{row['id']} · {row['prompt']} · TAEHV {row['metrics']['taehv21']['full_psnr_db']:.2f} dB · Compiled {row['metrics']['compiled']['full_psnr_db']:.2f} dB"
        draw.text((10,i*296+4),text,font=font,fill='#183440')
        sheet.paste(Image.open(NEW/row['media']['poster']),(0,i*296+32))
    sheet.save(NEW/name,quality=94)

def main():
    plan=json.loads((NEW/'plan.json').read_text())
    evaluation=json.loads((NEW/'evaluation.json').read_text())
    rows=evaluation['records']
    assert evaluation['completed'] and len(rows)==165
    assert [r['id'] for r in rows]==[r['id'] for r in plan['cases']]
    overall={key:aggregate(rows,key) for key in ('taehv21','eager','compiled')}
    by_step={str(s):{key:aggregate([r for r in rows if r['step']==s],key)
              for key in overall} for s in plan['steps']}
    paired={'mean_case_psnr_change_db':statistics.mean(r['metrics']['compiled']['full_psnr_db']-r['metrics']['eager']['full_psnr_db'] for r in rows),
            'mean_ssim_change':statistics.mean(r['metrics']['compiled']['ssim']-r['metrics']['eager']['ssim'] for r in rows),
            'mean_abs_rgb_difference':statistics.mean(r['compiled_minus_eager']['mean_abs_rgb'] for r in rows)}
    representative=[next(r for r in rows if r['prompt_id']==p and r['seed']==5 and r['step']==10)
                    for p in sorted({r['prompt_id'] for r in rows})]
    worst=sorted(rows,key=lambda r:r['metrics']['compiled']['full_psnr_db'])[:6]
    changed=sorted(rows,key=lambda r:r['compiled_minus_eager']['mean_abs_rgb'],reverse=True)[:6]
    report={'completed':True,'cases':165,'trajectories':33,'prompts':11,
            'technique_names':NAMES,'techniques':evaluation['techniques'],
            'overall':overall,'by_step':by_step,'paired_compiled_minus_eager':paired,
            'median_ms':evaluation['median_ms'],'timing_contract':plan['timing_contract'],
            'cold_compile_and_first_call_seconds':evaluation['cold_compile_and_first_call_seconds'],
            'scope':plan['limits'],'prior_reports_preserved':preserved(),
            'evaluation_sha256':sha256(NEW/'evaluation.json'),'plan_sha256':sha256(NEW/'plan.json'),
            'representatives':[r['id'] for r in representative],
            'worst_compiled':[r['id'] for r in worst],
            'largest_execution_difference':[r['id'] for r in changed],'records':rows}
    atomic_json(NEW/'report.json',report)
    contact(representative,'all_prompts_step10.jpg')
    contact(worst,'worst6_compiled.jpg')
    contact(changed,'largest6_execution_difference.jpg')
    data=[{'id':r['id'],'prompt_id':r['prompt_id'],'prompt':r['prompt'],
           'step':r['step'],'seed':r['seed'],
           'metrics':r['metrics'],'difference':r['compiled_minus_eager']['mean_abs_rgb'],
           **r['media']} for r in rows]
    table=''
    for key,label in NAMES.items():
        meta=evaluation['techniques'][key]
        psnr=f"{overall[key]['pooled_psnr_db']:.3f}" if key in overall else 'Reference'
        ssim=f"{overall[key]['mean_ssim']:.4f}" if key in overall else 'Reference'
        cells=[label,f"{evaluation['median_ms'][key]:.3f} ms",
               f"{meta['parameters']/1e6:.3f} M",f"{meta['computed_frames']} → 21",psnr,ssim]
        table+='<tr'+(' class="highlight"' if key=='compiled' else '')+'>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>'
    steps=''.join('<tr><td>'+s+'</td>'+''.join(f"<td>{by_step[s][key]['pooled_psnr_db']:.3f} / {by_step[s][key]['mean_ssim']:.4f}</td>" for key in overall)+'</tr>' for s in by_step)
    page=(Path(__file__).with_name('fourway_report.html')).read_text()
    timing_note=(f"RTX PRO 6000 Blackwell · BF16 · batch 1 · 21 × 480 × 832 preview. "
                 f"Each technique: 3 fixed inputs × 30 repeats, after 10 warmups per input. "
                 f"Rotating execution order; CUDA synchronized before and after every call. "
                 f"Input preparation, transfers, I/O, metrics, MP4 encoding and cold compilation are excluded.")
    export=evaluation['techniques']['compiled']
    eager=evaluation['techniques']['eager']
    implementation=(f"Eager BF16 and Compiled Quality + BF16 Export use the same network and identical BF16 parameter values. "
         f"Compilation uses torch.compile, mode={plan['compile_mode']}. "
         f"The export changes checkpoint storage from {eager['checkpoint_bytes']/1e6:.2f} MB to {export['checkpoint_bytes']/1e6:.2f} MB; "
         f"both have {export['runtime_weight_bytes']/1e6:.2f} MB of BF16 parameter tensors. "
         f"These are weight bytes, not peak GPU memory. Cold compile and first invocation: {evaluation['cold_compile_and_first_call_seconds']:.2f} s.")
    replacements={'__TABLE__':table,'__STEP_TABLE__':steps,'__TIMING_NOTE__':timing_note,
                  '__IMPLEMENTATION__':implementation,'__ROWS__':json.dumps(data).replace('</','<\\/'),
                  '__PAIR_NOTE__':f"Compiled minus eager: mean per-case PSNR {paired['mean_case_psnr_change_db']:+.5f} dB; SSIM {paired['mean_ssim_change']:+.6f}."}
    for old,new in replacements.items():page=page.replace(old,new)
    (NEW/'index.html').write_text(page)
    (NEW/'README.txt').write_text(
        "Four-technique unseen comparison — 2026-10-07\n\n"
        "Open index.html. All media are local and synchronized in one four-column MP4 per case.\n"
        "Left to right: Native | TAEHV 21 endpoints | Eager BF16 | Compiled Quality + BF16 Export\n"
        "165 predicted-clean latents, 33 trajectories, 11 prompts, 3 seeds and 5 diffusion stages.\n"
        "Native computes 81 frames and selects aligned endpoints 0,4,...,80.\n"
        "TAEHV 21 endpoints disables all temporal upsampling and directly computes 21 frames.\n"
        "All candidate outputs are newly decoded from the same latent; native reference RGB is hash-verified cached output.\n"
        "Native timing uses actual decoding, not loading cached RGB.\n"
        "All four techniques are freshly timed in the same run on the same GPU.\n\n"
        +timing_note+"\n\n"+implementation+"\n\n"
        "Metrics use raw RGB before MP4 encoding: full 21-frame PSNR, SSIM on frames 0/10/20.\n"
        "Native reference is FP16 cached RGB. Parameter counts describe the loaded models after removing their encoders; Native still includes its unused quantization head.\n"
        "This is the same frozen unseen set. No retraining, model selection or new OOD claim.\n"
        "The two earlier reports are preserved and verified by file hashes.\n")
    verified=[]
    for row in rows:
        video=NEW/row['media']['video']
        assert sha256(video)==row['media']['video_sha256']
        assert sha256(NEW/row['media']['poster'])==row['media']['poster_sha256']
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(video),
            '-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=60)
        verified.append({'id':row['id'],'fully_decoded':True,'sha256':sha256(video)})
    atomic_json(NEW/'media_validation.json',{'videos':165,'all_fully_decoded':True,'records':verified})
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','by_step','techniques')},indent=2))

if __name__=='__main__':
    main()
