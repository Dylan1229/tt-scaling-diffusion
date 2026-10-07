# L2R preview decoder

Standalone, independent implementation of the preview-decoder architecture in
[Early Failure Detection and Intervention in Video Diffusion Models](https://arxiv.org/abs/2603.14320)
(Fig. 3, Fig. S1, Sec. 3.1 and Appendix A.2). Scope ends at RGB previews.
No alignment scorer, failure detector, search, or intervention is included.

The [official repository](https://github.com/kaist-ami/Early-failure-video-diffusion)
was still README-only on 2026-10-01. This is not author
code or an exact reproduction of their unreleased weights/training recipe.

## Architecture

Predicted clean latent -> causal residual block -> learned spatial upsampling
-> causal residual block -> RGB projection.

- Tensor layout: [batch, channels, latent frames, latent height, latent width].
- Each residual block: three causal 2x3x3 convolutions on the main branch,
  one on the skip branch, and ReLU activations, following Fig. S1(b).
- Learned upsampling uses current/past 3x3 neighborhoods and does not add frames.
- Projection: 9x9 then 3x3 2D convolutions, applied per frame.
- Hidden width remains equal to the input latent channel count.
- RGB output uses [-1,1] units; clamp only for export.
- Causal chunked decoding preserves a seven-latent-frame history.

### Explicit implementation choices

The paper does not fully specify the upsampling tensor arrangement or channel
sharing. Here a shared Conv3d(1, scale^2, 2x3x3) predicts subpixel values per
latent channel, then pixel shuffle increases spatial size. Filters start as
nearest-neighbor interpolation and are learned during training. We choose
replicate causal/spatial padding and a ReLU between the RGB projection layers.
These are disclosed engineering choices, not claims about the authors' code.

| Configuration | Latent channels | Spatial scale | Parameters |
|---|---:|---:|---:|
| Paper-shaped setup | 16 | 8 | 59,395 |
| TTS Wan 2.2 TI2V-5B adaptation | 48 | 16 | 524,995 |

The default rounds to the paper's 0.059M count; matching that rounded number
does not establish that our internal implementation is identical.

## Difference from the actual Wan video decoder

The TTS Wan 2.2 VAE uses 48-channel latents, decoder base width 256 and stage
widths up to 1024, multiple residual stages, normalization and temporal/spatial
upsampling. The loaded native decoder has 555,049,228 parameters, versus
524,995 in this adaptation; parameter count alone is not a latency ratio. Its decoder also handles a final spatial unpatchification.
L2R has only two narrow residual blocks, one spatial upsampling operation and
an RGB head. It preserves latent frame count. An 81-frame Wan video has 21
latent frames, so this adaptation produces 21 preview frames, not 81.
This targets semantic previews rather than final reconstruction fidelity.

## TTS bridge

Confirmed against ttsd/models/wan22_adapter.py and
ttsd/runners/generate/decode_latents.py on Yukelab.

- Feed captured posterior_means/step_NNN.pt (predicted clean samples), or a
  genuinely final clean latent. Intermediate latents/step_NNN.pt are noisy
  post-step states and are not equivalent inputs.
- For TTS's flow convention, clean_hat = pre_step_latent - sigma * velocity.
  The three inputs must refer to the same pre-step state.
- For UniPC, prefer TTS's existing scheduler-converted posterior mean capture.
- Undo Wan diffusion normalization exactly once: z_vae = z_diffusion * std + mean.
  The mean/std values and compression scales travel with each checkpoint.
- Do not transfer a Wan 2.1 or CogVideoX checkpoint to Wan 2.2 just because
  the architecture looks similar: their VAE latent spaces differ.

## Usage

Use an existing PyTorch environment. Installation is optional when running
from the repository root; no changes to the shared TTS environment are needed.

```bash
python -m pytest -q
python -m l2r_preview.decode \
  --checkpoint outputs/smoke/decoder_4000.pt \
  --latent /path/to/posterior_means/step_009.pt \
  --output outputs/preview.mp4 --device cuda:0 --source-fps 16
```

The export uses source_fps / temporal_scale, preserving the spacing between
sampled frames. There is no temporal interpolation or scoring.

Python integration:
```python
import torch
from l2r_preview.io import load_checkpoint

model, spec, metadata = load_checkpoint("decoder.pt", device="cuda:0")
z_vae = spec.denormalize(predicted_clean_latent.float())
with torch.inference_mode():
    rgb = model(z_vae.to("cuda:0"))  # [B, 3, latent_frames, H, W]
```

## Training and smoke demo

A fresh network cannot meaningfully decode a latent. Trained weights are
required. The paper trains separate converters for each VAE using 33k real
videos encoded by that VAE. It does not provide a complete optimizer/loss recipe.

Our bounded demo instead fits a few cached diffusion latents against their
native VAE-decoded RGB frames. This is a teacher-fitting smoke test, not the
paper's real-video training procedure or evidence of broad generalization.
The loss is RGB MSE with AdamW, an explicitly chosen reproduction setting.

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.prepare_tts_pairs \
  --model-path /path/to/cached/Wan2.2-TI2V-5B-Diffusers \
  --train-latent /path/to/clean_training_latent.pt \
  --holdout-latent /path/to/another_clean_latent.pt \
  --output outputs/smoke/pairs --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -m l2r_preview.train \
  --data outputs/smoke/pairs --output outputs/smoke/decoder_4000.pt \
  --steps 4000 --max-seconds 120 --learning-rate 0.0005 --device cuda:0
```

Pair files contain native-VAE-space latent and temporally aligned RGB tensors.
Training reads only train_*.pt; holdout_*.pt are reserved for reconstruction
checks and never selected by the training loader. The checkpoint and report
record provenance, training scope, elapsed time and memory usage.

## Verified local experiment

The repository and primary artifacts are on Yukelab at
`/data/datasets/peihao/l2r-preview`. Reuse
`/data/datasets/peihao/tt-scaling-diffusion/.venv/bin/python`; run project
commands over SSH. No TTS source files or shared dependencies were modified.

The demo fits final latents from three cached videos (swimmer, coffee drinker,
bird/cat), then previews a held-out bear/zebra final latent and a held-out
swimmer clean prediction from step index 9. The intermediate example shares
a prompt with a training clip but comes from a different run; it is not a
held-out prompt. All inputs are [1,48,21,30,52]; output is [1,3,21,480,832].

- `outputs/smoke/pairs/manifest.json`: source paths, hashes, frame alignment,
  VAE parameter counts, and native decode execution records.
- `outputs/smoke/decoder_4000.pt`: demo checkpoint, including normalization.
- `outputs/smoke/decoder_4000.training.json`: exact fit settings and loss history.
- `outputs/smoke/preview_4000/index.html`: videos with native references.
- `outputs/smoke/preview_4000/contact_sheet.jpg`: first/middle/last frames.
- `outputs/smoke/cli_preview.mp4`: direct CLI output from the saved
  intermediate predicted-clean latent, without loading the native VAE.

The initial 600-step attempt is retained in `outputs/smoke/preview` and
`outputs/smoke/decoder.pt`. Its colors/details were weak, prompting the
longer, lower-learning-rate fit. Neither checkpoint is a general pretrained
decoder. Quality still needs broader training, especially for fine detail
and unseen VAE latents. No paper-level quality or speedup is established.

To rebuild visual comparisons (there is no scoring stage):
```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.render_comparison \
  --checkpoint outputs/smoke/decoder_4000.pt --data outputs/smoke/pairs \
  --output outputs/smoke/preview_4000 --device cuda:0 \
  --pair train_01.pt --pair holdout_00.pt --pair holdout_01.pt
```

The CPU tests check temporal causality, exact overlap chunking, channel/time
layout, normalization, flow-prediction sign, checkpoint round trips, and
gradient/training behavior. Small shape tests exercise both configurations.

## Pretrained TAEHV comparison (2026-10-02)

We tested upstream TAEHV's `taew2_2` checkpoint for Wan2.2 TI2V-5B,
pinned to `011dfc2112197741c540e0bdd5b7b67bcc930771`.
"5B" names the diffusion generator; the tiny decoder has 9,923,532 parameters
(11,418,108 including its unused encoder), versus 555,049,228 for Wan's native
decoder. The MIT license and upstream citation accompany the results.

TAEHV accepts normalized diffusion-space **NTCHW** and returns RGB in [0,1].
Do not apply the native VAE's mean/std denormalization to its input.
The native VAE accepts denormalized **NCTHW**. Default TAEHV restores all
81 frames from 21 latent frames, including its three-frame causal prefix trim.

Six cached latents decoded successfully: three final videos and swimmer clean
predictions at steps 5, 10, and 30. Four had previously cached uncompressed
native-VAE references. On their 21 corresponding reference frames, mean
per-video PSNR was 30.874 dB; mean SSIM across nine frames per video was 0.9081.
These are native-decoder agreement measurements, not ground-truth video quality
or evidence about downstream selection decisions.

Observed TAEHV times were 85 ms parallel (six-clip median of three-repeat
medians), and 128 ms sequential (coffee clip only). Peak allocated GPU memory
was 4.274 GiB parallel and 0.447 GiB sequential with decoder-only weights.
**Another job shared GPU 0 at launch and during measurement. These timings are
provisional, and no new native-VAE GPU benchmark was launched. No fair runtime
speedup ratio is established.** Future benchmark stages reject a busy GPU.

All GPU work used one GPU and existing dependencies; the final comparison was
rendered on CPU from cached references. Source videos, raw decoded tensors,
checkpoint hashes, logs, and provenance remain under
`outputs/taehv_20261002/`; `index.html` plays locally after download.

Reproduce the quality report from the cached outputs:
```bash
python -m examples.fetch_taehv
CUDA_VISIBLE_DEVICES="" python -m examples.render_taehv_cached
```

For a fresh comparison when an assigned GPU is idle, use its UUID:
```bash
python -m examples.compare_taehv --stage prepare
CUDA_VISIBLE_DEVICES=GPU_UUID python -m examples.compare_taehv --stage taehv
CUDA_VISIBLE_DEVICES=GPU_UUID python -m examples.compare_taehv --stage native
CUDA_VISIBLE_DEVICES="" python -m examples.compare_taehv --stage render
```

The `benchmark` optional dependencies include scikit-image, safetensors,
Diffusers, and tqdm; the existing Yukelab environment already contains them.
The GPU preflight and external runtime bounds used for this experiment are
documented in `outputs/taehv_20261002/validation.json`.

## OpenVid-5K training (2026-10-06)

The real-video dataset is encoded and verified at
`/data/datasets/peihao/openvid-5k/encoding/wan22_5b_bf16`.
The new manifest loader reads native, unnormalized BF16 safetensors and decodes
the corresponding source MP4 frames on demand. It does not load a VAE or use
the previous three-clip teacher-fit weights.

Initial run: `outputs/openvid5k_20261006`, 524,995 parameters, 4,500 training
clips, 250 validation clips, 250 test clips. Train from scratch for 90,000
optimizer updates (20 passes through the training IDs), one independently
sampled crop per video per pass. Each target covers two causal endpoint frames
and an 8x12 latent-cell core (128x192 RGB), with seven past latent frames and
the complete spatial receptive-field halo. Crops at real video boundaries
retain the model's ordinary padding. No geometric or temporal augmentation
changes the encoded latent/RGB correspondence.

Recipe chosen for this baseline, not claimed as the authors' training recipe:
AdamW, RGB MSE, peak LR 3e-4, 500-update warmup, cosine decay to 3e-5,
weight decay 1e-4, gradient-norm clip 1.0, float32 parameters and BF16 autocast.
All 250 validation clips use deterministic held-out patches once per epoch.
`best.pt` is selected by validation MSE; the 250 test patches are evaluated
only after training finishes. Reported PSNR clips predictions to [-1,1].
These patch metrics do not establish full-video or intermediate-diffusion
preview quality; those require visual inspection and a TAEHV comparison.

Run on Yukelab, using the existing environment and GPU 2:
```bash
bash /data/datasets/peihao/l2r-preview/outputs/openvid5k_20261006/run.sh
# After an interruption, use the same source/configuration:
bash /data/datasets/peihao/l2r-preview/outputs/openvid5k_20261006/run.sh --resume
```

The runner pins one GPU UUID, uses two bounded CPU readers, reserves 50 GiB
free disk, checks for competing GPU processes every minute, and has a
three-hour per-invocation budget. It saves model, optimizer, RNG and cursor
state atomically every 500 updates and on controlled interruption.
`config.json` records manifests and source hashes; the source snapshot is
under `source/`. `progress.json`, `metrics.jsonl`, `train.log`,
`last.pt`, `best.pt`, epoch checkpoints and final `test.json` are local
to the run directory. Existing artifacts and shared environments are preserved.

Preflight:
- All 12 CPU tests passed, including real MP4 frame alignment, native latent
  units, split-integrity checks and crop/full-frame decoder equivalence.
- Two fixed training crops: MSE 0.29942 -> 0.01212 in 400 updates
  (12.76 seconds, 1.14 GB peak allocated VRAM). This is an overfit check.
- A bounded training job stopped at update 29, then loaded its model/optimizer
  and resumed to update 120 successfully.
- Preflight outputs are in `outputs/openvid5k_preflight_20261006`,
  `outputs/openvid5k_pipeline_check_20261006` and
  `outputs/openvid5k_resume_check_20261006`.

## OpenVid-trained preview comparison (2026-10-06)

The completed 90,000-update decoder was checked on six held-out real clips and
six saved diffusion latents, including steps 5/10/30. Results, raw RGB, source
hashes and the self-contained playback page are under `outputs/openvid5k_review_20261006/`.
Full-spatial previews preserve recognizable subjects/scenes but lose detail.
This implementation has no speed or memory advantage over TAEHV: 107 ms for
21 frames vs TAEHV 47 ms for 81 frames; peak GPU allocations are 5.75 vs 4.27 GiB.
These are BF16, batch-1 decode-only medians on an otherwise idle GPU, with
different output-frame workloads. Six-clip mean PSNR vs real RGB is 29.29 dB
for L2R and 30.97 dB for TAEHV. See the run README and report for full protocol.

## Efficient pyramid prototype (2026-10-06)

`l2r_preview/pyramid.py` introduces two untrained progressive preview decoders.
Low-resolution causal/spatial processing is followed by decreasing-width x2
subpixel stages, ending directly in RGB. Complete 21x480x832 BF16 forwards
measured 1.93 ms / 0.132 GiB (compact, 0.350M parameters) and 3.28 ms / 0.229 GiB
(balanced, 1.029M), compared with 20.35 ms / 1.149 GiB for upstream TAEHV in its
21-frame preview mode. These are eager, batch-1, warmed decode-only timings,
not quality measurements. Five new causality/context/gradient tests pass.
See `outputs/pyramid_design_20261006/README.txt` and JSON reports for provenance
and training-integration requirements. No new training was launched and the
existing trained decoder/loader were not replaced.


## Trained pyramid exploration (2026-10-06)

See outputs/pyramid_search_20261006/README.txt and web/index.html for four trained variants, usage, raw results and limitations.

Recommended default: quality_full/best.pt (1.905M parameters), 5.375ms versus100.462ms for the old decoder. Full-test PSNR/SSIM:27.394/0.8407 versus27.369/0.8390. Wide (5.352M):27.726/0.8506 at9.056ms. TAEHV21:27.355/0.8599 at20.216ms. TAEHV full81:28.953/0.9004 at46.191ms. Measurements use BF16, batch1, RTX PRO6000 Blackwell and250 held-out videos. Timing is decoder-only.

The loader supports old L2R and pyramid checkpoints. Use --precision bfloat16 --chunk-frames 0 for full parallel decoding. Historical untrained architecture benchmarks above remain historical; the new experiment contains trained weights.


## Unseen diffusion validation and optional compiled runtime (2026-10-07)

Frozen Quality was tested on33 cached trajectories from11 previously unused prompts,3seeds and5steps (165latents). The evaluator compares the native VAE, Quality and TAEHV on the same predicted-clean latent; see `outputs/unseen_diffusion_20261007/index.html` and its frozen plan. This is a reconstruction pilot, not failure-detector validation.

Optional persistent inference uses `l2r_preview.runtime.make_preview_decoder(model, "max-autotune-no-cudagraphs")` under `torch.inference_mode()`. The callable consumes native unnormalized BF16 NCTHW latents and returns clamped RGB[0,1]. The CLI exposes `--compile-mode` with CUDA and `--chunk-frames 0`. Timing excludes cold compilation; reuse the runtime instead of constructing it at every step. Default behavior remains eager.

Deployment weights are in `outputs/compression_after_unseen_20261007/checkpoints/quality_bf16.pt`. This export preserves original BF16 inference tensors while reducing checkpoint disk size, not parameter count. A bounded Balanced distillation attempt did not improve validation and was not promoted. Full measurements,logs,configuration,and source are saved in the two output directories.
