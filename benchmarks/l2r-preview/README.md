# Fast latent-to-RGB video previews

This repository contains our trained Quality preview decoder, implementation, and complete comparison reports. The latest report compares Native, TAEHV 21 endpoints, Eager BF16, and Compiled Quality + BF16 Export.

## Download and view

The [complete release bundle](https://github.com/Dylan1229/tt-scaling-diffusion/releases/tag/l2r-preview-20261007) contains all three HTML reports and videos, trained Quality weights, source, and raw results. Extract `l2r-preview-20261007.zip`, then open `l2r-preview-20261007/docs/index.html` in a browser.

This Git branch contains the implementation and compact evidence. Full media and checkpoints are Release assets, following the parent repository's artifact policy. No Git LFS or remote video server is needed to view the downloaded bundle.

- [Four-way metrics and timing evidence](reports/20261007/)
- [All 11 prompts at diffusion step 10](reports/20261007/all_prompts_step10.jpg)

## Full release bundle contents

- `docs/`: latest four-way report, all 165 comparison videos, contact sheets, raw metrics, timing samples, frozen plan, source snapshots, and checksums.
- `l2r_unseen_20261007/`: preserved original unseen report.
- `l2r_unseen_compiled_20261007/`: preserved compiled comparison report.
- `l2r_preview/`, `examples/`, `tests/`: decoder, training, evaluation, report generation, and tests.
- `weights/`: trained Quality FP32 checkpoint and its BF16 export.
- `project_notes/experiment_history.md`: original project README and historical experiment notes.
- `publication.json`, `publication_checksums.json`: package scope, provenance, and SHA-256 hashes.

Training videos, latent caches, raw reference tensors, and generator/teacher weights are not bundled. Scripts and reports record their provenance and original experiment paths; reproduction requires supplying those assets.

## Measured comparison

RTX PRO 6000 Blackwell; PyTorch 2.7.1 + CUDA 12.8; BF16; batch 1. Same latent shape `[1, 48, 21, 30, 52]`; displayed RGB output is 21 frames at 480 x 832.

| Technique | Parameters loaded | Median decode time | Mean PSNR / SSIM |
| --- | ---: | ---: | ---: |
| Native | 555,060,892 | 4,526.33 ms | Reference |
| TAEHV 21 endpoints | 9,903,052 | 19.32 ms | 28.239 dB / 0.86146 |
| Eager BF16 | 1,904,580 | 5.20 ms | 29.129 dB / 0.86401 |
| Compiled Quality + BF16 Export | 1,904,580 | 2.40 ms | 29.131 dB / 0.86417 |

Timing uses three fixed inputs, ten warmups and thirty measured calls per input per technique, rotating technique order and synchronizing CUDA. It includes decoder forward and RGB conversion/clamping; Native also selects endpoints after decoding all 81 frames. The other techniques produce 21 frames directly. Input preparation, transfer, compilation, metrics, and video encoding are excluded. See [raw timing samples](reports/20261007/evaluation.json).

Quality is measured against cached raw Native RGB on 165 held-out predicted-clean latents: 11 unseen prompts x 3 seeds x 5 diffusion steps. These are 33 trajectories, not 165 independent videos. The Quality model was trained on 4,500 OpenVid video pairs, with 250 validation and 250 test clips. This validates preview reconstruction on the tested set; it does not establish early-failure detection accuracy.

## Decode a latent

Use a Python environment with a CUDA-compatible PyTorch build. Run from the extracted release directory, or from `benchmarks/l2r-preview` after obtaining its weights from the release. Install with `python -m pip install -e .`.

~~~bash
python -m l2r_preview.decode \
  --checkpoint weights/quality_bf16.pt \
  --latent /path/to/predicted_clean.pt \
  --output outputs/preview.mp4 \
  --device cuda:0 --precision bfloat16 \
  --chunk-frames 0 --compile-mode max-autotune-no-cudagraphs
~~~

The input should be a Wan2.2 TI2V-5B predicted-clean latent in diffusion normalization (the CLI default). For intermediate diffusion states, first form the model's predicted clean sample, or supply matching velocity and sigma arguments. A raw noisy latent alone is a different input distribution. The adapter denormalizes once before decoding.

The CLI includes first-use compilation. To obtain steady-state latency, keep the model and compiled callable alive across requests:

~~~python
import torch
from l2r_preview.io import load_checkpoint, load_latent
from l2r_preview.runtime import make_preview_decoder

model, spec, _ = load_checkpoint('weights/quality_bf16.pt')
model = model.to(device='cuda', dtype=torch.bfloat16)
decode = make_preview_decoder(model, 'max-autotune-no-cudagraphs')
latent = spec.denormalize(load_latent('/path/to/predicted_clean.pt'))
latent = latent.to(device='cuda', dtype=torch.bfloat16)
with torch.inference_mode():
    rgb = decode(latent)  # B, 3, T, H, W; RGB in [0, 1]
~~~

Run a few warmups before timing; synchronize CUDA around measured calls. Compilation is shape-dependent.

## Architecture and provenance

The current Quality model is `PyramidPreviewDecoder`: a low-resolution feature trunk with causal depthwise temporal mixing, progressive spatial subpixel upsampling, and no temporal upsampling. It decodes 21 temporal endpoints instead of reconstructing all 81 frames. The FP32 and BF16 checkpoints contain the same 1,904,580-parameter architecture; exporting BF16 halves weight storage relative to FP32 without reducing parameter count.

The original paper-inspired causal L2R prototype is also preserved in the source and historical notes. This project is an independent implementation and the Quality architecture is our subsequent design, not the paper authors' released implementation.

The comparison uses [TAEHV](https://github.com/madebyollin/taehv) at commit `011dfc2112197741c540e0bdd5b7b67bcc930771`, with `taew2_2.safetensors`. Use `examples/fetch_taehv.py` to obtain upstream assets under their own license. No third-party weights are redistributed in this package.

Preserved reports retain their original titles, timings, and configurations. Use the latest four-way report for the current comparison; historical timings and the original TAEHV temporal configuration differ.
