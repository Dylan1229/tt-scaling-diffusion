# runners

CLI entry points, invoked as `python -m ttsd.runners.<sub>.<module>`. Heavy
dependencies (Wan, DINOv2, VBench) load lazily, so `--help` and imports work
without a GPU.

Stage flow:

```
generate (baseline)  → runs/baseline/   (video.mp4, latents/, posterior_means/)
        │
        └─ posterior_means/ ──decode_latents──► runs/dino_input_frames/   (DINOv2 input frames)
                                                    │
                            ┌───────────────────────┴───────────────────────┐
                     features/cls_similarity                        features/patch_features
                     → runs/cls_features/                           → runs/patch_features/
                            │                                                │
                            └───────────────► analysis/ , report/ ◄──────────┘
```

| subfolder | what it holds |
|---|---|
| `generate/` | sweep generation + VAE latent decode |
| `features/` | DINOv2 CLS and patch feature extraction from decoded frames |
| `analysis/` | per-seed scalar metrics aligned against VBench scores |
| `report/` | figures for the DINOv2 feature-property report |
| `utilities/` | shared loaders (`seed_vbench_loaders`) + ranking helper (`ranking`) imported by analysis/report |

VBench scores come from `ttsd.eval.vbench`; the generation model wrapper is
`ttsd.models.wan22_adapter`.
