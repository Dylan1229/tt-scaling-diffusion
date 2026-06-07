# features

DINOv2 feature extraction from decoded posterior-mean frames. Both runners read
a seed dir of `step_*.mp4` via `--dino-input-frames-dir`.

| module | computes | writes |
|---|---|---|
| `cls_similarity.py` | DINOv2 CLS feature per frame; diagonal / frame-neighbor / posterior-neighbor cosine-similarity matrices | `posterior_mean_features.npy` + `posterior_mean_*_similarity.npy` (+ PNG heatmaps) |
| `patch_features.py` | DINOv2 patch tokens on a stride-subsampled frame grid | `posterior_mean_patch_features.npy` (`T × N_sub × P × D`, float16) |

```
python -m ttsd.runners.features.cls_similarity --dino-input-frames-dir <frames>/<seed> --output-dir <cls>/<seed>
python -m ttsd.runners.features.patch_features --dino-input-frames-dir <frames>/<seed> --output-dir <patch>/<seed>
```

Multi-GPU wrappers: `scripts/extract_cls_similarity_batch.sh`, `scripts/extract_patch_features_batch.sh`.
