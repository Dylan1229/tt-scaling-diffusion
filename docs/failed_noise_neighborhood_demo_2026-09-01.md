# Failed-noise neighborhood demo result (2026-09-01)

## Summary

The failed parent was locally perturbed at four noise distances. Manual review found **5 successes, 26 failures, and 1 ambiguous case** among the 32 tested neighbors. The closest successful tested neighbor was **n19_a010** at alpha `0.10` with cosine similarity `0.995040774` and RMS distance `0.100062214`.

This means at least one tested local perturbation of the failed parent noise produced a video where the red ball clearly entered the blue goal/box opening. It does **not** estimate the probability of success in the neighborhood.

## Fixed generation setup

- Model: `/data/datasets/fanjiang/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635`
- Input image: `runs/toy_red_ball_i2v_v2/input.png`
- Input SHA-256: `b1aad4e150009199e5a59c2f7867e32d9ae229d5f923dc53ffc857f14f95a8c9`
- Prompt: `Static camera. A red ball moves in a straight horizontal line from left to right, enters through the open left side of a stationary blue box, and stops inside. The ball stays at the same height. The box does not move or change shape.`
- Scheduler: `UniPCMultistepScheduler`
- Size / frames / fps: `832x480`, `81` frames, `24` fps
- Inference steps / guidance: `50`, guidance scale `5.0`
- Parent seed: `0`
- Alphas: `0.02`, `0.05`, `0.10`, `0.20`; eight perturbations per alpha
- Perturbation formula: `neighbor = sqrt(1 - alpha**2) * parent + alpha * epsilon`, where `epsilon` is generated from perturbation seed `10000 + index`.

## Parent reinjection validation

Task 2 reproduced the failed parent and completed the explicit parent-noise reinjection path. The runner would raise `RuntimeError("explicit parent reinjection diverged from the captured run")` on mismatch; the prepare run completed and wrote the parent control video, all-frame sheet, metadata, and `DONE` marker. Manual review of the parent sheet showed the ball approaching the goal but never clearly entering the open interior.

## Manual labels

| Sample ID | Alpha | Perturbation seed | Cosine similarity | RMS distance | Manual label |
|---|---:|---:|---:|---:|---|
| n00_a002 | 0.02 | 10000 | 0.999846458 | 0.020002875 | failure |
| n01_a002 | 0.02 | 10001 | 0.999848962 | 0.020011729 | failure |
| n02_a002 | 0.02 | 10002 | 0.999846101 | 0.019982222 | failure |
| n03_a002 | 0.02 | 10003 | 0.999846995 | 0.020000149 | failure |
| n04_a002 | 0.02 | 10004 | 0.999846935 | 0.020016035 | failure |
| n05_a002 | 0.02 | 10005 | 0.999846697 | 0.020007310 | failure |
| n06_a002 | 0.02 | 10006 | 0.999845862 | 0.020012721 | failure |
| n07_a002 | 0.02 | 10007 | 0.999846458 | 0.020002447 | failure |
| n08_a005 | 0.05 | 10008 | 0.998794973 | 0.050021902 | failure |
| n09_a005 | 0.05 | 10009 | 0.998795271 | 0.050002098 | failure |
| n10_a005 | 0.05 | 10010 | 0.998793542 | 0.050033692 | failure |
| n11_a005 | 0.05 | 10011 | 0.998796225 | 0.050009873 | failure |
| n12_a005 | 0.05 | 10012 | 0.998796761 | 0.049992166 | failure |
| n13_a005 | 0.05 | 10013 | 0.998799264 | 0.049992021 | failure |
| n14_a005 | 0.05 | 10014 | 0.998795867 | 0.050025064 | failure |
| n15_a005 | 0.05 | 10015 | 0.998798609 | 0.049986482 | failure |
| n16_a010 | 0.10 | 10016 | 0.995042264 | 0.100045741 | failure |
| n17_a010 | 0.10 | 10017 | 0.995034814 | 0.100106210 | failure |
| n18_a010 | 0.10 | 10018 | 0.995030463 | 0.100154586 | failure |
| n19_a010 | 0.10 | 10019 | 0.995040774 | 0.100062214 | success |
| n20_a010 | 0.10 | 10020 | 0.995040655 | 0.100079432 | ambiguous |
| n21_a010 | 0.10 | 10021 | 0.995032847 | 0.100119136 | success |
| n22_a010 | 0.10 | 10022 | 0.995028019 | 0.100193135 | failure |
| n23_a010 | 0.10 | 10023 | 0.995031357 | 0.100169264 | failure |
| n24_a020 | 0.20 | 10024 | 0.979833424 | 0.201071933 | failure |
| n25_a020 | 0.20 | 10025 | 0.979833245 | 0.201059982 | success |
| n26_a020 | 0.20 | 10026 | 0.979827344 | 0.201089352 | failure |
| n27_a020 | 0.20 | 10027 | 0.979875147 | 0.200849429 | failure |
| n28_a020 | 0.20 | 10028 | 0.979859769 | 0.200950280 | failure |
| n29_a020 | 0.20 | 10029 | 0.979865074 | 0.200899050 | success |
| n30_a020 | 0.20 | 10030 | 0.979872525 | 0.200881734 | failure |
| n31_a020 | 0.20 | 10031 | 0.979818404 | 0.201143891 | success |

## Counts by alpha

| Alpha | Success | Failure | Ambiguous |
|---:|---:|---:|---:|
| 0.02 | 0 | 8 | 0 |
| 0.05 | 0 | 8 | 0 |
| 0.10 | 2 | 5 | 1 |
| 0.20 | 3 | 5 | 0 |
| **Total** | **5** | **26** | **1** |

Successful sample IDs: `n19_a010`, `n21_a010`, `n25_a020`, `n29_a020`, `n31_a020`.

Ambiguous sample IDs: `n20_a010`.

## Closest successful neighbor

Closest successful tested neighbor by cosine similarity to the parent noise:

- Sample ID: `n19_a010`
- Alpha: `0.10`
- Perturbation seed: `10019`
- Cosine similarity: `0.995040774`
- RMS distance: `0.100062214`
- Manual observation: after the goal becomes a blue box, the ball is visibly inside the opening.

## Artifacts

Remote run directory:

`/data/datasets/peihao/tt-scaling-diffusion/.worktrees/noise-neighborhood-demo/runs/toy_red_ball_i2v_v2/noise_neighborhood_v1`

Local review copies:

`runs/toy_red_ball_i2v_v2/noise_neighborhood_v1`

Review labels:

`runs/toy_red_ball_i2v_v2/noise_neighborhood_v1/review_labels.json`

Comparison montage:

`runs/toy_red_ball_i2v_v2/noise_neighborhood_v1/comparison_montage.jpg`

## Limitations

- Labels are manual semantic judgments from the 81-frame sheets; there is no automated semantic scorer.
- Only eight perturbations were tested at each alpha.
- The result is evidence that some tested neighbors succeeded, not a probability claim about the whole noise neighborhood.
- Several successful cases and the ambiguous case show visible goal-shape drift into a box-like/perspective form, so the success finding should be read together with the montage and raw sheets.
