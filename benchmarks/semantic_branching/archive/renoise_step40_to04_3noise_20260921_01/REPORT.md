# Three-noise comparison

Only fresh noise changes; source step40 latent, target step4 noise level, golden retriever prompt and46-step restart are fixed.

All three outputs are recognizable dogs with golden-retriever appearance. No visible cat reversion was observed in the81-frame reviews.

| Sample | Noise seed | Main differences |
| --- | --- | --- |
| 1 | 20260922 | Orange-brown coat, raised-leg gait, lower tail, patchier grass with pale streaks. |
| 2 | 20260923 | Longer side view initially, lower sweeping feathered tail; turns away and recedes. |
| 3 | 20260924 | Rounder torso, raised pale tail, rear-three-quarter view; more conspicuous hindquarter/body distortion. |

All tend to face left/away, with a broadly similar orientation to the original cat trajectory. They do not fully satisfy the left-to-right side-view motion prompt. The videos are visually distinct but not anatomically flawless. This one-source, three-noise comparison does not establish a general success rate or isolate why the broad orientation remains similar.

See comparison.mp4, contact_sheet.jpg and per-seed videos. validation.json verifies fixed parameters and distinct noise/input/output hashes.
