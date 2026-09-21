# Step 40 -> step 4 re-noising feasibility case

Source: original cat trajectory, seed 0, latent after 40 completed denoising steps.
Intervention: newly sampled independent standard Gaussian noise, seed 20260921.
Conditioning after intervention: golden retriever dog prompt throughout.
Model/settings: Wan2.2 TI2V-5B in T2V mode; 832x480, 81 frames, 24 fps, CFG 5.0; original 50-step UniPC schedule.

## Re-noising

The scheduler uses the flow convention x_sigma = (1-sigma) x0 + sigma epsilon.

- Source sigma after step 40: 0.55678790807724
- Target sigma after step 4: 0.982924222946167
- a = (1-sigma_target)/(1-sigma_source) = 0.03852732667954569
- b = sqrt(sigma_target^2 - a^2 sigma_source^2) = 0.9826901134503093
- x_restart = a * x_step40 + b * fresh_random_noise

This matches nominal flow marginal signal/noise coefficients. It does not assume the finite-solver source latent has a known sample-specific clean/noise decomposition. No saved step4 latent or prior noise is loaded.

## Denoising restart

Create a fresh UniPC scheduler; retain exactly timesteps[4:] and sigmas[4:] from the original 50-step schedule, including terminal sigma0. Clear multistep history and run 46 denoising updates with the dog prompt, covering original schedule steps5 through50. The first update warms up at solver order1.

## Validation

- Saved source latent hash equals the original all-cat trajectory latent hash at completed step40.
- Target and negative prompt embedding hashes match the earlier experiment.
- Sliced-schedule restart is bit-identical to full-schedule begin_index4 over 46 synthetic updates (maximum absolute difference 0).
- Actual run contains exactly46 updates, all dog conditioned; final sigma0.
- All81 output frames are finite; output shape is 81x480x832x3.
- Generation and decoding: 61.02 seconds.
- One new random-noise sample; this run does not estimate a success probability.

## Artifacts

- video.mp4: the new result alone.
- comparison.mp4: synchronized comparison, left original cat, middle prompt switch at40 without noise, right current re-noising intervention.
- comparison.jpg: same three conditions at frames0/16/32/48/64/80.
- all_frames.jpg: all81 frames of the new result in chronological order.
- metadata.json: settings, checksums and per-step trace.
- run_case.py: runnable experiment source.
- fresh_random_noise.pt / renoised_latent.pt: reproducible intervention tensors, retained remotely.

Controls are reused from the preceding experiment. No extra model generations were performed for this case.

## Visual result

Recognizable golden retriever dog throughout all81 inspected frames. The existing prompt-switch-at40 control remains an orange tabby cat.

Not a clean-quality pass: conspicuous black/white shoe-like artifacts and distorted leg anatomy. Direction and view do not fully match the side-view left-to-right prompt.

One-case feasibility of late-state semantic change by fresh random re-noising demonstrated; normal anatomical quality and reliability across seeds not established.
