# Failed-Noise Gaussian Neighborhood Demo Design

## Goal

Test whether a small change to a semantically failed initial noise can move the same deterministic Wan denoising trajectory into a visibly successful result.

This is a quick feasibility demo, not an estimate of neighborhood success probability.

## Fixed generation setup

Reuse the existing `toy_red_ball_i2v_v2` setup:

- Wan 2.2 TI2V-5B
- Existing input image and prompt
- Failed parent: seed 0
- 480×832, 81 frames
- 50 inference steps
- Guidance scale 5.0
- The model, image conditioning, prompt, scheduler, timesteps, and all generation settings remain unchanged

Only the initial Gaussian noise changes between neighborhood trials. The existing seed 4 result is retained as a visual example of success; it is not rerun or included in the neighborhood results.

## Neighborhood construction

Capture seed 0's exact initial latent noise, `z_fail`, from the normal pipeline path. Generate 32 reproducible neighbors: eight independent samples at each perturbation level

- 0.02
- 0.05
- 0.10
- 0.20

For perturbation level `a` and independent standard Gaussian noise `e`, construct

`z_neighbor = sqrt(1 - a²) * z_fail + a * e`.

This creates a correlated Gaussian neighbor without inflating the expected noise variance. Record the perturbation seed, actual RMS distance, cosine similarity, and norm ratio for every neighbor.

## Data flow

1. Load the existing Wan pipeline and fixed red-ball inputs.
2. Run seed 0 through the ordinary latent-preparation path while capturing its initial noise.
3. Save that noise and pass it back explicitly for an unchanged parent run.
4. Confirm the explicit parent run still shows the known failure. Stop if it does not.
5. Construct and save the 32 neighbor noises deterministically.
6. Denoise each neighbor through the same deterministic UniPC path.
7. Save each video, an all-frame review sheet, and its metadata.
8. Review every result and summarize the labels.

A one-neighbor pilot runs before the full set. Completed outputs are skipped on restart so an interruption does not waste finished generations.

## Visual evaluation

There is no automatic semantic scorer.

Review every video across its full duration. Label it:

- **Success:** the red ball clearly enters the blue goal at least once.
- **Failure:** it never clearly enters.
- **Ambiguous:** the event cannot be judged confidently.

Later exit, imperfect stopping, or failure to remain inside does not invalidate an otherwise clear entry. The report includes the successful videos, ambiguous cases, counts by perturbation level, and the closest observed success. With eight trials per level, counts are descriptive only.

## Outputs

The experiment produces:

- The captured failed parent noise
- The explicit parent-control video
- 32 neighbor noise tensors and videos
- One all-frame review sheet and metadata record per video
- A manifest mapping each output to perturbation level and random seed
- A concise visual-review summary and comparison montage

Write outputs atomically and mark a sample complete only after its video, review sheet, and metadata all exist.

## Validation and failure handling

Before the full run:

- Check the perturbation formula on a small tensor for reproducibility, shape preservation, and the zero-perturbation identity.
- Verify that explicitly injecting the captured parent noise reproduces the failed behavior.
- Verify all generation settings match the original seed 0 metadata.
- Run and inspect one neighbor end to end.

Abort rather than mixing results if the model version, input image, prompt, scheduler, dimensions, step count, or guidance differs. Preserve partial completed work and resume only missing samples.

## Execution

All model execution occurs on `yukelab`. If multiple suitable GPUs are available, create the parent artifact first and then split independent neighbor generations across them; otherwise run serially. Parallelism must not change any sample's inputs or denoising path.

The compute server was unreachable when this design was written, so implementation can proceed locally but validation and generation must wait for remote access.

## Out of scope

- Wan 2.1 replication
- Automatic success scoring
- Nearby integer seeds as a substitute for nearby latent noise
- Perturbations after denoising begins
- Statistical claims about success probability
- A reusable search framework or production integration

## Completion criteria

The demo is complete when the parent failure and all 32 neighbors have valid outputs, every video has a visual label, and the report states whether any perturbed initial noise produced a clear success and how close the nearest such noise was.
