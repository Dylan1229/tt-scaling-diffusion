# Prompt-switch pilot: cat to dog after step 5

Observed result: the original video shows an orange cat; the step-5 switch and full-target reference show golden dogs in inspected frames 0,16,32,48,64,80. This supports a category change for this ONE case / seed only. It does not establish a latest effective step or a success rate. The requested left-to-right direction is not strictly followed; no formal quality or motion evaluation was run.

Wan2.2 TI2V-5B in T2V mode; seed 0; 50 steps; native UniPC order 2, flow_shift 5; CFG 5; empty negative prompt; max text length 512; 832x480; 81 frames at 24fps. Exact prompts and weights snapshot are recorded in each mode's metadata.json.

original_cat: A for all 50 steps.
switch_step05: A for the first 5 completed steps, then B for the remaining 45. The positive prompt embedding is replaced after callback index 4.
target_dog: B for all 50 steps.

Native latent and scheduler state/history are preserved, without re-noising or scheduler reset. A-conditioned multistep history therefore remains at the initial B step. The A prefix is replayed with the same seed: this pilot actually executes three full trajectories (150 total steps), so it makes no compute-saving claim.

Validation: A vs A-to-B float32 latent hashes match after each of steps 1-5; captured scheduler model-output histories match at step 5; latents differ after step 6. Every callback asserts the conditioning actually used. All outputs contain 81 finite decoded frames. Sigma after step 5 is 0.978284478187561. See validation.json.

comparison.mp4: left original, center step-5 switch, right target.
comparison.png: same ordering top to bottom.
run_pilot.py is the exact worker used. Its generic method field describes the overall experiment; mode and steps_log specify each individual trajectory. make_comparison.py verifies and renders the outputs. Each mode folder retains video.mp4, frames_uint8.npy, latent_after_step05.pt, and metadata.json.

All generation and comparison processing ran on yukelab; preview copies were downloaded for viewing.
