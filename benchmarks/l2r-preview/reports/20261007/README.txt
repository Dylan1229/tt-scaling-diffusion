Four-technique unseen comparison — 2026-10-07

Open index.html. All media are local and synchronized in one four-column MP4 per case.
Left to right: Native | TAEHV 21 endpoints | Eager BF16 | Compiled Quality + BF16 Export
165 predicted-clean latents, 33 trajectories, 11 prompts, 3 seeds and 5 diffusion stages.
Native computes 81 frames and selects aligned endpoints 0,4,...,80.
TAEHV 21 endpoints disables all temporal upsampling and directly computes 21 frames.
All candidate outputs are newly decoded from the same latent; native reference RGB is hash-verified cached output.
Native timing uses actual decoding, not loading cached RGB.
All four techniques are freshly timed in the same run on the same GPU.

RTX PRO 6000 Blackwell · BF16 · batch 1 · 21 × 480 × 832 preview. Each technique: 3 fixed inputs × 30 repeats, after 10 warmups per input. Rotating execution order; CUDA synchronized before and after every call. Input preparation, transfers, I/O, metrics, MP4 encoding and cold compilation are excluded.

Eager BF16 and Compiled Quality + BF16 Export use the same network and identical BF16 parameter values. Compilation uses torch.compile, mode=max-autotune-no-cudagraphs. The export changes checkpoint storage from 7.63 MB to 3.82 MB; both have 3.81 MB of BF16 parameter tensors. These are weight bytes, not peak GPU memory. Cold compile and first invocation: 32.73 s.

Metrics use raw RGB before MP4 encoding: full 21-frame PSNR, SSIM on frames 0/10/20.
Native reference is FP16 cached RGB. Parameter counts describe the loaded models after removing their encoders; Native still includes its unused quantization head.
This is the same frozen unseen set. No retraining, model selection or new OOD claim.
The two earlier reports are preserved and verified by file hashes.
