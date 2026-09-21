# Full dog prompt control

Dog prompt throughout all50 original denoising steps. Initial float32 latent is the exact saved fresh Gaussian noise from the preceding step40-to4 re-noising run (seed20260921), verified by SHA256. No cat latent or cat conditioning is used.

Fresh full50 UniPC scheduler; same model snapshot, CFG5, 832x480,81 frames,24fps and empty negative prompt. All50 steps checked for dog conditioning and terminal sigma0.

This paired comparison holds the Gaussian realization fixed. The full-start run also includes four additional denoising updates, so output differences cannot be attributed solely to the retained cat latent.
