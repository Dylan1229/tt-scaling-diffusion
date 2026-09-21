# Three fresh-noise re-noising samples

All three use the exact saved cat latent after40 completed denoising steps, from source seed0. Each adds independently seeded fresh standard Gaussian noise to the nominal noise level after step4, then applies the identical golden retriever prompt for46 restarted UniPC updates. Only the fresh-noise seed changes:20260922,20260923,20260924.

Flow re-noising: x_restart=a*x40+b*z; a=(1-sigma4)/(1-sigma40), b=sqrt(sigma4^2-a^2*sigma40^2). a=0.03852732667954569, b=0.9826901134503093.

Model Wan2.2 TI2V-5B T2V; CFG5;832x480;81frames;24fps; original50-step schedule suffix with solver history reset. Source, prompt embeddings, noise strength and schedule are checked identical. Noise tensors, re-noised latents and videos have distinct hashes. See validation.json and per-seed metadata.

comparison.mp4 synchronizes the three clips. contact_sheet.jpg samples frames0/16/32/48/64/80. Each seed directory contains its video, full81-frame sheet and metadata. Tensor artifacts remain on yukelab.
