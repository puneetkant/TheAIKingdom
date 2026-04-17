# 6.4.2 Text-to-Image Generation

Text-to-image models (Stable Diffusion, DALL-E, Imagen, Flux) use diffusion processes to iteratively denoise random noise conditioned on text embeddings. The forward process adds Gaussian noise across T timesteps; the reverse learns to denoise. This folder implements the forward diffusion process, noise schedules, and SNR curves from scratch.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Forward diffusion on 8×8 image, linear vs cosine β schedules, SNR curve, denoised frame grid |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `diffusion_forward.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| DDPM | Denoising Diffusion Probabilistic Models; T-step Markov chain |
| Noise schedule | β_t controls how much noise is added at each step |
| Cosine schedule | Smoother noise schedule; avoids abrupt noising |
| SNR | Signal-to-noise ratio: α_t / (1−α_t); governs difficulty |
| CFG | Classifier-free guidance; sharpens text conditioning |

## Learning Resources

- Ho et al. *DDPM* (2020)
- Nichol & Dhariwal *Improved DDPM* (2021)
- Rombach et al. *Stable Diffusion / LDM* (2022)
