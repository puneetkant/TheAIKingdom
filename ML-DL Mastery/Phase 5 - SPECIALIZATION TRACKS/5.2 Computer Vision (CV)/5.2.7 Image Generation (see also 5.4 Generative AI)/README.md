# 5.2.7 Image Generation

VAE, GAN, Diffusion Models for image synthesis. See also 5.4 Generative AI for deep dives.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | GAN training loop concept |
| `working_example2.py` | VAE latent sampling → decode → sample grid on digits |
| `working_example.ipynb` | Interactive: VAE reparametrisation → GAN loss |

## Quick Reference

```python
# VAE loss = Reconstruction + KL divergence
recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
kl_loss    = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum()
loss = recon_loss + beta * kl_loss

# DCGAN Generator step
noise = torch.randn(batch_size, nz, 1, 1)
fake  = generator(noise)
g_loss = criterion(discriminator(fake), real_labels)  # fool D

# Diffusion: add noise schedule
alphas = torch.linspace(0.9999, 0.98, T)
x_noisy = sqrt(alpha) * x + sqrt(1-alpha) * noise
```

## Generative Model Comparison

| Model | Training | Inference | Quality |
|-------|---------|-----------|---------|
| VAE | Stable | Fast | Blurry |
| GAN | Unstable | Fast | Sharp |
| Diffusion | Stable | Slow | SOTA |
| Flow | Stable | Medium | Good |

## Learning Resources
- [GANs explained (Goodfellow)](https://arxiv.org/abs/1406.2661)
- [DDPM paper](https://arxiv.org/abs/2006.11239)

Create or explore generative model results.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
