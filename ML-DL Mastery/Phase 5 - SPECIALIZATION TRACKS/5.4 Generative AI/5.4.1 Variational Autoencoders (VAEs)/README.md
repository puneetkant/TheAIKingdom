# 5.4.1 Variational Autoencoders (VAEs)

ELBO objective, reparametrisation trick, encoder/decoder, KL divergence, latent interpolation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Standard AE vs VAE reconstruction |
| `working_example2.py` | Linear VAE on digits → latent interpolation |
| `working_example.ipynb` | Interactive: reparametrisation + KL contour |

## Quick Reference

```python
# Encoder output
mu, logvar = encoder(x)  # two head outputs

# Reparametrisation trick (differentiable sampling)
eps = torch.randn_like(logvar)
z = mu + eps * torch.exp(0.5 * logvar)

# ELBO loss = reconstruction + KL
recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
loss = recon_loss + beta * kl_loss  # beta-VAE when beta != 1

# Generation (sample from prior)
z = torch.randn(n, latent_dim)
x_gen = decoder(z)
```

## Key Concepts

| Term | Meaning |
|------|---------|
| ELBO | Evidence Lower BOund = -loss |
| KL | Regularises latent to N(0,I) |
| β-VAE | Higher β → disentangled latents |
| VQ-VAE | Discrete codebook instead of Gaussian |

## Learning Resources
- [Original VAE paper](https://arxiv.org/abs/1312.6114)
- [Understanding VAEs (blog)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

Build an autoencoder-style encoder/decoder flow.

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
