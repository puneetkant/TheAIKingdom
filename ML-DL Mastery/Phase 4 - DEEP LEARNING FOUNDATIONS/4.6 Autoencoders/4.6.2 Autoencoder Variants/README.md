# 4.6.2 Autoencoder Variants

Denoising AE, Sparse AE (L1 latent), Variational AE (reparametrisation trick + KL divergence).

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Sparse AE with L1 penalty on latent activations |
| `working_example2.py` | Denoising AE vs Standard AE — noisy reconstruction comparison |
| `working_example.ipynb` | Interactive: denoising AE setup → VAE reparametrisation |

## Quick Reference

```python
# Denoising AE: add noise to input, keep clean target
x_noisy = x + torch.randn_like(x) * 0.5
x_hat   = model(x_noisy)
loss    = F.mse_loss(x_hat, x)  # target = clean x

# VAE: reparametrisation trick
mu, log_var = encoder(x)
eps = torch.randn_like(mu)
z   = mu + eps * (log_var * 0.5).exp()
x_hat = decoder(z)
recon = F.mse_loss(x_hat, x)
kl    = -0.5 * (1 + log_var - mu**2 - log_var.exp()).mean()
loss  = recon + beta * kl
```

## Variant Comparison

| Variant | Key Difference | Added Constraint | Use |
|---------|---------------|-----------------|-----|
| Basic AE | Clean→clean | None | Compression |
| Denoising AE | Noisy→clean | Robustness | Denoising |
| Sparse AE | L1 on z | Sparsity | Feature learning |
| VAE | z ~ N(μ,σ) | KL divergence | Generation |

## Learning Resources
- [VAE paper (Kingma 2013)](https://arxiv.org/abs/1312.6114)
- [Denoising AE (Vincent 2010)](https://dl.acm.org/doi/10.1145/1390156.1390294)

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
