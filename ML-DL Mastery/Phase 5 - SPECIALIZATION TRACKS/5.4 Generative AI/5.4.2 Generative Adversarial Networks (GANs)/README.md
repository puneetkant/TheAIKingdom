# 5.4.2 Generative Adversarial Networks (GANs)

Min-max game, DCGAN, conditional GANs, WGAN, mode collapse, training stability.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | DCGAN architecture sketch |
| `working_example2.py` | Linear GAN on 1-D Gaussian → training loss curves |
| `working_example.ipynb` | Interactive: GAN losses + D(x) visualisation |

## Quick Reference

```python
# GAN objective (non-saturating)
d_loss = -log(D(x_real)) - log(1 - D(G(z)))
g_loss = -log(D(G(z)))                       # non-saturating G

# WGAN loss (critic not discriminator)
d_loss = -mean(D(x_real)) + mean(D(G(z)))    # + gradient penalty
g_loss = -mean(D(G(z)))

# Conditional GAN (feed label as extra channel)
x_fake = G(z, label)
score  = D(x, label)

# Training loop order
for step in range(n):
    train_discriminator(real_batch, fake_batch)
    train_generator(fake_batch)
```

## GAN Variants

| Model | Key feature | Use case |
|-------|------------|---------|
| DCGAN | Conv layers | Image generation |
| cGAN | Class conditioning | Conditional gen |
| WGAN-GP | Wasserstein loss | Stable training |
| StyleGAN2 | Style injection | HQ face gen |
| CycleGAN | Cycle consistency | Unpaired translation |

## Learning Resources
- [Original GAN paper](https://arxiv.org/abs/1406.2661)
- [GAN Lab (interactive)](https://poloclub.github.io/ganlab/)

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
