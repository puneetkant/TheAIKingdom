# 4.6.1 Basic Autoencoders

Encoder-decoder MLP, bottleneck representation, reconstruction loss (MSE).

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Autoencoder on toy 2D data — visualise latent space |
| `working_example2.py` | Cal Housing 8→3→8: train, test MSE, reconstruction scatter |
| `working_example.ipynb` | Interactive: train → loss curve → reconstruction plot |

## Quick Reference

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_in, n_z):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in,64), nn.ReLU(), nn.Linear(64, n_z))
        self.decoder = nn.Sequential(nn.Linear(n_z,64), nn.ReLU(), nn.Linear(64, n_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

loss = nn.MSELoss()(x_hat, x)   # reconstruction loss
```

## Autoencoder Applications

| Application | Bottleneck | Loss |
|-------------|-----------|------|
| Dimensionality reduction | n_z << n_in | MSE |
| Denoising | Same as input | MSE on clean |
| Anomaly detection | Large reconstruction error = anomaly | MSE |
| Generative (VAE) | μ, σ → reparametrize | MSE + KL |

## Learning Resources
- [Autoencoder tutorial (PyTorch)](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Deep Learning book Ch.14](https://www.deeplearningbook.org/contents/autoencoders.html)

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
