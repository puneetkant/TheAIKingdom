# 4.6.3 Autoencoder Applications

Anomaly detection (reconstruction error), dimensionality reduction, image denoising.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Latent space visualisation (2D projection) |
| `working_example2.py` | Anomaly detection: train on normal, score anomalies → ROC-AUC |
| `working_example.ipynb` | Interactive: train on normal → reconstruction error histogram → AUC |

## Quick Reference

```python
# Anomaly detection with AE
model.train()
optimizer.zero_grad()
x_hat, _ = model(x_normal)
loss = F.mse_loss(x_hat, x_normal)
loss.backward(); optimizer.step()

# Scoring (at inference)
model.eval()
with torch.no_grad():
    x_hat, _ = model(x_test)
    recon_error = ((x_hat - x_test)**2).mean(dim=1)
    is_anomaly = recon_error > threshold
```

## Application Map

| Application | Signal | How |
|-------------|--------|-----|
| Anomaly detection | High reconstruction error | Train on normal only |
| Dimensionality reduction | Latent z | Encoder output |
| Denoising | Clean output | DAE training |
| Data generation | Sample z ~ N(0,1) | VAE decoder |

## Learning Resources
- [AE for anomaly detection](https://arxiv.org/abs/2101.03154)
- [PyTorch autoencoder tutorial](https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/)

Explore this topic with a small practical project or coding exercise.

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
