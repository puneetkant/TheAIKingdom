# 2.3.5 Joint Distributions and Transformations

Bivariate Gaussian, marginalisation, covariance/correlation, Box-Muller transform.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | 2D joint PDF by hand, conditional distribution |
| `working_example2.py` | Bivariate Gaussian via Cholesky, Box-Muller, Cal Housing correlation heatmap |
| `working_example.ipynb` | Interactive: bivariate scatter → marginals → Box-Muller → correlation |

## Quick Reference

```python
import numpy as np

# Bivariate Gaussian via Cholesky
L = np.linalg.cholesky(sigma)     # sigma = covariance matrix (PD)
X = np.random.randn(N, 2) @ L.T + mu

# Marginalise: just take column
X1 = X[:, 0]   # ~ N(mu[0], sigma[0,0])

# Box-Muller: Uniform -> Gaussian
Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2)

# Correlation
np.corrcoef(X.T)        # shape (d, d)
np.cov(X.T)             # covariance matrix
```

## ML Connections
- **Multivariate Gaussian** — Gaussian process, VAE latent space
- **Correlation matrix** — feature selection, PCA preprocessing
- **Change of variables** — normalising flows, reparameterisation trick

## Learning Resources
- [Seeing Theory: Joint Distributions](https://seeing-theory.brown.edu/probability-distributions/index.html#section2)
- **Book:** *Pattern Recognition and ML* (Bishop) Ch. 2.3

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
