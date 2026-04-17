# 2.3.7 Estimation Theory

MLE (Gaussian, Bernoulli), MAP with Beta conjugate prior, bias-variance decomposition.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Moment estimators, consistency proof by simulation |
| `working_example2.py` | MLE Gaussian (log-likelihood contour), Bernoulli MLE+CI, MAP, bias-variance table |
| `working_example.ipynb` | Interactive: MLE Gaussian → Bernoulli CI → MAP → bias-variance |

## Quick Reference

```python
import numpy as np

# MLE Gaussian
mu_hat    = data.mean()
sigma_hat = np.sqrt(np.mean((data - mu_hat)**2))  # biased MLE
sigma_unb = np.std(data, ddof=1)                  # unbiased

# MLE Bernoulli
p_hat = data.mean()
se    = np.sqrt(p_hat * (1 - p_hat) / len(data))

# MAP: conjugate Beta(alpha,beta) prior
p_map = (alpha + k - 1) / (alpha + beta + n - 2)
# Posterior mean
p_post = (alpha + k) / (alpha + beta + n)
```

## Bias-Variance Trade-off
$$\text{MSE} = \text{Bias}^2 + \text{Variance}$$

- Low complexity → high bias, low variance (underfitting)
- High complexity → low bias, high variance (overfitting)

## ML Connections
- **Training loss minimisation** = approximate MLE
- **L2 regularisation** = MAP with Gaussian prior
- **Cross-validation** selects the bias-variance sweet spot

## Learning Resources
- [StatQuest: MLE](https://youtu.be/XepXtl9YKwc)
- **Book:** *All of Statistics* (Wasserman) Ch. 9

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
