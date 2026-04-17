# 2.3.4 Common Probability Distributions

Gallery of ML-relevant distributions: Gaussian, Laplace, Beta, Binomial, Poisson, Student-t, Chi², Dirichlet.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual PMF/PDF computation, sampling |
| `working_example2.py` | 9-panel distribution gallery (saved PNG), moment table |
| `working_example.ipynb` | Interactive: Gaussian → Laplace → Beta → moment table |

## Quick Reference

```python
import numpy as np

np.random.normal(mu, sigma, N)          # Gaussian
np.random.laplace(loc, scale, N)        # Laplace → L1 prior
np.random.exponential(scale, N)         # Exponential
np.random.binomial(n, p, N)             # Binomial
np.random.poisson(lam, N)               # Poisson
np.random.standard_t(df, N)             # Student-t
np.random.beta(a, b, N)                 # Beta (conjugate for p)
np.random.dirichlet([a1, a2, a3], N)    # Dirichlet (topic models)
```

## ML Connections

| Distribution | ML Role |
|-------------|---------|
| Gaussian | Weight init, noise model, Gaussian process |
| Laplace | L1 / Lasso prior (sparse weights) |
| Bernoulli/Binomial | Binary classification, Naive Bayes |
| Beta | Bayesian prior for probabilities |
| Dirichlet | LDA topic models, Bayesian NN priors |
| Student-t | Robust regression, Bayesian inference |

## Learning Resources
- [Seeing Theory: Distributions](https://seeing-theory.brown.edu/probability-distributions/)
- [StatQuest: Beta Distribution](https://youtu.be/v1uUgTcInQk)

Compute probabilities, expectations, and distributions.

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
