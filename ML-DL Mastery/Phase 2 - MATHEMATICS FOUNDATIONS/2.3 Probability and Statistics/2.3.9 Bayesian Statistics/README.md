# 2.3.9 Bayesian Statistics

Beta-Binomial conjugate prior, sequential Bayesian update, Metropolis-Hastings MCMC.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Bayes rule applied, Gaussian-Gaussian conjugate |
| `working_example2.py` | Beta-Binomial update plot, sequential update, Metropolis-Hastings MCMC |
| `working_example.ipynb` | Interactive: conjugate update → sequential → MCMC trace+histogram |

## Quick Reference

```python
# Beta-Binomial conjugate model
# Prior: p ~ Beta(alpha, beta)  ->  Posterior: p|data ~ Beta(alpha+k, beta+n-k)
alpha_post = alpha + k
beta_post  = beta + n - k
posterior_mean = alpha_post / (alpha_post + beta_post)

# Gaussian-Gaussian conjugate
# Prior: mu ~ N(mu_0, sigma_0²)  ->  Posterior mu|data ~ N(mu_post, sigma_post²)
mu_post = (ybar/sig_lik**2 + mu0/sig0**2) / (n/sig_lik**2 + 1/sig0**2)

# Metropolis-Hastings step
proposal = current + N(0, prop_std)
log_ratio = log_posterior(proposal) - log_posterior(current)
if log(uniform) < log_ratio: accept
```

## ML Connections
- **Bayesian neural networks** — weight posterior instead of point estimate
- **Gaussian processes** — non-parametric Bayesian regression
- **Variational inference (VI/ELBO)** — scalable Bayesian approximation
- **MAP regularisation** — Bayesian interpretation of L2/L1

## Learning Resources
- [StatQuest: Bayesian Statistics](https://youtu.be/BrK7X_XlGB8)
- **Book:** *Bayesian Data Analysis* (Gelman et al.)
- [PyMC tutorials](https://www.pymc.io/projects/examples/)

Analyze sample data and calculate summary statistics.

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
