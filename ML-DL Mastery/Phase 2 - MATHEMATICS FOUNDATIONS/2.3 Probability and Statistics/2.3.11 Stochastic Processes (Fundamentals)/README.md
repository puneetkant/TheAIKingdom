# 2.3.11 Stochastic Processes (Fundamentals)

Markov chains, stationary distribution (eigenvector method), random walk ±√t envelope, Brownian motion, Geometric BM.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Markov chain simulation, convergence to stationary |
| `working_example2.py` | Full Markov chain + stationary, random walk plot, Brownian + Geometric BM |
| `working_example.ipynb` | Interactive: Markov → random walk → Brownian → Geometric BM |

## Quick Reference

```python
import numpy as np

# Stationary distribution (left eigenvector for λ=1)
vals, vecs = np.linalg.eig(P.T)
stat = np.real(vecs[:, np.argmin(np.abs(vals-1))])
stat /= stat.sum()

# Verify: P^n converges
Pk = np.linalg.matrix_power(P, 100)  # rows ≈ stat

# Brownian motion (Wiener process)
dW = np.random.normal(0, dt**0.5, n)  # W(t+dt) - W(t) ~ N(0, dt)
W  = np.cumsum(dW)

# Geometric BM: S(t) = S0 * exp((mu - 0.5*sig²)*t + sig*W)
```

## ML Connections
- **Hidden Markov Models (HMMs)** — speech recognition, sequence labelling
- **Markov Decision Processes (MDPs)** — foundation of reinforcement learning
- **MCMC** — Bayesian sampling via Markov chains (Metropolis-Hastings, NUTS)
- **Diffusion models** — forward noising is discrete Brownian; reverse is learned

## Learning Resources
- [StatQuest: Markov Chains](https://youtu.be/i3AkTO9HLXo)
- **Book:** *Introduction to Probability Models* (Sheldon Ross)
- [3Blue1Brown: Random walks](https://youtu.be/OkmNXy7er84)

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
