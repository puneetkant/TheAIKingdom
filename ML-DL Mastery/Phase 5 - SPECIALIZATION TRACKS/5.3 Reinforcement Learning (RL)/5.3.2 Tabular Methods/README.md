# 5.3.2 Tabular Methods

Q-Learning, SARSA, Monte Carlo methods, epsilon-greedy exploration.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Monte Carlo policy evaluation |
| `working_example2.py` | Q-Learning vs SARSA comparison on 5×5 GridWorld |
| `working_example.ipynb` | Interactive: Q-Learning return curve |

## Quick Reference

```python
# Q-Learning (off-policy)
Q[s, a] += alpha * (r + gamma * Q[s_next].max() - Q[s, a])

# SARSA (on-policy)
a_next = eps_greedy(Q, s_next, eps)
Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])

# Epsilon-greedy
def eps_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    return Q[s].argmax()
```

## Method Comparison

| Method | On/Off-policy | Bootstrapping | Convergence |
|--------|--------------|---------------|-------------|
| MC | On | No | Slow, low bias |
| TD(0) | Both | Yes | Medium |
| Q-Learning | Off | Yes | Fast, overestimate |
| SARSA | On | Yes | Safer paths |
| Double-Q | Off | Yes | Less overestimate |

## Learning Resources
- [Sutton & Barto Ch 5-6](http://incompleteideas.net/book/the-book-2nd.html)

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
