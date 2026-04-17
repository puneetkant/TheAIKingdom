# 5.3.1 RL Fundamentals

MDPs, Bellman equations, policy evaluation, value iteration, Q-values.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Random policy rollout on GridWorld |
| `working_example2.py` | Value iteration on 4×4 GridWorld → optimal policy |
| `working_example.ipynb` | Interactive: step-by-step value iteration |

## Quick Reference

```python
# Bellman optimality equation
V[s] = max_a { R(s,a) + gamma * V[s'] }

# Value Iteration (pseudo)
V = np.zeros(n_states)
while True:
    delta = 0
    for s in states:
        v_new = max(R[s,a] + gamma * V[T[s,a]] for a in actions)
        delta = max(delta, abs(v_new - V[s]))
        V[s] = v_new
    if delta < theta: break

# Key concepts
# - Markov Property: P(s' | s0..st, a) = P(s' | st, a)
# - Return Gt = sum_{k=0}^{inf} gamma^k * R_{t+k+1}
# - Policy pi: s -> a (deterministic) or pi(a|s) (stochastic)
```

## Topic Map

| Concept | Equation | Note |
|---------|----------|------|
| Return | $G_t = \sum_{k} \gamma^k R_{t+k}$ | Discounted sum |
| V-function | $V^\pi(s) = E[G_t \mid s_t=s]$ | Policy value |
| Q-function | $Q^\pi(s,a) = E[G_t \mid s_t=s, a_t=a]$ | Action value |
| Advantage | $A(s,a) = Q(s,a) - V(s)$ | Used in A3C |

## Learning Resources
- [Sutton & Barto RL book (free)](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

Implement a basic RL agent or environment.

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
