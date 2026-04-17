# 5.3.3 Deep Reinforcement Learning

DQN, REINFORCE, Actor-Critic, PPO, replay buffers, target networks.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Policy gradient (REINFORCE) from scratch |
| `working_example2.py` | DQN components: replay buffer, target network, linear Q |
| `working_example.ipynb` | Interactive: replay buffer + discounted returns |

## Quick Reference

```python
# DQN core loop
for s, a, r, ns, done in batch:
    target = r if done else r + gamma * target_net(ns).max()
    loss = F.mse_loss(online_net(s)[a], target)

# Target network update (hard)
if step % C == 0:
    target_net.load_state_dict(online_net.state_dict())

# REINFORCE loss
returns = compute_returns(rewards, gamma)
loss = -sum(log_prob[t] * returns[t] for t in range(T))

# PPO clip objective
ratio = pi_new(a|s) / pi_old(a|s)
L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```

## Algorithm Comparison

| Algorithm | Policy | Value | Replay | Notes |
|-----------|--------|-------|--------|-------|
| DQN | ε-greedy | Q-network | ✓ | Discrete actions |
| REINFORCE | Stochastic | None | ✗ | High variance |
| A2C | Stochastic | V-net | ✗ | Synchronous |
| PPO | Stochastic | V-net | ✗ | Clipped, stable |
| SAC | Stochastic | Q-net | ✓ | Continuous, entropy |

## Learning Resources
- [Spinning Up by OpenAI](https://spinningup.openai.com/)
- [CleanRL implementations](https://github.com/vwxyzjn/cleanrl)

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
