# 5.3.4 RL Environments and Tools

Gymnasium, Atari, MuJoCo, custom envs, vectorized envs, Stable-Baselines3.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Custom Gym-like env stub |
| `working_example2.py` | k-Armed Bandit env + UCB vs ε-greedy comparison |
| `working_example.ipynb` | Interactive: library check + custom MDP |

## Quick Reference

```python
# Gymnasium (new gym)
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated: break
env.close()

# Custom environment
class MyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, (4,))
        self.action_space = gym.spaces.Discrete(2)
    def reset(self, seed=None): return np.zeros(4), {}
    def step(self, action): return np.zeros(4), 0, False, False, {}

# Stable-Baselines3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
```

## Environment Catalogue

| Env | Library | Type | Actions |
|-----|---------|------|---------|
| CartPole-v1 | Gymnasium | Classic | Discrete |
| LunarLander-v2 | Gymnasium | Classic | Discrete |
| Ant-v4 | MuJoCo | Continuous | Continuous |
| ALE/Pong | Arcade | Atari | Discrete |

## Learning Resources
- [Gymnasium docs](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

Check your Python environment and install packages.

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
