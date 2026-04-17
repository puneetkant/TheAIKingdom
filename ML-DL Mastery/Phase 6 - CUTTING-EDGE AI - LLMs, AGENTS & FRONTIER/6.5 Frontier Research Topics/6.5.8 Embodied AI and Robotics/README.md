# 6.5.8 Embodied AI and Robotics

Embodied AI agents act in physical or simulated environments — perceiving sensory inputs, planning actions, and learning from interaction. Policy learning methods (REINFORCE, PPO, SAC) train agents on reward signals from the environment. This folder implements a 6×6 GridWorld environment with a REINFORCE policy gradient agent that learns to navigate to a goal.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | GridWorld 6×6, REINFORCE policy gradient 200 episodes, learned policy arrow grid |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `embodied_ai.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| REINFORCE | Policy gradient: ∇J = E[∇logπ(a|s)·R] |
| Actor-critic | Baseline V(s) reduces variance in policy gradient |
| Sim-to-real | Transfer policy from simulation to real hardware |
| RT-2 | Vision-language-action model for robot control |
| Reward shaping | Dense rewards guide learning in sparse environments |

## Learning Resources

- Sutton & Barto *Reinforcement Learning* (2018)
- Brohan et al. *RT-2* (2023)
- OpenAI *Gym* documentation
