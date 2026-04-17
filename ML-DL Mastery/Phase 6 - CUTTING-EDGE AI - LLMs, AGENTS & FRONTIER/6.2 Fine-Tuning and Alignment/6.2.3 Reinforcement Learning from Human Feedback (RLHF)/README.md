# 6.2.3 Reinforcement Learning from Human Feedback (RLHF)

RLHF is the three-stage pipeline that aligns LLMs with human preferences: (1) supervised fine-tuning, (2) reward model training from pairwise comparisons, and (3) PPO optimisation against the reward model with a KL penalty to prevent reward hacking. This folder simulates reward model training, PPO reward curves, and KL-divergence tracking.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Reward model training simulation, PPO reward accumulation, KL penalty sweep |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `rlhf_demo.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Reward model | Scores responses; trained from human pairwise preferences |
| PPO | Proximal Policy Optimisation; clips policy update ratio |
| KL penalty | Prevents policy deviating too far from reference model |
| Bradley-Terry model | Pairwise preference → reward probability |
| Reward hacking | Policy exploits flaws in learned reward model |

## Learning Resources

- Ouyang et al. *InstructGPT* (2022)
- Stiennon et al. *Learning to summarise* (2020)
- Schulman et al. *PPO* (2017)
