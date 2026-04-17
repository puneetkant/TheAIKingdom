# 6.5.5 AI Safety and Alignment

AI safety research ensures advanced AI systems remain beneficial, honest, and corrigible as capabilities scale. Key concerns include reward hacking (Goodhart’s Law), deceptive alignment, power-seeking, and sycophancy. Mitigations include RLHF, constitutional AI, debate, and interpretability-based oversight. This folder simulates reward hacking, KL divergence tracking, and constitutional filtering.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Reward hacking simulation, KL divergence tracking, constitutional AI filtering, β penalty sweep |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `ai_safety.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Goodhart’s Law | When a measure becomes a target, it ceases to be a good measure |
| KL regularisation | Penalise deviation from reference policy during RLHF |
| Constitutional AI | Self-critique using a set of principles |
| Corrigibility | Model accepts human correction and shutdown |
| Scalable oversight | Use AI assistance to evaluate AI outputs |

## Learning Resources

- Bai et al. *Constitutional AI* (2022)
- Hadfield-Menell et al. *The Off-Switch Game* (2016)
- Anthropic alignment research (2023–2024)
