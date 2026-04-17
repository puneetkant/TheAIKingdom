# 6.2.4 Direct Alignment Methods

Direct Preference Optimisation (DPO) and its variants (IPO, KTO, ORPO) eliminate the need for a separate reward model by re-framing the RLHF objective as a supervised loss directly on (chosen, rejected) pairs. This folder implements the DPO loss derivation, preference margin analysis, and compares convergence with RLHF baselines.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | DPO loss surface, preference margin training curve, comparison with reward model baseline |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `dpo_demo.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| DPO | Closed-form solution that subsumes the reward model |
| Preference margin | log π(chosen)/π(ref) − log π(rejected)/π(ref) |
| IPO | Identity Preference Optimisation; regularised version |
| KTO | Kahneman-Tversky Optimisation; uses prospect theory |
| ORPO | Odds Ratio Preference Optimisation; no reference model |

## Learning Resources

- Rafailov et al. *DPO* (2023)
- Azar et al. *IPO* (2023)
- Ethayarajh et al. *KTO* (2024)
