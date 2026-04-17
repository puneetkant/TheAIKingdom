# 7.7 Ethics and Responsible AI

Responsible AI requires proactively auditing systems for fairness, transparency, and safety before deployment. Key fairness metrics include Demographic Parity, Equalized Odds, Calibration, and Disparate Impact Ratio. Mitigations include threshold equalisation, reweighting, and adversarial debiasing. This folder implements a loan approval fairness audit with all four metrics and a model card template.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Demographic Parity, Equalized Odds, calibration curves, Disparate Impact, model card |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `responsible_ai.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Demographic Parity | Equal positive prediction rates across groups |
| Equalized Odds | Equal TPR and FPR across groups |
| Disparate Impact | Ratio of positive rates; threshold ≥ 0.8 (4/5 rule) |
| Calibration | Model confidence matches empirical frequencies |
| Model Card | Structured documentation of model purpose, limits, risks |

## Learning Resources

- Barocas, Hardt & Narayanan *Fairness and Machine Learning* (fairmlbook.org)
- Mitchell et al. *Model Cards* (2019)
- Mehrabi et al. *A Survey on Bias and Fairness in ML* (2021)
