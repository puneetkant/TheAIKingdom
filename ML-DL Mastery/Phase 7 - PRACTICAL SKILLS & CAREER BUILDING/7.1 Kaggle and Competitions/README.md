# 7.1 Kaggle and Competitions

Kaggle competitions are the fastest way to build practical ML intuition — you get real datasets, a leaderboard, and a community of top practitioners sharing notebooks. Key skills include cross-validation strategy, feature engineering, ensembling (stacking, blending), and avoiding public leaderboard overfitting. This folder implements K-fold CV, ensemble stacking, and a leaderboard shake-up simulation.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | 5-fold CV, meta-learner stacking, public vs private LB shake-up scatter |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `kaggle_competition.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Cross-validation | Estimate generalisation error; stratified K-fold |
| Ensemble stacking | Train meta-learner on base model OOF predictions |
| LB shake-up | Public/private split → public rank ≠ private rank |
| Feature engineering | Domain-specific transformations; biggest impact |
| Target encoding | Replace category with target mean (CV-safe) |

## Learning Resources

- Kaggle Learn courses
- Kaggle Grandmaster notebooks
- *Feature Engineering for Machine Learning* (Zheng & Casari)
