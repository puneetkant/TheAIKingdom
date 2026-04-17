# 2.3.10 Information Theory

Shannon entropy, KL divergence, cross-entropy loss, mutual information, Jensen-Shannon divergence.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Entropy/KL demo, channel capacity basics |
| `working_example2.py` | Entropy plot, KL asymmetry, cross-entropy loss, mutual information |
| `working_example.ipynb` | Interactive: entropy → KL → CE loss → binary entropy plot |

## Quick Reference

```python
import numpy as np

# Shannon entropy (bits)
H = -np.sum(p * np.log2(p + 1e-15))

# KL divergence (nats) — not symmetric
KL = np.sum(p * np.log(p / q + 1e-15))

# Cross-entropy loss (one-hot labels)
CE = -np.log(y_pred[true_class] + 1e-15)

# Relation: CE(P,Q) = H(P) + KL(P||Q)
# For one-hot labels: H(P)=0, so CE = KL

# Mutual information
MI = H(X) + H(Y) - H(X, Y)
```

## ML Connections
- **Cross-entropy loss** — standard classification training objective
- **KL in VAEs** — regularization term in ELBO = -KL(q||p) + E[log p(x|z)]
- **Mutual information** — feature selection, InfoNCE contrastive loss
- **Entropy** — decision tree splits (information gain = MI), uncertainty quantification

## Learning Resources
- [Shannon's 1948 Paper (Bell System Technical Journal)](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- [StatQuest: Entropy and Information Gain](https://youtu.be/YtebGVx-Fxw)
- **Book:** *Elements of Information Theory* (Cover & Thomas)

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
