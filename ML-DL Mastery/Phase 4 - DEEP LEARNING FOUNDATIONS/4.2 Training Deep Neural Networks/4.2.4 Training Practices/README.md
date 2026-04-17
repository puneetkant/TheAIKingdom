# 4.2.4 Training Practices

Mini-batch SGD, gradient clipping, early stopping, batch size effects.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Gradient norm monitoring |
| `working_example2.py` | Compare full-batch vs mini-batch vs clipping vs early stopping |
| `working_example.ipynb` | Interactive: mini-batch + gradient clipping + early stopping |

## Quick Reference

```python
# Gradient clipping (PyTorch)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Early stopping (manual)
best_val, patience, wait = np.inf, 10, 0
if val_loss < best_val:
    best_val = val_loss; wait = 0; torch.save(model.state_dict(), 'best.pt')
else:
    wait += 1
    if wait >= patience: break

# Mini-batch
from torch.utils.data import DataLoader, TensorDataset
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
```

## Practical Tips

| Technique | Why |
|-----------|-----|
| Gradient clipping | Prevents exploding gradients (RNNs) |
| Mini-batches | Noise regularises + GPU efficiency |
| Early stopping | Automatic regularisation |
| Checkpoint best | Recover optimal model at stop |

## Learning Resources
- [Gradient clipping (Pascanu 2013)](https://arxiv.org/abs/1211.5063)
- [PyTorch training loop guide](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)

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
