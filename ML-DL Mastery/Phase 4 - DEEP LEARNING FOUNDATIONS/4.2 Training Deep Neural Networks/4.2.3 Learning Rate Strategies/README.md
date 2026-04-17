# 4.2.3 Learning Rate Strategies

Step decay, cosine annealing, linear warmup, ReduceLROnPlateau, cyclic LR.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | LR finder range test |
| `working_example2.py` | Schedule visualisation + effect on training loss |
| `working_example.ipynb` | Interactive: LR curve plots → compare Constant vs Warmup+Cos |

## Quick Reference

```python
import torch.optim.lr_scheduler as sched

scheduler = sched.CosineAnnealingLR(optimizer, T_max=200)
scheduler = sched.StepLR(optimizer, step_size=50, gamma=0.5)
scheduler = sched.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
scheduler = sched.OneCycleLR(optimizer, max_lr=0.1, epochs=100, steps_per_epoch=len(loader))

# call per epoch
scheduler.step()
```

## Schedule Comparison

| Schedule | When to Use |
|----------|-------------|
| Constant | Quick experiments |
| Step Decay | Classic training recipes |
| Cosine Annealing | Most DL tasks |
| Linear Warmup + Cosine | Transformers, fine-tuning |
| ReduceLROnPlateau | Unknown optimal LR |
| 1-Cycle | Fast convergence (Super-Convergence) |

## Learning Resources
- [Cyclical LR paper (Smith 2017)](https://arxiv.org/abs/1506.01186)
- [PyTorch schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

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
