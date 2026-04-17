# 4.3.2 Pooling Layers

MaxPool, AvgPool, Global Average Pooling — spatial downsampling and translation invariance.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Pooling vs no pooling on feature maps |
| `working_example2.py` | Manual maxpool/avgpool/global + visualisation |
| `working_example.ipynb` | Interactive: implement pooling → compare shapes |

## Quick Reference

```python
import torch.nn as nn

nn.MaxPool2d(kernel_size=2, stride=2)        # halves H,W
nn.AvgPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))                 # global avg pool → (B,C,1,1)
nn.Flatten()                                  # → (B, C) for classifier
```

## Pooling Properties

| Type | Invariant To | Use |
|------|-------------|-----|
| MaxPool | Small translations | Standard CNNs |
| AvgPool | Smooth spatial summary | GoogLeNet, lightweight |
| Global AvgPool | Any spatial size input | Replace FC layers |

## Learning Resources
- [CS231n Pooling](https://cs231n.github.io/convolutional-networks/#pool)
- [PyTorch MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

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
