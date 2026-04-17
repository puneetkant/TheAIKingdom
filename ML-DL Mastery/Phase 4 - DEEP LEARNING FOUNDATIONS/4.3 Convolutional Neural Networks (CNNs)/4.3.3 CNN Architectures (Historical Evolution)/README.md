# 4.3.3 CNN Architectures (Historical Evolution)

LeNet → AlexNet → VGG → GoogLeNet → ResNet — architectural innovations timeline.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Inception module concept demo |
| `working_example2.py` | Architecture table + digits flatten vs pooled proxy |
| `working_example.ipynb` | Interactive: arch timeline → residual block forward demo |

## Architecture Timeline

| Year | Model | Key Innovation | Params |
|------|-------|---------------|--------|
| 1998 | LeNet-5 | Conv+Pool stacking | ~60K |
| 2012 | AlexNet | Deep ReLU + Dropout | ~60M |
| 2014 | VGG-16 | Depth with 3×3 | ~138M |
| 2014 | GoogLeNet | Inception modules | ~7M |
| 2015 | ResNet-50 | Skip connections | ~25M |
| 2017 | DenseNet | Dense connections | ~8M |

## Residual Block

```python
# Skip connection (identity shortcut)
def residual_block(x):
    h = relu(conv(x))
    h = conv(h)
    return relu(h + x)   # ← skip
```

## Learning Resources
- [ResNet paper (He 2015)](https://arxiv.org/abs/1512.03385)
- [PyTorch torchvision models](https://pytorch.org/vision/stable/models.html)

Explore convolutional operations for image data.

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
