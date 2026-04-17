# 4.3.4 Transfer Learning

Feature extraction (frozen backbone) vs fine-tuning — leverage pretrained CNN weights.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Load pretrained torchvision model, extract features |
| `working_example2.py` | PCA as backbone proxy: feature extraction vs fine-tuning, few-shot |
| `working_example.ipynb` | Interactive: feature extraction vs fine-tune + few-shot comparison |

## Quick Reference

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet-50
model = models.resnet50(weights='DEFAULT')

# Feature extraction: freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier head
model.fc = nn.Linear(2048, num_classes)

# Fine-tuning: unfreeze later layers
for param in model.layer4.parameters():
    param.requires_grad = True
```

## Transfer Learning Strategies

| Strategy | Frozen? | Data needed | Speed |
|---------|---------|-------------|-------|
| Feature extraction | All backbone | Very little | Fastest |
| Fine-tune top layers | Lower backbone | Moderate | Medium |
| Full fine-tune | Nothing | Large | Slowest |

## Learning Resources
- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [HuggingFace fine-tuning guide](https://huggingface.co/docs/transformers/training)

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
