# 5.2.4 Image Classification

CNNs for image classification: ResNet, EfficientNet, ViT. Fine-tuning pre-trained models.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | torchvision ResNet18 fine-tuning pattern |
| `working_example2.py` | PCA features vs raw pixels → SVM → confusion matrix |
| `working_example.ipynb` | Interactive: digits SVM → PCA+SVM accuracy comparison |

## Quick Reference

```python
from torchvision import models, transforms

# Load pretrained ResNet18
model = models.resnet18(weights="DEFAULT")
# Replace final layer for new task
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze backbone, train head only
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
```

## Model Comparison

| Model | Top-1 ImageNet | Params | Notes |
|-------|---------------|--------|-------|
| ResNet-50 | 76.1% | 25M | Residual connections |
| EfficientNet-B4 | 83.0% | 19M | Compound scaling |
| ViT-B/16 | 81.8% | 86M | Attention-only |
| ConvNeXt-T | 82.1% | 29M | Modern CNN |

## Learning Resources
- [torchvision models](https://pytorch.org/vision/stable/models.html)
- [ImageNet leaderboard](https://paperswithcode.com/sota/image-classification-on-imagenet)

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
