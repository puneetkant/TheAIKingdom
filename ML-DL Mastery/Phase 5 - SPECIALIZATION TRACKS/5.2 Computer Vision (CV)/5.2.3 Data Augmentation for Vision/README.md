# 5.2.3 Data Augmentation for Vision

Flips, rotations, colour jitter, cutout/mixup, RandomErasing — augmentation to improve generalisation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | torchvision.transforms pipeline |
| `working_example2.py` | Numpy augmentations: flip/rotate/noise/cutout visualised |
| `working_example.ipynb` | Interactive: augmentation grid on digits |

## Quick Reference

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomErasing(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])
```

## Augmentation Techniques

| Technique | Type | When to use |
|-----------|------|------------|
| Horizontal flip | Geometric | Most vision tasks |
| Random crop | Geometric | Classification |
| Color jitter | Photometric | Medical imaging, etc. |
| Cutout | Occlusion | Object recognition |
| Mixup | Label mixing | Classification |
| MosaicAugment | Multi-image | Detection (YOLO) |

## Learning Resources
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
- [Albumentations library](https://albumentations.ai/)

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
