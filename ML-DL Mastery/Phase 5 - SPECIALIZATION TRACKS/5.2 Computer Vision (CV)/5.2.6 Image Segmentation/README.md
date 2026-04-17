# 5.2.6 Image Segmentation

Semantic, instance, panoptic segmentation. FCN, U-Net, Mask RCNN, SAM.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | U-Net architecture overview (PyTorch) |
| `working_example2.py` | K-means pixel clustering → segment boundaries visualisation |
| `working_example.ipynb` | Interactive: colour clustering → segment map |

## Quick Reference

```python
# U-Net encoder-decoder (simplified)
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DoubleConv(256, 128)  # concat with enc2 skip
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)   # concat with enc1 skip
        self.out  = nn.Conv2d(64, out_ch, 1)

# Segmentation loss
criterion = nn.CrossEntropyLoss()  # for semantic
# OR DiceLoss / FocalLoss for imbalanced masks
```

## Segmentation Types

| Type | Output | Model |
|------|--------|-------|
| Semantic | Class per pixel | FCN, U-Net, DeepLab |
| Instance | Unique mask per object | Mask RCNN |
| Panoptic | Semantic + instance | Panoptic FPN |

## Learning Resources
- [U-Net paper](https://arxiv.org/abs/1505.04597)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

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
