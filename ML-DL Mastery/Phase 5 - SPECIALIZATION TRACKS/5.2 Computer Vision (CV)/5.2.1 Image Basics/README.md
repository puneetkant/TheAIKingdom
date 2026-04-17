# 5.2.1 Image Basics

Pixels, channels, color spaces, histograms, brightness/contrast adjustments.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | RGB → HSV conversion, channel separation |
| `working_example2.py` | Synthetic image → grayscale → brightness/contrast → histogram |
| `working_example.ipynb` | Interactive: RGB image → grayscale visualisation |

## Quick Reference

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and inspect
img = np.array(Image.open("photo.jpg"))   # (H, W, 3) uint8
print(img.shape, img.dtype, img.min(), img.max())

# Grayscale (luminance)
gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

# Normalize to [0,1]
img_f = img.astype(np.float32) / 255.0

# Histogram
plt.hist(gray.ravel(), bins=256, range=(0,255))
```

## Key Concepts

| Concept | Details |
|---------|---------|
| Pixel | Discrete intensity value, 0–255 (uint8) |
| Channel | R, G, B (or H, S, V in HSV) |
| Resolution | Height × Width (H, W) |
| Aspect ratio | W / H |
| Bit depth | 8-bit = 256 levels per channel |

## Learning Resources
- [PIL/Pillow docs](https://pillow.readthedocs.io/)
- [CS231n image notes](https://cs231n.github.io/)

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
