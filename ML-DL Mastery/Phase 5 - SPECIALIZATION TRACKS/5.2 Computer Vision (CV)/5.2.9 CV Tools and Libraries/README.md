# 5.2.9 CV Tools and Libraries

OpenCV, Pillow, torchvision, scikit-image, Albumentations — ecosystem guide.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | OpenCV pipeline (BGRA, Canny, resize) |
| `working_example2.py` | Library detection + numpy image operations demo |
| `working_example.ipynb` | Interactive: library check → digits gallery |

## Quick Reference

```python
# OpenCV
import cv2
img = cv2.imread("img.jpg")                    # BGR uint8
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(img, (224, 224))
edges = cv2.Canny(gray, 50, 150)

# Pillow
from PIL import Image
img = Image.open("img.jpg").convert("RGB")
img = img.resize((224, 224), Image.LANCZOS)
arr = np.array(img)                            # (H, W, 3) uint8

# torchvision
from torchvision import transforms
t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
```

## Library Comparison

| Library | Format | Speed | GPU | Best use |
|---------|--------|-------|-----|---------|
| OpenCV | BGR uint8 | Fast | No | Classical CV |
| Pillow | RGB uint8 | Medium | No | Preprocessing |
| torchvision | Tensor | Fast | Yes | DL training |
| scikit-image | float64 | Medium | No | Scientific |
| Albumentations | numpy | Fast | Partial | Augmentation |

## Learning Resources
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [torchvision docs](https://pytorch.org/vision/stable/)

Process images and build CV examples.

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
