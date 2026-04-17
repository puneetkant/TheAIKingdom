# 5.2.2 Classical Computer Vision

Sobel/Canny edge detection, Harris corners, HOG features, SIFT keypoints, template matching.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | OpenCV Canny + Harris corner detection |
| `working_example2.py` | Sobel edges + HOG features from scratch → SVM accuracy vs raw pixels |
| `working_example.ipynb` | Interactive: Sobel → edge visualisation |

## Quick Reference

```python
import cv2
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Edge detection
edges = cv2.Canny(img, threshold1=50, threshold2=150)

# Harris corners
corners = cv2.cornerHarris(img.astype(float32), blockSize=2, ksize=3, k=0.04)
img[corners > 0.01 * corners.max()] = 255  # mark corners

# HOG features
hog = cv2.HOGDescriptor()
h = hog.compute(img)  # (N, 1) feature vector

# SIFT
sift = cv2.SIFT_create()
kps, desc = sift.detectAndCompute(img, None)
```

## Classical CV Techniques

| Technique | Purpose | Key parameter |
|-----------|---------|--------------|
| Gaussian blur | Noise reduction | kernel size, σ |
| Sobel/Laplacian | Edge detection | kernel size |
| Canny | Robust edges | low/high threshold |
| Harris | Corner detection | k=0.04, blockSize |
| HOG | Shape descriptor | cells, bins |
| SIFT/ORB | Keypoint matching | nfeatures |

## Learning Resources
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [CS231n: image features](https://cs231n.stanford.edu/)

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
