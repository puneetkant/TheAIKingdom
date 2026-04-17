# 5.2.5 Object Detection

Anchor boxes, IoU, NMS, YOLO/RCNN architectures. Single-stage vs two-stage detectors.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | YOLO inference demo with torchvision |
| `working_example2.py` | IoU computation + NMS implementation + before/after visualisation |
| `working_example.ipynb` | Interactive: IoU calculation → NMS filtering |

## Quick Reference

```python
import torch
from torchvision.ops import nms, box_iou

boxes  = torch.tensor([[10, 10, 60, 60], [12, 12, 62, 62]], dtype=torch.float)
scores = torch.tensor([0.95, 0.85])
keep   = nms(boxes, scores, iou_threshold=0.5)

# IoU matrix
iou_matrix = box_iou(boxes, boxes)  # (N, N)

# YOLOv5 inference (ultralytics)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
results = model("image.jpg")
results.pandas().xyxy[0]  # DataFrame of detections
```

## Detector Comparison

| Detector | Stage | Speed | Accuracy |
|----------|-------|-------|----------|
| Faster RCNN | Two-stage | Slow | High |
| SSD | One-stage | Medium | Medium |
| YOLO v5/v8 | One-stage | Fast | High |
| DETR | One-stage | Medium | High |

## Learning Resources
- [YOLO paper series](https://arxiv.org/abs/1506.02640)
- [Towards Data Science: RCNN family](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)

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
