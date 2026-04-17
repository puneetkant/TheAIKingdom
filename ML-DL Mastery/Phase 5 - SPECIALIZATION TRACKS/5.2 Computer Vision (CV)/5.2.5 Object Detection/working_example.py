"""
Working Example: Object Detection
Covers bounding boxes, IoU, NMS, anchor boxes, mAP,
and major detection architectures.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Bounding box formats ───────────────────────────────────────────────────
def bounding_box_formats():
    print("=== Bounding Box Formats ===")
    print("  [x1, y1, x2, y2] — xyxy: top-left, bottom-right corners")
    print("  [cx, cy, w, h]   — cxcywh: centre x, centre y, width, height")
    print("  [x1, y1, w, h]   — xywh:   COCO format")
    print()

    # Conversion functions
    def xyxy_to_xywh(b):   return [b[0], b[1], b[2]-b[0], b[3]-b[1]]
    def xywh_to_xyxy(b):   return [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    def xyxy_to_cxcywh(b): return [(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]]
    def cxcywh_to_xyxy(b): return [b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2]

    box_xyxy = [50, 30, 150, 100]
    print(f"  xyxy:    {box_xyxy}")
    print(f"  → xywh:  {xyxy_to_xywh(box_xyxy)}")
    print(f"  → cxcywh:{xyxy_to_cxcywh(box_xyxy)}")

    box_xywh = [50, 30, 100, 70]
    print(f"\n  xywh:   {box_xywh}")
    print(f"  → xyxy: {xywh_to_xyxy(box_xywh)}")


# ── 2. IoU (Intersection over Union) ──────────────────────────────────────────
def compute_iou(b1, b2):
    """Both boxes in xyxy format."""
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1    = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2    = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def iou_demo():
    print("\n=== Intersection over Union (IoU) ===")
    print("  IoU = |A ∩ B| / |A ∪ B|  (ranges 0 to 1)")
    print()
    cases = [
        ("Perfect match",     [50, 50, 150, 150], [50, 50, 150, 150]),
        ("Partial overlap",   [50, 50, 150, 150], [100, 100, 200, 200]),
        ("No overlap",        [50, 50, 100, 100], [150, 150, 200, 200]),
        ("Contained",         [50, 50, 200, 200], [75, 75, 125, 125]),
        ("Near-miss",         [50, 50, 150, 150], [148, 50, 200, 150]),
    ]
    for name, b1, b2 in cases:
        iou = compute_iou(b1, b2)
        print(f"  {name:<18} IoU = {iou:.4f}")

    print()
    print("  Detection thresholds:")
    for t, desc in [(0.5, "COCO standard match"), (0.75, "Strict match"),
                    (0.5, "Typically used in PASCAL VOC")]:
        print(f"    IoU > {t}: {desc}")

    # GIoU, DIoU, CIoU overview
    print()
    print("  Extended IoU variants (for better gradient flow):")
    variants = [
        ("GIoU", "Generalised IoU; penalises non-overlapping by enclosing box area"),
        ("DIoU", "Distance IoU; adds centre-point distance penalty"),
        ("CIoU", "Complete IoU; adds aspect ratio consistency term"),
        ("EIoU", "Efficient IoU; separates w/h distance penalties"),
    ]
    for v, d in variants:
        print(f"    {v:<6} {d}")


# ── 3. NMS (Non-Maximum Suppression) ──────────────────────────────────────────
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes  : list of [x1,y1,x2,y2]
    scores : list of confidence scores
    Returns indices of kept boxes.
    """
    order = np.argsort(scores)[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious < iou_threshold]
    return keep


def nms_demo():
    print("\n=== Non-Maximum Suppression (NMS) ===")
    boxes  = [
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [120, 120, 220, 220],
        [300, 300, 400, 400],
        [305, 302, 405, 402],
    ]
    scores = [0.90, 0.75, 0.60, 0.85, 0.70]
    kept   = nms(boxes, scores, iou_threshold=0.5)

    print(f"  Input:  {len(boxes)} boxes")
    print(f"  Kept:   {len(kept)} boxes (indices {kept})")
    for k in kept:
        print(f"    Box {k}: score={scores[k]:.2f}  {boxes[k]}")
    print()
    print("  NMS variants:")
    variants = [
        ("Hard NMS",  "Remove all boxes with IoU > threshold"),
        ("Soft-NMS",  "Decay scores by IoU; gentler suppression"),
        ("DIoU-NMS",  "Use DIoU instead of IoU (overlapping in same class)"),
        ("WBF",       "Weighted Box Fusion: merge overlapping box coordinates"),
        ("Class-wise","Run NMS per class independently (standard)"),
    ]
    for v, d in variants:
        print(f"    {v:<12} {d}")


# ── 4. Anchor boxes ───────────────────────────────────────────────────────────
def anchor_boxes():
    print("\n=== Anchor Boxes ===")
    print("  Motivation: pre-define boxes at each spatial location; regress offsets")
    print()
    scales  = [0.5, 1.0, 2.0]
    ratios  = [0.5, 1.0, 2.0]
    base_sz = 32
    anchors = []
    for s in scales:
        for r in ratios:
            w = base_sz * s * np.sqrt(r)
            h = base_sz * s / np.sqrt(r)
            anchors.append((w, h))

    print(f"  Base size = {base_sz}  Scales = {scales}  Ratios = {ratios}")
    print(f"  Generated {len(anchors)} anchor shapes:")
    for i, (w, h) in enumerate(anchors):
        area = w * h; ratio = w / h
        print(f"    Anchor {i}: w={w:6.1f}  h={h:6.1f}  area={area:6.0f}  ratio={ratio:.2f}")

    print()
    print("  YOLO anchors: learned via k-means on training set bounding boxes")
    print("  Feature Pyramid Network (FPN): multi-scale feature maps → multi-scale anchors")


# ── 5. Detection architectures ───────────────────────────────────────────────
def detection_architectures():
    print("\n=== Object Detection Architectures ===")
    two_stage = [
        ("R-CNN (2014)",        "Region proposal (SS) → CNN per crop → SVM; slow"),
        ("Fast R-CNN (2015)",   "Single forward pass; RoI pooling; faster"),
        ("Faster R-CNN (2015)", "Region Proposal Network (RPN) → end-to-end"),
        ("Mask R-CNN (2017)",   "Faster R-CNN + mask head; instance seg too"),
        ("Cascade R-CNN (2018)","Multi-stage refinement with increasing IoU thresholds"),
    ]
    one_stage = [
        ("YOLO v1 (2016)",      "Single pass; S×S grid cells; 45 FPS"),
        ("SSD (2016)",          "Multi-scale anchors; VGG backbone"),
        ("RetinaNet (2017)",    "FPN + Focal Loss; solves class imbalance"),
        ("YOLO v5/v8 (2020/23)","CSP backbone; anchor-based+anchor-free variants"),
        ("FCOS (2019)",         "Anchor-free; centre-ness prediction"),
        ("DETR (2020)",         "Transformer-based; bipartite matching; no NMS"),
        ("DINO/Co-DETR (2023)", "DETR variants; state-of-art on COCO"),
    ]
    print("  Two-stage detectors:")
    for m, d in two_stage: print(f"    {m:<25} {d}")
    print("\n  One-stage detectors:")
    for m, d in one_stage: print(f"    {m:<25} {d}")


# ── 6. Detection metrics ──────────────────────────────────────────────────────
def detection_metrics():
    print("\n=== Detection Metrics ===")
    print("  mAP — mean Average Precision")
    print("    AP per class = area under Precision-Recall curve")
    print("    mAP = mean(AP) over all classes")
    print()
    print("  COCO metrics:")
    metrics = [
        ("mAP@50",       "IoU threshold 0.50"),
        ("mAP@75",       "IoU threshold 0.75 (stricter)"),
        ("mAP@[.5:.95]", "mean of mAP at IoU 0.5 to 0.95, step 0.05 (primary COCO)"),
        ("mAP_S",        "Small objects: area < 32² px"),
        ("mAP_M",        "Medium objects: 32² < area < 96² px"),
        ("mAP_L",        "Large objects: area > 96² px"),
        ("AR@1",         "Average Recall with max 1 det per image"),
        ("AR@100",       "Average Recall with max 100 dets per image"),
    ]
    for m, d in metrics:
        print(f"    {m:<20} {d}")

    # Toy precision-recall computation
    print()
    print("  Toy AP calculation (11-point interpolation):")
    precisions = np.array([1.0, 0.9, 0.85, 0.80, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2])
    recalls    = np.linspace(0, 1, 11)
    ap_11      = precisions.mean()
    print(f"    Precision at 11 recall levels: {np.round(precisions, 2)}")
    print(f"    AP (11-point interpolation): {ap_11:.4f}")


if __name__ == "__main__":
    bounding_box_formats()
    iou_demo()
    nms_demo()
    anchor_boxes()
    detection_architectures()
    detection_metrics()
