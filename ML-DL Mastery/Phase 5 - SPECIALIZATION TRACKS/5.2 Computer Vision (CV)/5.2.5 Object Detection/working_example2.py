"""
Working Example 2: Object Detection — sliding window + IoU demo
================================================================
Implements IoU, NMS, and a sliding window concept on synthetic boxes.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def iou(box1, box2):
    """box = [x1, y1, x2, y2]"""
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-9)

def nms(boxes, scores, iou_thresh=0.5):
    """Non-Maximum Suppression."""
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        rest = order[1:]
        order = [j for j in rest if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

def demo():
    print("=== Object Detection Concepts ===")
    # Synthetic detection boxes with scores
    boxes = np.array([
        [10, 10, 60, 60], [12, 12, 62, 62], [15, 15, 65, 65],  # overlapping detections
        [100, 80, 150, 130], [102, 82, 152, 132],               # second object cluster
        [50, 150, 90, 200],                                     # isolated box
    ], dtype=float)
    scores = np.array([0.95, 0.85, 0.75, 0.90, 0.80, 0.70])

    print(f"  Boxes before NMS: {len(boxes)}")
    kept = nms(boxes, scores, iou_thresh=0.5)
    print(f"  Boxes after NMS: {len(kept)} — indices: {kept}")

    # IoU demo
    b1, b2 = boxes[0], boxes[1]
    print(f"  IoU(box0, box1) = {iou(b1, b2):.4f}")
    print(f"  IoU(box0, box3) = {iou(boxes[0], boxes[3]):.4f}")

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, title, sel_boxes, sel_scores in [
        (axes[0], "Before NMS", boxes, scores),
        (axes[1], "After NMS",  boxes[kept], scores[kept]),
    ]:
        ax.set_xlim(0, 200); ax.set_ylim(0, 220); ax.invert_yaxis()
        ax.set_title(title)
        for b, s in zip(sel_boxes, sel_scores):
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                       linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(b[0], b[1]-3, f"{s:.2f}", color="red", fontsize=8)
    plt.tight_layout(); plt.savefig(OUTPUT / "object_detection_nms.png"); plt.close()
    print("  Saved object_detection_nms.png")

if __name__ == "__main__":
    demo()
