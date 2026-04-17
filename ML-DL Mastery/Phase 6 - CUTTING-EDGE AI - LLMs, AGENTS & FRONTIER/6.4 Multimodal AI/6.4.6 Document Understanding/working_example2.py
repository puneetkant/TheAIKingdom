"""
Working Example 2: Document Understanding
Layout-aware feature extraction, bounding box analysis,
and table structure recognition from a synthetic document.
Run: python working_example2.py
"""
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def iou(box_a, box_b):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-10)


def extract_layout_features(box):
    """Extract layout features from a bounding box [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return {
        "x_center": (x1 + x2) / 2,
        "y_center": (y1 + y2) / 2,
        "width": w,
        "height": h,
        "area": w * h,
        "aspect_ratio": w / (h + 1e-10),
        "x_norm": (x1 + x2) / 2 / 800,
        "y_norm": (y1 + y2) / 2 / 1000,
    }


def demo():
    print("=== Document Understanding: Layout Feature Extraction ===")

    # Synthetic document layout (page 800×1000)
    PAGE_W, PAGE_H = 800, 1000

    elements = [
        {"type": "title",    "box": [50,  40, 750, 100]},
        {"type": "text",     "box": [50, 120, 370, 300]},
        {"type": "text",     "box": [430, 120, 750, 300]},
        {"type": "table",    "box": [50, 320, 750, 550]},
        {"type": "figure",   "box": [50, 570, 370, 780]},
        {"type": "caption",  "box": [50, 790, 370, 840]},
        {"type": "text",     "box": [430, 570, 750, 840]},
        {"type": "footer",   "box": [50, 950, 750, 990]},
    ]

    # Table rows (within the table bounding box)
    table_box = elements[3]["box"]
    row_h = (table_box[3] - table_box[1]) / 5
    table_rows = []
    for r in range(5):
        y1 = table_box[1] + r * row_h
        row = {"type": f"row_{r}", "box": [table_box[0], y1, table_box[2], y1 + row_h]}
        table_rows.append(row)

    features = [extract_layout_features(e["box"]) for e in elements]
    print(f"  Document: {PAGE_W}×{PAGE_H}px, {len(elements)} elements")
    for elem, feat in zip(elements, features):
        print(f"  [{elem['type']:8s}] area={feat['area']:6.0f}, "
              f"aspect={feat['aspect_ratio']:.2f}, y_norm={feat['y_norm']:.2f}")

    # Reading order (top-to-bottom, left-to-right)
    sorted_elems = sorted(elements, key=lambda e: (e["box"][1] // 100, e["box"][0]))
    print("\n  Reading order:", [e["type"] for e in sorted_elems])

    COLORS = {
        "title": "#e74c3c", "text": "#3498db", "table": "#2ecc71",
        "figure": "#9b59b6", "caption": "#f39c12", "footer": "#95a5a6"
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # Document layout
    for elem in elements:
        x1, y1, x2, y2 = elem["box"]
        color = COLORS.get(elem["type"], "#bdc3c7")
        rect = Rectangle((x1, PAGE_H - y2), x2 - x1, y2 - y1,
                          linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.3)
        axes[0].add_patch(rect)
        axes[0].text((x1 + x2) / 2, PAGE_H - (y1 + y2) / 2, elem["type"],
                      ha="center", va="center", fontsize=7, color="black")
    axes[0].set(xlim=(0, PAGE_W), ylim=(0, PAGE_H),
                title="Document Layout", xlabel="x", ylabel="y")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.2)

    # Feature distribution
    areas = [f["area"] for f in features]
    types = [e["type"] for e in elements]
    colors_list = [COLORS.get(t, "#bdc3c7") for t in types]
    axes[1].bar(types, areas, color=colors_list)
    axes[1].set(ylabel="Area (px²)", title="Element Area by Type")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    # IoU matrix between elements
    n = len(elements)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            iou_matrix[i, j] = iou(elements[i]["box"], elements[j]["box"])
    im = axes[2].imshow(iou_matrix, cmap="Blues", vmin=0, vmax=0.3)
    axes[2].set(title="IoU Between Elements",
                xticks=range(n), yticks=range(n),
                xticklabels=[e["type"][:4] for e in elements],
                yticklabels=[e["type"][:4] for e in elements])
    axes[2].tick_params(axis="x", rotation=30)
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT / "document_understanding.png", dpi=100)
    plt.close()
    print("  Saved document_understanding.png")


if __name__ == "__main__":
    demo()
