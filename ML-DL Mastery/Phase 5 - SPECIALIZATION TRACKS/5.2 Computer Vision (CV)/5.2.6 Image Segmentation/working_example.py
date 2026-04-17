"""
Working Example: Image Segmentation
Covers semantic vs instance vs panoptic segmentation,
U-Net architecture, pixel-level metrics, and superpixels.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_segmentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_synthetic_scene(H=64, W=64):
    """3-class synthetic image: background=0, object1=1, object2=2."""
    img   = np.zeros((H, W, 3), dtype=np.float32)
    label = np.zeros((H, W), dtype=np.int32)
    # Background gradient
    img  += np.linspace(0.1, 0.3, W)[None, :, None]
    # Object 1: red circle
    for i in range(H):
        for j in range(W):
            if (i-20)**2 + (j-20)**2 < 15**2:
                img[i, j]   = [0.9, 0.2, 0.2]
                label[i, j] = 1
    # Object 2: blue rectangle
    img[38:58, 38:58] = [0.2, 0.2, 0.9]
    label[38:58, 38:58] = 2
    return img, label


# ── 1. Segmentation types ────────────────────────────────────────────────────
def segmentation_types():
    print("=== Segmentation Task Types ===")
    types = [
        ("Semantic",  "Assign class label to every pixel; no instance distinction",
                      "FCN, DeepLab, SegFormer"),
        ("Instance",  "Detect + segment each object instance; different IDs for same class",
                      "Mask R-CNN, YOLACT, CondInst"),
        ("Panoptic",  "Semantic + instance; all pixels assigned; unified framework",
                      "Panoptic FPN, MaskFormer, Mask2Former"),
        ("Amodal",    "Segment visible + occluded regions of objects",
                      "AmodalMask, AISFormer"),
        ("Medical",   "High precision; often 3D (CT/MRI); U-Net dominates",
                      "U-Net, nnU-Net, TransUNet"),
    ]
    for t, desc, models in types:
        print(f"\n  {t} Segmentation:")
        print(f"    {desc}")
        print(f"    Models: {models}")


# ── 2. Pixel-level metrics ────────────────────────────────────────────────────
def compute_miou(pred, target, n_classes):
    ious = []
    for c in range(n_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return np.mean(ious), ious


def segmentation_metrics():
    print("\n=== Segmentation Metrics ===")
    _, label = make_synthetic_scene()
    n_classes = 3
    # Simulate a noisy prediction
    rng  = np.random.default_rng(0)
    pred = label.copy()
    noise_mask = rng.random(label.shape) < 0.1
    pred[noise_mask] = rng.integers(0, n_classes, noise_mask.sum())

    pixel_acc = (pred == label).mean()
    miou, per_class = compute_miou(pred, label, n_classes)

    print(f"  Pixel accuracy: {pixel_acc:.4f}")
    print(f"  mIoU:           {miou:.4f}")
    for c in range(n_classes):
        print(f"    Class {c} IoU: {per_class[c]:.4f}  "
              f"({int((label==c).sum())} pixels)")

    print()
    print("  Key metrics summary:")
    ms = [
        ("Pixel Accuracy",    "(TP + TN) / total pixels; biased toward large classes"),
        ("Mean Pixel Acc.",   "Mean accuracy per class; balances small classes"),
        ("mIoU",              "Mean IoU over classes; primary benchmark metric"),
        ("Dice / F1",         "2·TP / (2·TP + FP + FN); common in medical imaging"),
        ("Boundary F1 (BF1)", "Evaluates quality of predicted boundaries"),
        ("Panoptic Quality",  "PQ = SQ × RQ; recognition + segmentation quality"),
    ]
    for m, d in ms:
        print(f"    {m:<22} {d}")


# ── 3. U-Net architecture ────────────────────────────────────────────────────
def unet_architecture():
    print("\n=== U-Net Architecture ===")
    print("  Originally for biomedical image segmentation (Ronneberger 2015)")
    print()
    print("  Architecture: Encoder–Decoder with skip connections")
    print()

    # Print architecture as text diagram
    levels = [
        (572, 572,  64, "Input"),
        (570, 570,  64, "Conv3×3, 64, ReLU ×2"),
        (284, 284,  64, "MaxPool 2×2"),
        (282, 282, 128, "Conv3×3, 128, ReLU ×2"),
        (141, 141, 128, "MaxPool 2×2"),
        (139, 139, 256, "Conv3×3, 256, ReLU ×2"),
        ( 69,  69, 256, "MaxPool 2×2"),
        ( 67,  67, 512, "Conv3×3, 512, ReLU ×2"),
        ( 33,  33, 512, "MaxPool 2×2"),
        ( 31,  31,1024, "Conv3×3, 1024 (bottleneck)"),
    ]
    print("  Encoder (contracting path):")
    for h, w, c, desc in levels:
        print(f"    {h:>3}×{w:<3} ×{c:<4}  {desc}")

    print()
    print("  Decoder (expanding path) with skip connections:")
    print("    UpConv 2×2 → concat skip → Conv3×3 ×2  (×4)")
    print("    Final: Conv1×1 → n_classes")
    print()
    print("  Key features:")
    print("    Skip connections preserve fine spatial detail")
    print("    Trained with overlap-tile strategy for large images")
    print("    Data augmentation (elastic deformations) critical")
    print("    Loss: weighted cross-entropy (border pixels up-weighted)")


# ── 4. Fully convolutional network (FCN) ─────────────────────────────────────
def fcn_overview():
    print("\n=== FCN (Fully Convolutional Network) ===")
    print("  Long et al. (2015) — first end-to-end CNN for semantic segmentation")
    print()
    print("  Key idea: replace FC layers in classifiers with conv layers")
    print("    Classifier (AlexNet/VGG): conv → pool → FC → FC → softmax")
    print("    FCN:                      conv → pool → conv → upsample → class map")
    print()
    print("  Upsampling methods:")
    ups = [
        ("Bilinear interp.", "Fixed upsampling; no learnable params"),
        ("Transposed conv.",  "Learnable; can learn to upsample"),
        ("Unpooling",         "Remember max-pool indices; sparse upsample"),
        ("Pixel shuffle",     "Rearrange channel dims to spatial (ESPCN)"),
    ]
    for m, d in ups:
        print(f"    {m:<20} {d}")
    print()
    print("  FCN variants:")
    print("    FCN-32s: stride 32 upsample (coarse)")
    print("    FCN-16s: combine pool4 skip → stride 16")
    print("    FCN-8s:  combine pool3 skip → stride 8  (best quality)")


# ── 5. Modern methods overview ────────────────────────────────────────────────
def modern_methods():
    print("\n=== Modern Segmentation Models ===")
    models = [
        ("DeepLab v3+",   2018, "Atrous conv + ASPP + decoder; state-of-art on VOC"),
        ("PSPNet",        2017, "Pyramid pooling module; scene parsing"),
        ("SegFormer",     2021, "Mix Transformer (MiT) encoder; lightweight"),
        ("Mask2Former",   2022, "Masked attention transformer; panoptic sota"),
        ("SAM",           2023, "Segment Anything; zero-shot; 1B params; 11M images"),
        ("SAM 2",         2024, "Real-time video segmentation; streaming memory"),
        ("SegGPT",        2023, "In-context segmentation; arbitrary target"),
        ("OneFormer",     2022, "Single model for all three task types"),
    ]
    print(f"  {'Model':<16} {'Year'} {'Notes'}")
    print(f"  {'─'*16} {'─'*4} {'─'*50}")
    for m, y, d in models:
        print(f"  {m:<16} {y}  {d}")

    print()
    print("  Benchmarks:")
    benchmarks = [
        ("PASCAL VOC 2012",   "21 classes; mIoU; DeepLab v3+ ~89.0"),
        ("ADE20K",            "150 classes; mIoU; SegFormer-B5 ~51.8"),
        ("Cityscapes",        "19 classes; urban driving; mIoU ~85+"),
        ("COCO panoptic",     "PQ; Mask2Former ~57.8 PQ"),
        ("Medical (BraTS)",   "Dice; 3D brain tumour MRI"),
    ]
    for b, d in benchmarks:
        print(f"    {b:<22} {d}")


if __name__ == "__main__":
    segmentation_types()
    segmentation_metrics()
    unet_architecture()
    fcn_overview()
    modern_methods()
