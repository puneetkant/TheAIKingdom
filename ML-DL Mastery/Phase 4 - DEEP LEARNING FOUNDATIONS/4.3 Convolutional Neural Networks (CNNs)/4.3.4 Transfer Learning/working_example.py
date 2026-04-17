"""
Working Example: Transfer Learning
Covers feature extraction, fine-tuning, domain adaptation, when to use
transfer learning, layer freezing, learning rate strategies, and examples.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os


# ── Simulate pretrained CNN features ─────────────────────────────────────────
def make_pretrained_features(n_samples, n_classes, n_features=2048, rng=None):
    """Simulate high-quality features from a pretrained model."""
    rng = rng or np.random.default_rng(42)
    # Features cluster around class centroids (simulating good representations)
    centroids = rng.standard_normal((n_classes, n_features)) * 3
    labels    = rng.integers(0, n_classes, n_samples)
    X = centroids[labels] + rng.standard_normal((n_samples, n_features)) * 0.5
    return X, labels

def make_random_features(n_samples, n_classes, n_features=2048, rng=None):
    """Random features (no pretrained knowledge)."""
    rng = rng or np.random.default_rng(42)
    labels = rng.integers(0, n_classes, n_samples)
    X = rng.standard_normal((n_samples, n_features))
    return X, labels


# ── 1. Motivation ─────────────────────────────────────────────────────────────
def motivation():
    print("=== Transfer Learning: Motivation ===")
    print("  Deep networks learn hierarchical representations:")
    print("    Layer 1: edges, blobs, colour gradients")
    print("    Layer 2: textures, corners, simple shapes")
    print("    Layer 3: object parts (eyes, wheels, ...)")
    print("    Layer 4-N: high-level semantics")
    print()
    print("  These low-level features are universal → reuse across tasks!")
    print()
    print("  Without transfer learning:")
    print("    • Need millions of labelled examples per task")
    print("    • Training from scratch takes days/weeks")
    print()
    print("  With transfer learning:")
    print("    • Fine-tune with hundreds/thousands of examples")
    print("    • Training takes hours/minutes")
    print("    • Often outperforms scratch training with limited data")


# ── 2. Feature extraction ─────────────────────────────────────────────────────
def feature_extraction():
    print("\n=== Strategy 1: Feature Extraction ===")
    print("  Freeze all pretrained layers → pass data through → extract embeddings")
    print("  Train only a small classifier on top")
    print()

    rng = np.random.default_rng(1)
    n_train, n_test, n_classes = 200, 100, 5
    n_features = 512  # e.g., ResNet-50 avg pool output

    X_pretrained, y = make_pretrained_features(n_train + n_test, n_classes, n_features, rng)
    X_random,     _ = make_random_features(n_train + n_test, n_classes, n_features, rng)

    scaler_pre = StandardScaler().fit(X_pretrained[:n_train])
    scaler_rnd = StandardScaler().fit(X_random[:n_train])

    Xtr_pre = scaler_pre.transform(X_pretrained[:n_train])
    Xts_pre = scaler_pre.transform(X_pretrained[n_train:])
    Xtr_rnd = scaler_rnd.transform(X_random[:n_train])
    Xts_rnd = scaler_rnd.transform(X_random[n_train:])
    y_train  = y[:n_train]
    y_test   = y[n_train:]

    for name, Xtr, Xts in [("Pretrained features", Xtr_pre, Xts_pre),
                             ("Random features", Xtr_rnd, Xts_rnd)]:
        clf = LogisticRegression(max_iter=500, random_state=0).fit(Xtr, y_train)
        acc = clf.score(Xts, y_test)
        print(f"  {name:<25}: test accuracy = {acc:.4f}")

    print()
    print("  Algorithm:")
    print("  1. Load pretrained model (ResNet, VGG, EfficientNet)")
    print("  2. Remove final classification head")
    print("  3. Freeze all weights (requires_grad=False in PyTorch)")
    print("  4. Run all images through frozen model → save embeddings")
    print("  5. Train LR / SVM / small MLP on embeddings")


# ── 3. Fine-tuning ────────────────────────────────────────────────────────────
def fine_tuning():
    print("\n=== Strategy 2: Fine-tuning ===")
    print("  Unfreeze some/all pretrained layers; retrain with small LR")
    print()
    print("  Typical schedule:")
    print("    Phase 1: freeze all; train new head (5-10 epochs, lr=1e-3)")
    print("    Phase 2: unfreeze top blocks; train with lr=1e-4 to 1e-5")
    print("    Phase 3: unfreeze all; discriminative LR (lower for early layers)")
    print()
    print("  Discriminative learning rates:")
    print("    Early layers: lr × 0.01  (universal features — barely change)")
    print("    Middle layers: lr × 0.1  (task-general features)")
    print("    Final layers: lr × 1.0   (task-specific features — retrain fully)")
    print()

    # Simulate parameter counts to show what gets trained
    layers = [
        ("Block 1 (conv1-7)", 0.006, "Universal edges"),
        ("Block 2",           0.11,  "Textures"),
        ("Block 3",           0.51,  "Object parts"),
        ("Block 4",           1.5,   "High-level semantics"),
        ("Classifier head",   0.002, "Task-specific"),
    ]
    print(f"  {'Layer block':<24} {'Params(M)':<12} {'Phase 1':<12} {'Phase 2':<12} {'Phase 3'}")
    for name, params, desc in layers:
        p1 = "Frozen" if name != "Classifier head" else "Trained"
        p2 = "Frozen" if "Block 1" in name or "Block 2" in name else "Trained"
        p3 = "Trained"
        print(f"  {name:<24} {params:<12.3f} {p1:<12} {p2:<12} {p3}")


# ── 4. When to use transfer learning ─────────────────────────────────────────
def when_to_transfer():
    print("\n=== Decision Guide: Transfer Learning Strategy ===")
    print()
    print(f"  {'Dataset size':<20} {'Similar to source?':<22} {'Recommended strategy'}")
    rows = [
        ("Small (<1K)",      "Yes (e.g. dogs→cats)",  "Feature extraction only — risk overfitting if fine-tuning"),
        ("Small (<1K)",      "No (medical imaging)",  "Feature extraction; try top few layers only"),
        ("Medium (1K-100K)", "Yes",                   "Fine-tune top layers (Phase 1+2)"),
        ("Medium (1K-100K)", "No",                    "Fine-tune more layers with low LR"),
        ("Large (>100K)",    "Yes",                   "Full fine-tuning or train from scratch"),
        ("Large (>100K)",    "No",                    "Train from scratch OR full fine-tuning"),
    ]
    for ds, sim, strat in rows:
        print(f"  {ds:<20} {sim:<22} {strat}")


# ── 5. Popular pretrained models ──────────────────────────────────────────────
def pretrained_models():
    print("\n=== Popular Pretrained Models (ImageNet) ===")
    models = [
        ("ResNet-50",        "25.6M",  "Top-1: 76.0%",  "Torchvision/TF Hub; very stable"),
        ("ResNet-101",       "44.5M",  "Top-1: 77.4%",  "Larger ResNet"),
        ("VGG16",            "138M",   "Top-1: 71.6%",  "Large but simple"),
        ("EfficientNet-B4",  "19.3M",  "Top-1: 82.6%",  "Best accuracy/params trade-off"),
        ("MobileNetV3-L",    "5.5M",   "Top-1: 75.8%",  "Mobile/edge deployment"),
        ("DenseNet-121",     "8.0M",   "Top-1: 74.4%",  "Medical imaging popular"),
        ("ConvNeXt-Tiny",    "28.6M",  "Top-1: 82.1%",  "Modern CNN, ViT-style training"),
        ("ViT-B/16",         "86M",    "Top-1: 81.8%",  "Vision Transformer"),
        ("CLIP-ViT-L/14",    "427M",   "Multi-modal",   "Zero-shot; language-image pairs"),
    ]
    print(f"  {'Model':<20} {'Params':<10} {'Accuracy':<16} {'Notes'}")
    for name, params, acc, note in models:
        print(f"  {name:<20} {params:<10} {acc:<16} {note}")


# ── 6. Domain adaptation ──────────────────────────────────────────────────────
def domain_adaptation():
    print("\n=== Domain Adaptation ===")
    print("  Source domain ≠ Target domain (different data distributions)")
    print()
    print("  Examples:")
    print("    Synthetic → Real (GAN rendered cars → real photos)")
    print("    RGB photos → Medical scans (X-ray, MRI)")
    print("    Daytime → Nighttime driving")
    print()
    print("  Techniques:")
    techniques = [
        ("Instance reweighting",    "Upweight target-like source samples"),
        ("Feature alignment",       "MMD, CORAL: align source & target feature dists"),
        ("Adversarial training",    "Domain discriminator (DANN): confuse source/target"),
        ("Self-training (pseudo)",  "Train on target with model's own predictions"),
        ("CycleGAN style transfer", "Translate source images to look like target domain"),
    ]
    for name, desc in techniques:
        print(f"    {name:<28}: {desc}")


# ── 7. Practical example ──────────────────────────────────────────────────────
def practical_example():
    print("\n=== Practical Transfer Learning Example ===")
    print("  Task: classify 5 flower types (200 training images per class)")
    print("  Source: ImageNet pretrained ResNet-50")
    print()
    print("  Code (PyTorch pseudocode):")
    code = """
  import torch
  from torchvision import models, transforms

  # 1. Load pretrained model
  model = models.resnet50(pretrained=True)

  # 2. Freeze all layers
  for param in model.parameters():
      param.requires_grad = False

  # 3. Replace final FC layer (1000 classes → 5)
  num_features = model.fc.in_features       # 2048
  model.fc = torch.nn.Linear(num_features, 5)
  # Only model.fc parameters have requires_grad=True

  # 4. Phase 1: train head only
  optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
  train(model, optimizer, epochs=10)

  # 5. Phase 2: unfreeze last ResNet block
  for param in model.layer4.parameters():
      param.requires_grad = True
  optimizer = torch.optim.Adam([
      {"params": model.layer4.parameters(), "lr": 1e-4},
      {"params": model.fc.parameters(),     "lr": 1e-3},
  ])
  train(model, optimizer, epochs=10)
    """
    print(code)
    print("  Expected results:")
    print("    Scratch training (5k samples): ~65% accuracy")
    print("    Feature extraction:            ~88% accuracy")
    print("    Fine-tuning top block:         ~92% accuracy")
    print("    Full fine-tuning:              ~94% accuracy")


if __name__ == "__main__":
    motivation()
    feature_extraction()
    fine_tuning()
    when_to_transfer()
    pretrained_models()
    domain_adaptation()
    practical_example()
