"""
Working Example: Image Classification
Covers linear classifier, CNN-based classification pipeline,
training tricks, evaluation metrics, and top-1/top-5 accuracy.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_classification")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def relu(z): return np.maximum(0, z)


# ── 1. Dataset overview ───────────────────────────────────────────────────────
def dataset_overview():
    print("=== Image Classification Datasets ===")
    datasets = [
        ("MNIST",      "70K 28×28 gray",    "10 classes digits",         "~99% achievable"),
        ("CIFAR-10",   "60K 32×32 RGB",     "10 classes (animals/vehicles)","~99% with large ViT"),
        ("CIFAR-100",  "60K 32×32 RGB",     "100 classes",               "~92% with strong models"),
        ("ImageNet",   "1.2M 224×224 RGB",  "1000 classes",              "Top-1 ~90%+ ViT-G"),
        ("iNaturalist","859K variable",      "8000+ species",             "Challenging fine-grained"),
    ]
    print(f"  {'Dataset':<12} {'Size':<20} {'Description':<30} Sota")
    print(f"  {'─'*12} {'─'*20} {'─'*30} {'─'*20}")
    for d, s, desc, sota in datasets:
        print(f"  {d:<12} {s:<20} {desc:<30} {sota}")


# ── 2. Linear classifier (softmax regression) ─────────────────────────────────
def linear_classifier():
    print("\n=== Linear Classifier on Digits ===")
    digits = load_digits()
    X, y   = digits.data, digits.target
    X      = StandardScaler().fit_transform(X)
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    yp  = clf.predict(Xts)
    acc = accuracy_score(yts, yp)
    print(f"  Digits (8×8 grey) | Train: {len(Xtr)}  Test: {len(Xts)}")
    print(f"  Linear classifier accuracy: {acc:.4f}")
    print()
    # Confusion matrix (top-3 most confused pairs)
    cm = confusion_matrix(yts, yp)
    np.fill_diagonal(cm, 0)  # zero diag to find mistakes
    flat = cm.flatten()
    top_pairs = np.argsort(flat)[::-1][:3]
    print("  Top-3 most confused class pairs:")
    for idx in top_pairs:
        i, j = divmod(idx, 10)
        if cm[i,j] > 0:
            print(f"    True={i} → Pred={j}: {cm[i,j]} mistakes")


# ── 3. CNN from scratch ───────────────────────────────────────────────────────
class SimpleConvNet:
    """
    Minimal 2-layer CNN for 8×8 images (Digits dataset).
    Layer 1: Conv 3×3, 8 filters → ReLU → MaxPool 2×2
    Layer 2: Dense 200 → ReLU
    Output:  Dense 10 → Softmax
    """
    def __init__(self, rng=None):
        rng = rng or np.random.default_rng(0)
        s = 0.05
        # Conv layer: 8 filters, 3×3, single channel
        self.F  = rng.standard_normal((8, 1, 3, 3)) * s   # (nf, 1, kH, kW)
        self.bf = np.zeros(8)
        # FC layers (input: 8×3×3 = 72 after pool from 8×8 → 3×3 per filter)
        self.W1 = rng.standard_normal((72, 32)) * s   # flattened after pool
        self.b1 = np.zeros(32)
        self.W2 = rng.standard_normal((32, 10)) * s
        self.b2 = np.zeros(10)

    def conv_pool(self, X):
        """X: (B, 8, 8)  → (B, 8, 3, 3) after conv+maxpool"""
        B, H, W = X.shape
        nf = self.F.shape[0]; kH, kW = 3, 3
        out_H, out_W = H - kH + 1, W - kW + 1   # 6×6
        C = np.zeros((B, nf, out_H, out_W))
        for f in range(nf):
            k = self.F[f, 0]
            for i in range(out_H):
                for j in range(out_W):
                    C[:, f, i, j] = (X[:, i:i+kH, j:j+kW] * k).sum(axis=(1,2))
        C = relu(C + self.bf[None, :, None, None])
        # MaxPool 2×2
        ph, pw = out_H // 2, out_W // 2
        P = np.zeros((B, nf, ph, pw))
        for i in range(ph):
            for j in range(pw):
                P[:,:,i,j] = C[:,:,2*i:2*i+2, 2*j:2*j+2].max(axis=(2,3))
        return P

    def forward(self, X):
        P  = self.conv_pool(X)
        h1 = relu(P.reshape(len(X), -1) @ self.W1 + self.b1)
        return softmax(h1 @ self.W2 + self.b2), h1

    def fit(self, X, y, epochs=30, lr=0.05, bs=32, rng=None):
        rng = rng or np.random.default_rng(1)
        n   = len(X)
        losses = []
        for ep in range(epochs):
            idx = rng.permutation(n); ep_loss = 0
            for i in range(0, n, bs):
                Xb = X[idx[i:i+bs]]; yb = y[idx[i:i+bs]]
                probs, h1 = self.forward(Xb)
                ce = -np.log(probs[np.arange(len(yb)), yb] + 1e-9).mean()
                ep_loss += ce * len(Xb)
                # Grad at output (simplified, skip conv grad)
                dL = probs.copy(); dL[np.arange(len(yb)), yb] -= 1
                dL /= len(yb)
                dW2 = h1.T @ dL; db2 = dL.sum(0)
                self.W2 -= lr * np.clip(dW2, -1, 1)
                self.b2 -= lr * np.clip(db2, -1, 1)
                dh1 = dL @ self.W2.T * (h1 > 0)
                P_flat = self.conv_pool(Xb).reshape(len(Xb), -1)
                dW1 = P_flat.T @ dh1; db1 = dh1.sum(0)
                self.W1 -= lr * np.clip(dW1, -1, 1)
                self.b1 -= lr * np.clip(db1, -1, 1)
            losses.append(ep_loss / n)
        return losses


def cnn_demo():
    print("\n=== Minimal CNN on Digits (8×8) ===")
    digits = load_digits()
    X_raw  = digits.data.reshape(-1, 8, 8).astype(float) / 16.0
    y      = digits.target
    rng    = np.random.default_rng(42)
    idx    = rng.permutation(len(X_raw))
    X_raw, y = X_raw[idx], y[idx]
    Xtr, Xts = X_raw[:1400], X_raw[1400:]
    ytr, yts = y[:1400],     y[1400:]

    model = SimpleConvNet(rng=rng)
    losses = model.fit(Xtr, ytr, epochs=20, lr=0.05, rng=rng)

    probs_tr, _ = model.forward(Xtr)
    probs_ts, _ = model.forward(Xts)
    acc_tr = (probs_tr.argmax(1) == ytr).mean()
    acc_ts = (probs_ts.argmax(1) == yts).mean()

    print(f"  Train: {len(Xtr)}  Test: {len(Xts)}")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Train acc: {acc_tr:.4f}  Test acc: {acc_ts:.4f}")


# ── 4. Top-k accuracy ─────────────────────────────────────────────────────────
def topk_accuracy():
    print("\n=== Top-k Accuracy ===")
    print("  Top-1: correct class is the argmax prediction")
    print("  Top-5: correct class is among top-5 predicted classes")
    print("  Standard for ImageNet benchmarking")
    print()

    rng  = np.random.default_rng(7)
    n, C = 100, 1000
    logits = rng.standard_normal((n, C))
    labels = rng.integers(0, C, n)

    def topk(logits, labels, k):
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]
        correct     = sum(labels[i] in top_k_preds[i] for i in range(n))
        return correct / n

    print(f"  Simulated model (n={n}, C={C}):")
    for k in [1, 5, 10]:
        print(f"    Top-{k} accuracy: {topk(logits, labels, k):.4f}  "
              f"(random baseline ≈ {min(k/C, 1):.4f})")

    print()
    print("  ImageNet historical top-5 accuracy:")
    models = [
        ("AlexNet (2012)",  83.6),
        ("VGG-16 (2014)",   92.7),
        ("GoogLeNet (2014)",93.3),
        ("ResNet-50 (2015)",92.1),
        ("EfficientNet-B7", 97.1),
        ("ViT-H (2021)",    97.5),
        ("SoTA (2024)",     98.0),
    ]
    for m, acc in models:
        print(f"    {m:<22} {acc:.1f}%")


# ── 5. Common evaluation metrics ─────────────────────────────────────────────
def evaluation_metrics():
    print("\n=== Image Classification Metrics ===")
    metrics = [
        ("Top-1 accuracy",   "Fraction of samples where argmax = true class"),
        ("Top-5 accuracy",   "True class is in top-5 predictions"),
        ("Per-class accuracy","Accuracy for each category separately"),
        ("Macro F1",         "Unweighted mean of per-class F1 scores"),
        ("Weighted F1",      "F1 weighted by class frequency"),
        ("Confusion matrix", "C×C matrix showing prediction vs true class"),
        ("AUC-ROC",          "Area under ROC; for multi-class: OvR average"),
        ("ECE",              "Expected Calibration Error; measures over/under-confidence"),
    ]
    for m, d in metrics:
        print(f"  {m:<20} {d}")


if __name__ == "__main__":
    dataset_overview()
    linear_classifier()
    cnn_demo()
    topk_accuracy()
    evaluation_metrics()
