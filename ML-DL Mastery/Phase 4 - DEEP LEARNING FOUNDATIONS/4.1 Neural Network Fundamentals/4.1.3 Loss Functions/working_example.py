"""
Working Example: Loss Functions
Covers MSE, MAE, Huber, cross-entropy (binary/categorical), KL divergence,
hinge loss, focal loss, contrastive/triplet loss, and loss selection guide.
"""
import numpy as np


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── 1. Regression losses ──────────────────────────────────────────────────────
def regression_losses():
    print("=== Regression Loss Functions ===")
    rng  = np.random.default_rng(0)
    y    = rng.uniform(0, 10, 100)
    yhat = y + rng.normal(0, 1, 100)
    # Introduce a few large outliers
    yhat[[10, 20, 30]] = yhat[[10, 20, 30]] + 20

    r = y - yhat

    mse   = np.mean(r**2)
    rmse  = np.sqrt(mse)
    mae   = np.mean(np.abs(r))

    # Huber loss
    delta = 1.0
    huber = np.where(np.abs(r) <= delta, 0.5*r**2, delta*(np.abs(r) - 0.5*delta))
    huber_loss = np.mean(huber)

    # Log-cosh
    logcosh = np.mean(np.log(np.cosh(r)))

    print(f"  {'Loss':<20} {'Value':>10}  {'Notes'}")
    for name, val, note in [
        ("MSE",         mse,        "Large errors penalised heavily"),
        ("RMSE",        rmse,       "Same units as y"),
        ("MAE",         mae,        "Robust to outliers"),
        (f"Huber(δ={delta})", huber_loss, "MAE far, MSE near zero"),
        ("Log-Cosh",    logcosh,    "Smooth MAE approximation"),
    ]:
        print(f"  {name:<20} {val:>10.4f}  {note}")

    # Gradient comparison at a specific residual
    print(f"\n  Gradients at r=5 (outlier):")
    r_val = 5.0
    print(f"    MSE gradient:   2r = {2*r_val:.1f}  (large → sensitive to outliers)")
    print(f"    MAE gradient:   sign(r) = 1.0  (constant)")
    print(f"    Huber gradient: δ·sign(r) = {delta:.1f}  (clipped)")


# ── 2. Binary cross-entropy ──────────────────────────────────────────────────
def binary_cross_entropy():
    print("\n=== Binary Cross-Entropy (BCE) ===")
    print("  L = -Σ [y·log(ŷ) + (1-y)·log(1-ŷ)] / n")
    print("  Also called log loss; used for binary classification")

    rng   = np.random.default_rng(1)
    y     = rng.integers(0, 2, 100).astype(float)
    logit = rng.normal(0, 1, 100)
    yhat  = sigmoid(logit)

    eps = 1e-15
    bce = -np.mean(y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))
    print(f"\n  n={len(y)}  BCE={bce:.4f}")

    # Manual examples
    print(f"\n  Manual examples:")
    for y_true, y_pred in [(1, 0.9), (1, 0.1), (0, 0.05), (0, 0.95)]:
        L = -(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))
        print(f"    y={y_true}  ŷ={y_pred}  L={L:.4f}")

    print(f"\n  Gradient: ∂L/∂ŷ = (ŷ - y) / (ŷ(1-ŷ))  →  ∂L/∂logit = ŷ - y")


# ── 3. Categorical cross-entropy ─────────────────────────────────────────────
def categorical_cross_entropy():
    print("\n=== Categorical Cross-Entropy ===")
    print("  L = -Σ y_k·log(ŷ_k)   (only the true class term survives)")

    rng    = np.random.default_rng(2)
    n, K   = 50, 4
    logits = rng.normal(0, 1, (n, K))
    probs  = softmax(logits)
    labels = rng.integers(0, K, n)

    # One-hot encode
    y_one_hot = np.zeros((n, K))
    y_one_hot[np.arange(n), labels] = 1

    eps = 1e-15
    cce = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1))
    print(f"\n  n={n}  K={K} classes  CCE={cce:.4f}")

    # Examples
    print(f"\n  Manual examples (K=3):")
    for true_cls, logit in [(0, [3,1,0]), (1, [0,2,1]), (2, [1,1,3])]:
        p = softmax(np.array(logit, dtype=float))
        L = -np.log(p[true_cls] + eps)
        print(f"    true={true_cls}  probs={p.round(3)}  L={L:.4f}")


# ── 4. KL Divergence ─────────────────────────────────────────────────────────
def kl_divergence():
    print("\n=== KL Divergence ===")
    print("  D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))")
    print("  Measures how P differs from Q (not symmetric!)")
    print("  Used in VAEs, knowledge distillation, RL")

    rng = np.random.default_rng(3)
    K   = 5
    P   = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    Q1  = np.array([0.2, 0.2, 0.2, 0.2, 0.2])   # uniform
    Q2  = np.array([0.4, 0.3, 0.1, 0.1, 0.1])   # same as P

    def kl(p, q, eps=1e-15):
        return np.sum(p * np.log((p+eps)/(q+eps)))

    print(f"\n  P = {P}")
    print(f"  Q1 (uniform) = {Q1}")
    print(f"  Q2 = P")
    print(f"\n  D_KL(P||Q1) = {kl(P,Q1):.4f}  (large: Q1 is uniform)")
    print(f"  D_KL(P||Q2) = {kl(P,Q2):.4f}  (zero: identical)")
    print(f"  D_KL(Q1||P) = {kl(Q1,P):.4f}  (asymmetric!)")


# ── 5. Hinge loss (SVM) ──────────────────────────────────────────────────────
def hinge_loss():
    print("\n=== Hinge Loss (SVM-style) ===")
    print("  L = max(0, 1 - y·ŷ)  where y∈{-1,+1}")
    print("  Encourages margin ≥ 1 around decision boundary")

    y    = np.array([ 1,  1, -1, -1,  1], dtype=float)
    yhat = np.array([0.8, 0.3, -0.9, 0.2, -0.5])  # raw scores

    hinge = np.maximum(0, 1 - y * yhat)
    print(f"\n  {'y':<6} {'ŷ':<8} {'Hinge':<8} {'Interpretation'}")
    for yi, yi_hat, h in zip(y, yhat, hinge):
        interp = "correct+margin" if h == 0 else ("correct no-margin" if yi*yi_hat > 0 else "wrong")
        print(f"  {yi:<6} {yi_hat:<8} {h:<8.2f} {interp}")

    print(f"\n  Mean hinge loss: {hinge.mean():.4f}")


# ── 6. Focal loss (class imbalance) ──────────────────────────────────────────
def focal_loss():
    print("\n=== Focal Loss (RetinaNet) ===")
    print("  FL(p_t) = -(1-p_t)^γ · log(p_t)")
    print("  Down-weights easy examples; focuses on hard misclassifications")
    print("  γ=0 → standard cross-entropy  γ=2 → strong focus on hard")

    eps  = 1e-15
    p_t  = np.array([0.99, 0.7, 0.5, 0.3, 0.1, 0.01])  # predicted prob for true class
    ce   = -np.log(p_t + eps)

    print(f"\n  {'p_t':<8} {'CE':>10} {'FL(γ=1)':>12} {'FL(γ=2)':>12}")
    for p, c in zip(p_t, ce):
        fl1 = (1-p)**1 * c
        fl2 = (1-p)**2 * c
        print(f"  {p:<8} {c:>10.4f} {fl1:>12.4f} {fl2:>12.4f}")

    print(f"\n  When p_t=0.99 (easy): FL/CE = {(1-0.99)**2:.6f}  (nearly zero weight)")
    print(f"  When p_t=0.1 (hard):  FL/CE = {(1-0.1)**2:.4f}  (large weight)")


# ── 7. Contrastive and Triplet loss ──────────────────────────────────────────
def metric_learning_losses():
    print("\n=== Contrastive and Triplet Loss (Metric Learning) ===")
    print("  Used in Siamese networks, face recognition, embedding learning")
    print()
    print("  Contrastive: L = y·d² + (1-y)·max(0, m-d)²")
    print("    y=1 (same class): minimise distance d")
    print("    y=0 (diff class): push distance > margin m")
    print()
    print("  Triplet:     L = max(0, d(a,p) - d(a,n) + margin)")
    print("    Anchor a, Positive p (same), Negative n (diff)")
    print("    Forces d(a,p) + margin < d(a,n)")

    rng = np.random.default_rng(4)
    d   = 0.4  # distance between embeddings
    m   = 1.0  # margin
    for y_pair, label in [(1, "same class"), (0, "diff class")]:
        L = y_pair * d**2 + (1-y_pair) * max(0, m-d)**2
        print(f"\n  Contrastive  y={y_pair} ({label})  d={d}  L={L:.4f}")

    # Triplet
    d_ap, d_an = 0.3, 0.5
    L_trip = max(0, d_ap - d_an + 0.2)
    print(f"\n  Triplet  d(a,p)={d_ap}  d(a,n)={d_an}  margin=0.2  L={L_trip:.4f}")
    d_an2 = 0.1
    L_trip2 = max(0, d_ap - d_an2 + 0.2)
    print(f"  Triplet  d(a,p)={d_ap}  d(a,n)={d_an2}  margin=0.2  L={L_trip2:.4f}  (violated!)")


# ── 8. Loss selection guide ───────────────────────────────────────────────────
def loss_selection_guide():
    print("\n=== Loss Function Selection Guide ===")
    print(f"  {'Task':<35} {'Loss':<28} {'Notes'}")
    rows = [
        ("Regression",               "MSE",                "Sensitive to outliers"),
        ("Regression (outliers)",    "Huber / MAE",        "Robust"),
        ("Binary classification",    "Binary cross-entropy","With sigmoid output"),
        ("Multi-class (one label)",  "Cat. cross-entropy", "With softmax output"),
        ("Multi-label classification","BCE per class",     "Independent sigmoids"),
        ("SVM / linear classifier",  "Hinge loss",         "Max-margin classifier"),
        ("Imbalanced classification","Focal loss",         "Down-weights easy examples"),
        ("Generative model",         "KL divergence",      "VAE latent space"),
        ("Embedding / metric learning","Triplet / Contrastive","Siamese networks"),
    ]
    for r in rows:
        print(f"  {r[0]:<35} {r[1]:<28} {r[2]}")


if __name__ == "__main__":
    regression_losses()
    binary_cross_entropy()
    categorical_cross_entropy()
    kl_divergence()
    hinge_loss()
    focal_loss()
    metric_learning_losses()
    loss_selection_guide()
