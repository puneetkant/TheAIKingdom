"""
Working Example: Regularization for Deep Learning
Covers L1/L2 weight decay, dropout, batch normalisation, layer normalisation,
early stopping, data augmentation, and DropConnect/spatial dropout concepts.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_regularization")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def relu(z):    return np.maximum(0, z)
def relu_d(z):  return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))


# -- 1. L1 and L2 Weight Decay -------------------------------------------------
def weight_decay():
    print("=== L1 and L2 Weight Decay ===")
    print("  L2: L_total = L + lambdaSigmaw²    gradient += 2lambdaw  (shrinks towards 0)")
    print("  L1: L_total = L + lambdaSigma|w|   gradient += lambda·sign(w)  (sparsity)")

    rng = np.random.default_rng(0)
    # Simulate weights being updated with/without L2 regularisation
    w_init = rng.standard_normal(100) * 2

    print(f"\n  {'Lambda':<10} {'No reg (std)':>14} {'L2 reg (std)':>14} {'L1 reg (std)':>14}")
    for lam in [0.0, 0.001, 0.01, 0.1]:
        w_l2 = w_init.copy()
        w_l1 = w_init.copy()
        w_nr = w_init.copy()
        lr = 0.01
        grad = rng.standard_normal(100) * 0.1  # simulated gradient
        for _ in range(200):
            w_nr -= lr * grad
            w_l2 -= lr * (grad + 2*lam*w_l2)
            w_l1 -= lr * (grad + lam*np.sign(w_l1))
        print(f"  {lam:<10} {w_nr.std():>14.4f} {w_l2.std():>14.4f} {w_l1.std():>14.4f}")

    print(f"\n  L2 (weight decay): uniform shrinkage, rarely zero")
    print(f"  L1 (Lasso):        promotes exact zeros (sparse weights)")


# -- 2. Dropout ----------------------------------------------------------------
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, X):
        if self.training:
            self.mask = (np.random.default_rng().random(X.shape) > self.p).astype(float)
            return X * self.mask / (1 - self.p)   # inverted dropout (scale up)
        return X  # at test time: use all neurons (expected value same)


def dropout_demo():
    print("\n=== Dropout ===")
    print("  During training: randomly zero out fraction p of neurons")
    print("  Inverted dropout: scale by 1/(1-p) to keep expected values equal")
    print("  At test: use all neurons, no scaling needed")
    print()

    X = np.ones((1, 10))   # simple all-ones input
    for p in [0.0, 0.2, 0.5, 0.8]:
        drop = Dropout(p)
        np.random.seed(42)
        out  = drop.forward(X)
        n_active = (out > 0).sum()
        print(f"  p={p}: active={n_active}/10  mean(out)={out.mean():.4f}  (expected 1.0)")

    print(f"\n  Dropout as ensemble: each pass uses different sub-network")
    print(f"  2^n possible networks (n=neurons); ensemble at test time")
    print(f"  Typical: p=0.5 hidden layers, p=0.1-0.2 input layer")


# -- 3. Batch Normalisation ----------------------------------------------------
def batch_normalisation():
    print("\n=== Batch Normalisation ===")
    print("  For each mini-batch: x = (x - mu_B) / sqrt(sigma²_B + epsilon)")
    print("  Then: y = gamma·x + beta  (learnable scale and shift)")
    print("  Benefits: reduces covariate shift, allows higher LR, mild regularisation")

    class BatchNorm:
        def __init__(self, d, eps=1e-5, momentum=0.1):
            self.gamma = np.ones(d)
            self.beta  = np.zeros(d)
            self.eps   = eps
            self.momentum = momentum
            self.run_mean = np.zeros(d)
            self.run_var  = np.ones(d)
            self.training = True

        def forward(self, x):
            if self.training:
                mu  = x.mean(axis=0)
                var = x.var(axis=0)
                self.run_mean = (1-self.momentum)*self.run_mean + self.momentum*mu
                self.run_var  = (1-self.momentum)*self.run_var  + self.momentum*var
            else:
                mu, var = self.run_mean, self.run_var
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            return self.gamma * x_hat + self.beta

    rng = np.random.default_rng(1)
    X   = rng.normal(5, 3, (32, 8))   # skewed distribution
    bn  = BatchNorm(8)
    Xbn = bn.forward(X)

    print(f"\n  Input  X: mean={X.mean(axis=0).round(2)}  std={X.std(axis=0).round(2)}")
    print(f"  After BN: mean={Xbn.mean(axis=0).round(4)}  std={Xbn.std(axis=0).round(4)}")
    print(f"  (gamma=1, beta=0 -> zero mean, unit variance)")

    print(f"\n  BatchNorm vs LayerNorm:")
    print(f"    BatchNorm: normalise across BATCH dimension per feature")
    print(f"               works well for CNNs; batch size > 1 required")
    print(f"    LayerNorm: normalise across FEATURE dimension per sample")
    print(f"               works for transformers, RNNs (any batch size)")


# -- 4. Layer Normalization ----------------------------------------------------
def layer_normalisation():
    print("\n=== Layer Normalisation ===")
    print("  x = (x - mu_sample) / sqrt(sigma²_sample + epsilon)  per sample")
    print("  Each sample normalised independently (batch size doesn't matter)")

    rng = np.random.default_rng(2)
    X   = rng.normal(5, 3, (4, 16))   # 4 samples, 16 features

    eps    = 1e-5
    mu     = X.mean(axis=1, keepdims=True)
    var    = X.var(axis=1, keepdims=True)
    X_norm = (X - mu) / np.sqrt(var + eps)

    print(f"\n  Input:    per-sample mean={X.mean(axis=1).round(2)}")
    print(f"  After LN: per-sample mean={X_norm.mean(axis=1).round(6)}  std={X_norm.std(axis=1).round(6)}")


# -- 5. Early Stopping ---------------------------------------------------------
def early_stopping():
    print("\n=== Early Stopping ===")
    print("  Monitor validation loss; stop when it doesn't improve for 'patience' epochs")
    print("  Implicit regularisation: prevents overfitting to training data")

    # Simulate training/validation curves
    rng = np.random.default_rng(3)
    ep  = np.arange(200)
    # Training loss: monotonically decreasing
    tr_loss  = 1.0 / (1 + 0.05*ep) + 0.1 * rng.standard_normal(200) * 0.02
    # Val loss: decreases then increases (overfitting)
    val_loss = (1.0 / (1 + 0.03*ep) + 0.1*np.sin(ep/50) * (ep/200)
                + rng.standard_normal(200) * 0.02)

    best_ep   = val_loss.argmin()
    patience  = 20
    stopped   = best_ep + patience if best_ep + patience < 200 else 199

    print(f"\n  Best val epoch: {best_ep}  val_loss={val_loss[best_ep]:.4f}")
    print(f"  With patience={patience}: stopped at epoch {stopped}")
    print(f"  Final train_loss={tr_loss[stopped]:.4f}  vs best val_epoch train_loss={tr_loss[best_ep]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ep, tr_loss,  label="Train loss")
    ax.plot(ep, val_loss, label="Val loss")
    ax.axvline(best_ep, color='g', lw=2, linestyle='--', label=f"Best val (ep={best_ep})")
    ax.axvline(stopped, color='r', lw=2, linestyle='--', label=f"Early stop (ep={stopped})")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Early Stopping Demo")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "early_stopping.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot saved: {path}")


# -- 6. Regularization comparison ---------------------------------------------
def regularization_comparison():
    print("\n=== Regularization Comparison ===")
    print(f"  {'Method':<25} {'Effect':<35} {'When to use'}")
    rows = [
        ("L2 (weight decay)", "Shrinks weights uniformly",    "Default; most networks"),
        ("L1",                "Promotes sparse weights",       "Feature selection needed"),
        ("Dropout",           "Ensemble of subnetworks",       "Fully-connected layers"),
        ("Batch Norm",        "Normalises layer inputs",       "CNNs, deep nets"),
        ("Layer Norm",        "Per-sample normalisation",      "Transformers, RNNs"),
        ("Early Stopping",    "Limits training time",          "Always; cheap to use"),
        ("Data Augmentation", "Increases effective data size", "Vision, NLP"),
        ("DropConnect",       "Drops weights (not neurons)",   "Similar to Dropout"),
    ]
    for r in rows:
        print(f"  {r[0]:<25} {r[1]:<35} {r[2]}")


if __name__ == "__main__":
    weight_decay()
    dropout_demo()
    batch_normalisation()
    layer_normalisation()
    early_stopping()
    regularization_comparison()
