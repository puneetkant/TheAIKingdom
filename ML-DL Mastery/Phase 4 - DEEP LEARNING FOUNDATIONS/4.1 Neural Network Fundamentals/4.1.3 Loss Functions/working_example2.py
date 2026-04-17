"""
Working Example 2: Loss Functions — MSE, BCE, CE, Hinge, Huber
===============================================================
Manual loss implementations, gradients, numerical stability for cross-entropy.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def mse(y, y_hat):    return np.mean((y - y_hat)**2)
def mse_grad(y, y_hat): return 2 * (y_hat - y) / len(y)

def bce(y, p, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

def cross_entropy(y_true_idx, logits):
    """Numerically stable softmax cross-entropy."""
    z = logits - logits.max()
    log_sm = z - np.log(np.exp(z).sum())
    return -log_sm[y_true_idx]

def hinge(y, score):  return np.maximum(0, 1 - y * score).mean()
def huber(y, y_hat, delta=1.0):
    r = np.abs(y - y_hat)
    return np.where(r <= delta, 0.5*r**2, delta*(r - 0.5*delta)).mean()

def demo_losses():
    print("=== Loss Function Comparison ===")
    y_true = np.array([0., 1., 1., 0., 1.])
    y_pred = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    y_pm   = np.where(y_true == 0, -1., 1.)

    print(f"  MSE  = {mse(y_true, y_pred):.4f}")
    print(f"  BCE  = {bce(y_true, y_pred):.4f}")
    print(f"  Hinge= {hinge(y_pm, y_pred):.4f}")
    print(f"  Huber= {huber(y_true, y_pred, delta=0.5):.4f}")

def demo_bce_stability():
    print("\n=== BCE Numerical Stability ===")
    # Unstable (log of 0)
    p = np.array([0.0, 1.0])
    y = np.array([1.0, 0.0])
    safe_loss = bce(y, p)
    print(f"  Clipped BCE (p=[0,1]): {safe_loss:.4f}")

def demo_softmax_ce():
    print("\n=== Softmax Cross-Entropy ===")
    logits = np.array([2.0, 1.0, 0.1])
    for true_class in range(3):
        loss = cross_entropy(true_class, logits)
        print(f"  True class={true_class}:  CE={loss:.4f}")

def demo_loss_curves():
    p = np.linspace(0.01, 0.99, 100)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(p, -np.log(p), label="BCE (y=1)", lw=2)
    axes[0].plot(p, -np.log(1-p), label="BCE (y=0)", lw=2, ls="--")
    axes[0].set_xlabel("Predicted prob"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Binary Cross-Entropy"); axes[0].legend(); axes[0].set_ylim(0, 5)

    r = np.linspace(-3, 3, 200)
    axes[1].plot(r, r**2 / 2, label="MSE (Ld=½r²)", lw=2)
    axes[1].plot(r, np.abs(r), label="MAE", lw=2, ls="--")
    delta = 1.0
    huber_r = np.where(np.abs(r)<=delta, .5*r**2, delta*(np.abs(r)-.5*delta))
    axes[1].plot(r, huber_r, label=f"Huber(δ={delta})", lw=2, ls=":")
    axes[1].set_title("Regression Losses"); axes[1].legend(); axes[1].set_ylim(0, 4)

    plt.tight_layout(); plt.savefig(OUTPUT / "loss_functions.png"); plt.close()
    print("\n  Saved loss_functions.png")

if __name__ == "__main__":
    demo_losses()
    demo_bce_stability()
    demo_softmax_ce()
    demo_loss_curves()
