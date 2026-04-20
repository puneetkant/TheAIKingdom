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
    axes[1].plot(r, huber_r, label=f"Huber(delta={delta})", lw=2, ls=":")
    axes[1].set_title("Regression Losses"); axes[1].legend(); axes[1].set_ylim(0, 4)

    plt.tight_layout(); plt.savefig(OUTPUT / "loss_functions.png"); plt.close()
    print("\n  Saved loss_functions.png")

def demo_focal_loss():
    """Focal loss down-weights easy examples for class imbalance."""
    print("\n=== Focal Loss ===")
    def focal_loss(y, p, gamma=2.0, eps=1e-7):
        p = np.clip(p, eps, 1-eps)
        pt = np.where(y == 1, p, 1-p)
        return -np.mean(((1 - pt) ** gamma) * np.log(pt))

    p = np.linspace(0.01, 0.99, 100)
    y_ones = np.ones(100)  # all positive class
    bce_vals  = [-np.log(pp) for pp in p]
    focal_g1  = [focal_loss(np.array([1.]), np.array([pp]), gamma=1) for pp in p]
    focal_g2  = [focal_loss(np.array([1.]), np.array([pp]), gamma=2) for pp in p]
    focal_g5  = [focal_loss(np.array([1.]), np.array([pp]), gamma=5) for pp in p]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, bce_vals,  label="BCE (gamma=0)", lw=2)
    ax.plot(p, focal_g1, label="Focal gamma=1",  lw=2, linestyle="--")
    ax.plot(p, focal_g2, label="Focal gamma=2",  lw=2, linestyle="-.") 
    ax.plot(p, focal_g5, label="Focal gamma=5",  lw=2, linestyle=":")
    ax.set(xlabel="Predicted prob (correct class)", ylabel="Loss",
           title="Focal Loss vs BCE", ylim=(0, 5))
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "focal_loss.png"); plt.close()
    print("  Saved focal_loss.png")
    print(f"  At p=0.9: BCE={-np.log(0.9):.4f}  focal(g=2)={focal_loss(np.array([1.]),np.array([0.9]),2):.6f}")


def demo_loss_gradient_landscape():
    """Visualise how loss gradient changes with prediction for different losses."""
    print("\n=== Loss Gradient Landscapes ===")
    p = np.linspace(0.01, 0.99, 200)
    # BCE gradient: d/dp = -1/p for y=1
    bce_grad = -1.0 / p
    # Hinge gradient: -1 if 1-p > 0 (for y=1 mapping to margin=1-p)
    hinge_g = np.where(p < 1.0, -1.0, 0.0)
    # Huber gradient of MSE proxy: 2*(p-1) for y=1, clipped at delta=0.5
    delta = 0.5; r = p - 1.0
    huber_g = np.where(np.abs(r) <= delta, 2*r, -np.sign(r) * delta)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, np.clip(bce_grad, -20, 0), label="BCE grad (y=1)", lw=2)
    ax.plot(p, hinge_g,  label="Hinge grad", lw=2, linestyle="--")
    ax.plot(p, huber_g,  label="Huber grad", lw=2, linestyle=":")
    ax.set(xlabel="Predicted prob", ylabel="Gradient",
           title="Loss Gradient vs Prediction")
    ax.axhline(0, color="k", lw=0.5); ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "loss_gradients.png"); plt.close()
    print("  Saved loss_gradients.png")


if __name__ == "__main__":
    demo_losses()
    demo_bce_stability()
    demo_softmax_ce()
    demo_loss_curves()
    demo_focal_loss()
    demo_loss_gradient_landscape()
