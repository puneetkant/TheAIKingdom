"""
Working Example 2: Regularization for Deep Learning — L2, Dropout, Early Stopping
===================================================================================
Comparing L2 weight decay, dropout, and early stopping on a numpy MLP.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
relu    = lambda x: np.maximum(0, x)
relu_d  = lambda x: (x > 0).astype(float)

def train_net(X_tr, y_tr, X_val, y_val, n_hidden=64, lr=0.05, epochs=400,
              l2=0.0, dropout_rate=0.0, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((X_tr.shape[1], n_hidden)) * np.sqrt(2 / X_tr.shape[1])
    b1 = np.zeros(n_hidden)
    W2 = rng.standard_normal((n_hidden, 1)) * np.sqrt(2 / n_hidden)
    b2 = np.zeros(1)

    train_losses, val_losses = [], []
    for ep in range(epochs):
        # Forward (training mode with dropout)
        z1 = X_tr @ W1 + b1; a1 = relu(z1)
        if dropout_rate > 0:
            mask = (rng.random(a1.shape) > dropout_rate) / (1 - dropout_rate)
            a1 = a1 * mask
        else:
            mask = np.ones_like(a1)
        z2 = a1 @ W2 + b2; a2 = sigmoid(z2)

        y2 = y_tr.reshape(-1, 1); p = np.clip(a2, 1e-7, 1-1e-7)
        n = len(y_tr)
        loss = -np.mean(y2*np.log(p) + (1-y2)*np.log(1-p)) + 0.5*l2*(W1**2).sum()/n
        train_losses.append(loss)

        # Backward
        dz2 = (a2 - y2) / n
        dW2 = a1.T @ dz2 + l2 * W2 / n
        db2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * relu_d(z1) * mask
        dW1 = X_tr.T @ dz1 + l2 * W1 / n
        db1 = dz1.sum(0)
        W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1; b1 -= lr*db1

        # Val loss (no dropout)
        z1v = X_val @ W1 + b1; a1v = relu(z1v)
        z2v = a1v @ W2 + b2; a2v = sigmoid(z2v)
        yv = y_val.reshape(-1, 1); pv = np.clip(a2v, 1e-7, 1-1e-7)
        val_losses.append(-np.mean(yv*np.log(pv) + (1-yv)*np.log(1-pv)))

    return train_losses, val_losses

def demo():
    print("=== Regularization Comparison ===")
    X, y = make_moons(1000, noise=0.3, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    configs = [
        ("No reg",    dict(l2=0.0, dropout_rate=0.0)),
        ("L2=0.01",   dict(l2=0.01, dropout_rate=0.0)),
        ("Dropout50", dict(l2=0.0, dropout_rate=0.5)),
        ("Both",      dict(l2=0.01, dropout_rate=0.3)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, cfg in configs:
        tr_l, val_l = train_net(X_tr, y_tr, X_val, y_val, **cfg)
        gap = np.array(val_l[-20:]).mean() - np.array(tr_l[-20:]).mean()
        print(f"  {name:12s}: final_train={tr_l[-1]:.4f}  final_val={val_l[-1]:.4f}  gap={gap:.4f}")
        axes[0].plot(tr_l,  label=f"{name} tr")
        axes[1].plot(val_l, label=f"{name} val")

    for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE"); ax.set_title(title)
        ax.legend(fontsize=7); ax.set_ylim(0, 1.5)
    plt.tight_layout(); plt.savefig(OUTPUT / "regularization_comparison.png"); plt.close()
    print("  Saved regularization_comparison.png")

def demo_weight_norm_effect():
    """Show how L2 regularisation shrinks weight norms over training."""
    print("\n=== Weight Norm vs L2 Strength ===")
    X, y = make_moons(600, noise=0.3, random_state=1)
    X = StandardScaler().fit_transform(X)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
    l2_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    print(f"  {'L2':>8s} {'|W1|_F':>10s} {'Val loss':>10s}")
    for l2 in l2_values:
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((X_tr.shape[1], 64)) * np.sqrt(2/X_tr.shape[1])
        b1 = np.zeros(64)
        W2 = rng.standard_normal((64, 1)) * np.sqrt(2/64)
        b2 = np.zeros(1)
        for ep in range(200):
            lr = 0.05
            z1=X_tr@W1+b1; a1=np.maximum(0,z1); z2=a1@W2+b2; a2=1/(1+np.exp(-np.clip(z2,-500,500)))
            y2=y_tr.reshape(-1,1); p=np.clip(a2,1e-7,1-1e-7); n=len(y_tr)
            dz2=(a2-y2)/n; dW2=a1.T@dz2+l2*W2/n; db2=dz2.sum(0)
            dz1=(dz2@W2.T)*((z1>0).astype(float)); dW1=X_tr.T@dz1+l2*W1/n; db1=dz1.sum(0)
            W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1
        z1v=X_val@W1+b1; a1v=np.maximum(0,z1v); z2v=a1v@W2+b2; a2v=1/(1+np.exp(-np.clip(z2v,-500,500)))
        yv=y_val.reshape(-1,1); pv=np.clip(a2v,1e-7,1-1e-7)
        val_l = float(-np.mean(yv*np.log(pv)+(1-yv)*np.log(1-pv)))
        w1_norm = float(np.linalg.norm(W1, 'fro'))
        print(f"  {l2:>8.3f} {w1_norm:>10.3f} {val_l:>10.4f}")


def demo_early_stopping_effect():
    """Demonstrate early stopping: track best val loss and stop when it worsens."""
    print("\n=== Early Stopping ===")
    X, y = make_moons(600, noise=0.3, random_state=2)
    X = StandardScaler().fit_transform(X)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=2)
    # Overfitting net (no regularisation)
    tr_l, val_l = train_net(X_tr, y_tr, X_val, y_val, n_hidden=128, lr=0.1, epochs=400,
                             l2=0.0, dropout_rate=0.0, seed=0)
    # Early stopping: patience=20
    patience = 20; best_ep = 0; best_val = float("inf")
    for ep, vl in enumerate(val_l):
        if vl < best_val:
            best_val = vl; best_ep = ep
        elif ep - best_ep > patience:
            break
    print(f"  Best epoch: {best_ep}  best val loss: {best_val:.4f}")
    print(f"  Final val loss (no stopping): {val_l[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr_l,  color="steelblue", label="Train", lw=1)
    ax.plot(val_l, color="tomato",    label="Val",   lw=1)
    ax.axvline(best_ep, color="green", linestyle="--", label=f"Early stop ep={best_ep}")
    ax.set(xlabel="Epoch", ylabel="BCE", title="Early Stopping Demo")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "early_stopping.png"); plt.close()
    print("  Saved early_stopping.png")


if __name__ == "__main__":
    demo()
    demo_weight_norm_effect()
    demo_early_stopping_effect()
