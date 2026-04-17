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

if __name__ == "__main__":
    demo()
