"""
Working Example 2: Training Practices — gradient clipping, batch size, early stopping
=======================================================================================
Practical techniques for stable and efficient deep network training.

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

sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-500,500)))
relu    = lambda x: np.maximum(0, x)
relu_d  = lambda x: (x>0).astype(float)

def clip_gradients(grads, max_norm=1.0):
    """Global gradient norm clipping (list of arrays)."""
    total_norm = np.sqrt(sum((g**2).sum() for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        grads = [g * scale for g in grads]
    return grads, total_norm

def train(X_tr, y_tr, X_val, y_val, lr=0.1, epochs=300,
          batch_size=None, clip=None, patience=None, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2,64))*np.sqrt(2/2); b1=np.zeros(64)
    W2 = rng.standard_normal((64,1))*np.sqrt(2/64); b2=np.zeros(1)
    n = len(y_tr)
    bs = n if batch_size is None else batch_size

    best_val, wait = np.inf, 0
    tr_losses, val_losses, grad_norms = [], [], []

    for ep in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, bs):
            sl = idx[start:start+bs]
            Xb, yb = X_tr[sl], y_tr[sl]
            z1=Xb@W1+b1; a1=relu(z1); z2=a1@W2+b2; a2=sigmoid(z2)
            y2=yb.reshape(-1,1); p=np.clip(a2,1e-7,1-1e-7); nb=len(yb)
            dz2=(a2-y2)/nb; dW2=a1.T@dz2; db2=dz2.sum(0)
            dz1=(dz2@W2.T)*relu_d(z1); dW1=Xb.T@dz1; db1=dz1.sum(0)
            grads = [dW1, dW2]
            if clip:
                grads, gn = clip_gradients(grads, clip)
                grad_norms.append(gn)
            dW1, dW2 = grads
            W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2

        z1=X_tr@W1+b1; a1=relu(z1); a2=sigmoid(a1@W2+b2)
        p=np.clip(a2,1e-7,1-1e-7); y2=y_tr.reshape(-1,1)
        tr_losses.append(-np.mean(y2*np.log(p)+(1-y2)*np.log(1-p)))

        z1v=X_val@W1+b1; a1v=relu(z1v); a2v=sigmoid(a1v@W2+b2)
        pv=np.clip(a2v,1e-7,1-1e-7); yv=y_val.reshape(-1,1)
        vl = -np.mean(yv*np.log(pv)+(1-yv)*np.log(1-pv)); val_losses.append(vl)

        if patience:
            if vl < best_val: best_val=vl; wait=0
            else:
                wait+=1
                if wait >= patience:
                    print(f"    Early stop at epoch {ep+1}")
                    break

    return tr_losses, val_losses, grad_norms

def demo():
    print("=== Training Practices Demo ===")
    X,y=make_moons(800,noise=0.3,random_state=42)
    X=StandardScaler().fit_transform(X)
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.25,random_state=42)

    configs = [
        ("FullBatch",   dict(lr=0.05, batch_size=None, clip=None, patience=None)),
        ("Mini-batch32",dict(lr=0.05, batch_size=32, clip=None, patience=None)),
        ("Clip0.5",     dict(lr=0.1,  batch_size=32, clip=0.5,  patience=None)),
        ("EarlyStop",   dict(lr=0.05, batch_size=32, clip=None, patience=15)),
    ]
    fig, ax = plt.subplots(figsize=(9,5))
    for name, cfg in configs:
        tr_l, val_l, _ = train(X_tr, y_tr, X_val, y_val, **cfg)
        print(f"  {name:15s}: final_val={val_l[-1]:.4f}  epochs={len(val_l)}")
        ax.plot(val_l, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val BCE"); ax.set_title("Training Practices")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "training_practices.png"); plt.close()
    print("  Saved training_practices.png")

if __name__ == "__main__":
    demo()
