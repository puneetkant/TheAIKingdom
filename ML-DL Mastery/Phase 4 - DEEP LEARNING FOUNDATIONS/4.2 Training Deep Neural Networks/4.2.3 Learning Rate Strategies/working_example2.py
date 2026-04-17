"""
Working Example 2: Learning Rate Strategies — decay, warmup, cosine, ReduceLROnPlateau
========================================================================================
Simulate LR schedules and their effect on loss curves.

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
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def lr_constant(ep, warmup=0, base=0.1):     return base
def lr_step(ep, base=0.1, drop=0.5, every=50):
    return base * (drop ** (ep // every))
def lr_cosine(ep, base=0.1, total=200):
    return base * 0.5 * (1 + np.cos(np.pi * ep / total))
def lr_warmup_cosine(ep, base=0.1, warmup=20, total=200):
    if ep < warmup: return base * ep / warmup
    return base * 0.5 * (1 + np.cos(np.pi * (ep - warmup) / (total - warmup)))

def demo_schedules():
    print("=== Learning Rate Schedule Comparison ===")
    epochs = 200
    eps = np.arange(epochs)
    schedules = {
        "Constant 0.1": [lr_constant(e) for e in eps],
        "Step (×0.5 /50)": [lr_step(e) for e in eps],
        "Cosine": [lr_cosine(e, total=epochs) for e in eps],
        "Warmup+Cosine": [lr_warmup_cosine(e, total=epochs) for e in eps],
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, lrs in schedules.items():
        ax.plot(eps, lrs, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("LR Schedules")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "lr_schedules.png"); plt.close()
    for name, lrs in schedules.items():
        print(f"  {name:22s}: start={lrs[0]:.4f} final={lrs[-1]:.4f}")
    print("  Saved lr_schedules.png")

def demo_schedule_effect():
    print("\n=== LR Schedule Effect on Training ===")
    X, y = make_moons(600, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-500,500)))
    relu    = lambda x: np.maximum(0,x)
    relu_d  = lambda x: (x>0).astype(float)

    epochs = 200
    schedule_fns = {
        "Constant":      lambda e: 0.05,
        "Step":          lambda e: 0.1 * (0.5 ** (e // 50)),
        "Warmup+Cosine": lambda e: lr_warmup_cosine(e, 0.1, 20, epochs),
    }
    results = {}
    for sched_name, lr_fn in schedule_fns.items():
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((2, 32))*np.sqrt(2/2); b1 = np.zeros(32)
        W2 = rng.standard_normal((32,1))*np.sqrt(2/32); b2 = np.zeros(1)
        losses = []
        for ep in range(epochs):
            lr = lr_fn(ep)
            z1=X@W1+b1; a1=relu(z1); z2=a1@W2+b2; a2=sigmoid(z2)
            y2=y.reshape(-1,1); p=np.clip(a2,1e-7,1-1e-7); n=len(y)
            losses.append(-np.mean(y2*np.log(p)+(1-y2)*np.log(1-p)))
            dz2=(a2-y2)/n; dW2=a1.T@dz2; db2=dz2.sum(0)
            dz1=(dz2@W2.T)*relu_d(z1); dW1=X.T@dz1; db1=dz1.sum(0)
            W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1
        results[sched_name] = losses
        print(f"  {sched_name:15s}: final loss={losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(9,4))
    for name, ls in results.items(): ax.plot(ls, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Schedule Effect on Training")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "lr_schedule_effect.png"); plt.close()
    print("  Saved lr_schedule_effect.png")

if __name__ == "__main__":
    demo_schedules()
    demo_schedule_effect()
