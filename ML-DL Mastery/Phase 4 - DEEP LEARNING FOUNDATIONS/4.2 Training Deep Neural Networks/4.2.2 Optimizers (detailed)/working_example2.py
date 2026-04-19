"""
Working Example 2: Optimizers — SGD, Momentum, RMSProp, Adam (numpy)
======================================================================
Implement and compare 4 optimizers on a simple quadratic bowl.

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

# -- Numpy optimizer classes -------------------------------------------------
class SGD:
    def __init__(self, lr=0.01): self.lr = lr
    def update(self, p, g): return p - self.lr * g

class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, 0.
    def update(self, p, g):
        self.v = self.beta * self.v + (1-self.beta) * g
        return p - self.lr * self.v

class RMSProp:
    def __init__(self, lr=0.01, beta=0.99, eps=1e-8):
        self.lr, self.beta, self.eps, self.s = lr, beta, eps, 0.
    def update(self, p, g):
        self.s = self.beta * self.s + (1-self.beta) * g**2
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)

class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = 0., 0., 0
    def update(self, p, g):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*g
        self.v = self.b2*self.v + (1-self.b2)*g**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def demo_quadratic():
    """Minimize f(x) = x² from x0=10."""
    print("=== Optimizer comparison on f(x) = x² ===")
    x0 = 10.0
    opts = {"SGD": SGD(0.1), "Momentum": Momentum(0.1), "RMSProp": RMSProp(0.1), "Adam": Adam(0.1)}
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, opt in opts.items():
        x = x0; xs = [x]
        for _ in range(50):
            grad = 2 * x
            x = opt.update(x, grad)
            xs.append(x)
        ax.plot(xs, label=name)
        print(f"  {name:10s}: final x={xs[-1]:.6f}")

    ax.axhline(0, c="k", ls="--"); ax.set_xlabel("Iteration"); ax.set_ylabel("x value")
    ax.set_title("Optimizers on f(x)=x²"); ax.legend()
    plt.tight_layout(); plt.savefig(OUTPUT / "optimizers_quadratic.png"); plt.close()
    print("  Saved optimizers_quadratic.png")

def demo_nn_optimizers():
    """Compare on make_moons with a tiny numpy net."""
    print("\n=== Optimizer comparison on make_moons ===")
    X, y = make_moons(600, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-500,500)))
    relu    = lambda x: np.maximum(0,x)
    relu_d  = lambda x: (x>0).astype(float)

    results = {}
    for opt_name in ["SGD", "Momentum", "Adam"]:
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((2, 32))*np.sqrt(2/2); b1 = np.zeros(32)
        W2 = rng.standard_normal((32,1))*np.sqrt(2/32); b2 = np.zeros(1)
        opt_W1 = {"SGD": SGD(0.1), "Momentum": Momentum(0.05), "Adam": Adam(0.01)}[opt_name]
        losses = []
        for _ in range(200):
            z1=X@W1+b1; a1=relu(z1); z2=a1@W2+b2; a2=sigmoid(z2)
            y2=y.reshape(-1,1); n=len(y); p=np.clip(a2,1e-7,1-1e-7)
            l = -np.mean(y2*np.log(p)+(1-y2)*np.log(1-p)); losses.append(l)
            dz2=(a2-y2)/n; dW2=a1.T@dz2; db2=dz2.sum(0)
            dz1=(dz2@W2.T)*relu_d(z1); dW1=X.T@dz1; db1=dz1.sum(0)
            W1 = opt_W1.update(W1, dW1); b1 -= 0.05*db1
            W2 -= 0.05*dW2; b2 -= 0.05*db2
        results[opt_name] = losses
        print(f"  {opt_name:10s}: final loss={losses[-1]:.4f}")

if __name__ == "__main__":
    demo_quadratic()
    demo_nn_optimizers()
