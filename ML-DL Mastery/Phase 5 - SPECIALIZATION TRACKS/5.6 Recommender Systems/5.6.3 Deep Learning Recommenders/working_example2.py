"""
Working Example 2: Deep Learning Recommenders — Neural Collaborative Filtering (NCF)
======================================================================================
Implements embedding-based NCF from scratch with numpy SGD.

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

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

class SimpleNCF:
    """Minimal NCF: user emb + item emb -> MLP -> binary prediction."""
    def __init__(self, n_users, n_items, emb_dim=8, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.U = rng.normal(0, 0.1, (n_users, emb_dim))
        self.I = rng.normal(0, 0.1, (n_items, emb_dim))
        self.W = rng.normal(0, 0.1, (1, emb_dim * 2))
        self.b = np.zeros(1); self.lr = lr

    def forward(self, u, i):
        h = np.concatenate([self.U[u], self.I[i]])
        logit = self.W @ h + self.b
        return sigmoid(logit).item()

    def train_step(self, u, i, y):
        pred = self.forward(u, i)
        err = pred - y
        h = np.concatenate([self.U[u], self.I[i]])
        grad_w = err * h; grad_b = err
        grad_h = err * self.W.ravel()
        self.W -= self.lr * grad_w; self.b -= self.lr * grad_b
        self.U[u] -= self.lr * grad_h[:self.U.shape[1]]
        self.I[i] -= self.lr * grad_h[self.I.shape[1]:]
        return float(err**2)

def demo():
    print("=== Neural Collaborative Filtering (NCF) ===")
    n_users, n_items = 50, 30
    rng = np.random.default_rng(0)
    # Positive pairs (user liked item)
    pos = [(rng.integers(n_users), rng.integers(n_items)) for _ in range(300)]
    neg = [(rng.integers(n_users), rng.integers(n_items)) for _ in range(300)]
    data = [(u, i, 1) for u, i in pos] + [(u, i, 0) for u, i in neg]

    model = SimpleNCF(n_users, n_items)
    losses = []
    for epoch in range(30):
        np.random.shuffle(data)
        ep_loss = np.mean([model.train_step(u, i, y) for u, i, y in data])
        losses.append(ep_loss)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss {ep_loss:.4f}")

    # Evaluation: random pos vs 9 neg ranking
    hits = 0; n_eval = 50
    for _ in range(n_eval):
        u = rng.integers(n_users); pos_i = rng.integers(n_items)
        neg_items = rng.integers(n_items, size=9)
        items = [pos_i] + list(neg_items)
        scores = [model.forward(u, it) for it in items]
        if np.argmax(scores) == 0: hits += 1
    print(f"  HR@10: {hits/n_eval:.2f}")

    plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title("NCF Training Loss"); plt.tight_layout()
    plt.savefig(OUTPUT / "ncf_training.png"); plt.close()
    print("  Saved ncf_training.png")

if __name__ == "__main__":
    demo()
