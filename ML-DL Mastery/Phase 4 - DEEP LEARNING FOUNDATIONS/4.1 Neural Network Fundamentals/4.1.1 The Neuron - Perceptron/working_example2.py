"""
Working Example 2: The Neuron & Perceptron — numpy implementation + sklearn comparison
========================================================================================
McCulloch-Pitts neuron, Perceptron learning rule, sklearn Perceptron, linearly separable test.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification, make_circles
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Perceptron as SkPerceptron
    from sklearn.metrics import accuracy_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# -- Numpy Perceptron ---------------------------------------------------------
class NumpyPerceptron:
    def __init__(self, lr=0.1, n_iter=100):
        self.lr, self.n_iter = lr, n_iter

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_ = []
        for _ in range(self.n_iter):
            err = 0
            for xi, yi in zip(X, y):
                pred = self.predict_one(xi)
                delta = self.lr * (yi - pred)
                self.w += delta * xi
                self.b += delta
                err += int(delta != 0)
            self.errors_.append(err)
        return self

    def predict_one(self, x):
        return 1 if np.dot(self.w, x) + self.b >= 0 else -1

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

def demo_linearly_separable():
    print("=== Perceptron on Linearly Separable Data ===")
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=42)
    y_pm = np.where(y == 0, -1, 1)
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    p = NumpyPerceptron(lr=0.1, n_iter=50)
    p.fit(X_s, y_pm)
    acc = (p.predict(X_s) == y_pm).mean()
    print(f"  Numpy Perceptron accuracy: {acc:.4f}  (epochs: {len(p.errors_)}, last errors: {p.errors_[-1]})")

    # sklearn Perceptron
    sk = SkPerceptron(max_iter=100, random_state=42)
    sk.fit(X_s, y)
    print(f"  sklearn Perceptron accuracy: {accuracy_score(y, sk.predict(X_s)):.4f}")

def demo_not_linearly_separable():
    print("\n=== Perceptron on Non-Linearly Separable Data (circles) ===")
    X, y = make_circles(n_samples=300, noise=0.1, random_state=42)
    y_pm = np.where(y == 0, -1, 1)
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    p = NumpyPerceptron(lr=0.01, n_iter=200)
    p.fit(X_s, y_pm)
    acc = (p.predict(X_s) == y_pm).mean()
    print(f"  Perceptron on circles: {acc:.4f}  (won't converge — not linearly separable)")

def demo_decision_boundary():
    """Visualise the decision boundary of a trained perceptron."""
    print("\n=== Decision Boundary Visualisation ===")
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=7)
    y_pm = np.where(y == 0, -1, 1)
    sc = StandardScaler(); X_s = sc.fit_transform(X)
    p = NumpyPerceptron(lr=0.05, n_iter=100); p.fit(X_s, y_pm)

    x_min, x_max = X_s[:, 0].min() - 0.5, X_s[:, 0].max() + 0.5
    y_min, y_max = X_s[:, 1].min() - 0.5, X_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120),
                          np.linspace(y_min, y_max, 120))
    Z = p.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    axes[0].scatter(X_s[:, 0], X_s[:, 1], c=y_pm, cmap="RdBu", s=15)
    axes[0].set_title("Perceptron Decision Boundary")
    axes[1].plot(p.errors_); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Misclassifications")
    axes[1].set_title("Learning Curve")
    plt.tight_layout()
    fig.savefig(OUTPUT / "perceptron_boundary.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: perceptron_boundary.png")


def demo_mcculloch_pitts():
    """McCulloch-Pitts neuron: threshold logic unit implementing AND, OR, NOT."""
    print("\n=== McCulloch-Pitts Logic Gates ===")
    def mp_neuron(inputs, weights, threshold):
        return int(np.dot(inputs, weights) >= threshold)

    # AND gate
    print("  AND gate (w=[1,1], thresh=2):")
    for x1, x2 in [(0,0),(0,1),(1,0),(1,1)]:
        out = mp_neuron([x1, x2], [1, 1], 2)
        print(f"    {x1} AND {x2} = {out}")

    # OR gate
    print("  OR gate (w=[1,1], thresh=1):")
    for x1, x2 in [(0,0),(0,1),(1,0),(1,1)]:
        out = mp_neuron([x1, x2], [1, 1], 1)
        print(f"    {x1} OR  {x2} = {out}")


if __name__ == "__main__":
    demo_linearly_separable()
    demo_not_linearly_separable()
    demo_decision_boundary()
    demo_mcculloch_pitts()
