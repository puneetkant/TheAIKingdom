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

if __name__ == "__main__":
    demo_linearly_separable()
    demo_not_linearly_separable()
