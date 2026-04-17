"""
Working Example 2: ML System Design — training pipeline, data validation, serving interface
============================================================================================
End-to-end skeleton: preprocess → train → evaluate → serve.

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

# ── Data validation ────────────────────────────────────────────────────────────
def validate_schema(X, y, min_samples=10, n_features=4):
    assert X.ndim == 2, "X must be 2D"
    assert len(X) == len(y), "X/y length mismatch"
    assert X.shape[1] == n_features, f"Expected {n_features} features"
    assert len(X) >= min_samples, "Too few samples"
    assert not np.isnan(X).any(), "NaN in X"
    return True

# ── Simple preprocessing pipeline ─────────────────────────────────────────────
class Pipeline:
    def __init__(self):
        self.mean_ = None; self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0); self.std_ = X.std(axis=0).clip(1e-8)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ── Logistic regression model ─────────────────────────────────────────────────
class LogisticModel:
    def __init__(self, lr=0.1, n_iter=100):
        self.lr = lr; self.n_iter = n_iter; self.W = None; self.losses = []

    def fit(self, X, y):
        N, F = X.shape; self.W = np.zeros(F + 1)
        Xb = np.column_stack([np.ones(N), X])
        for _ in range(self.n_iter):
            logits = Xb @ self.W; probs = 1/(1+np.exp(-logits.clip(-50,50)))
            loss = -np.mean(y*np.log(probs+1e-8) + (1-y)*np.log(1-probs+1e-8))
            self.losses.append(loss)
            grad = (Xb.T @ (probs - y)) / N
            self.W -= self.lr * grad

    def predict_proba(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return 1/(1+np.exp(-(Xb @ self.W).clip(-50,50)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# ── Serving interface ─────────────────────────────────────────────────────────
class ModelServer:
    def __init__(self, pipeline, model):
        self.pipeline = pipeline; self.model = model

    def predict(self, raw_input):
        """Single-sample or batch prediction."""
        X = np.atleast_2d(raw_input)
        X_t = self.pipeline.transform(X)
        return self.model.predict(X_t).tolist()

def demo():
    np.random.seed(42)
    N, F = 200, 4
    X = np.random.randn(N, F)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    validate_schema(X, y, n_features=F)
    split = int(0.8 * N)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    pipe = Pipeline()
    X_tr_n = pipe.fit_transform(X_tr)
    X_te_n = pipe.transform(X_te)

    model = LogisticModel(lr=0.3, n_iter=80)
    model.fit(X_tr_n, y_tr)

    preds = model.predict(X_te_n)
    acc = (preds == y_te).mean()
    print(f"=== ML System Design ===\n  Test accuracy: {acc:.3f}")

    server = ModelServer(pipe, model)
    sample = X_te[0:3]
    preds_served = server.predict(sample)
    print(f"  Server predictions: {preds_served}  (true: {y_te[:3].tolist()})")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(model.losses); ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); plt.tight_layout()
    plt.savefig(OUTPUT / "ml_system_design.png"); plt.close()
    print("  Saved ml_system_design.png")

if __name__ == "__main__":
    demo()
