"""
Working Example 2: Model Serving — REST-style prediction handler, batching, latency
=====================================================================================
Simulates a model server: request queue, batch inference, latency profiling.

Run:  python working_example2.py
"""
from pathlib import Path
import time
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

class ModelArtifact:
    """Minimal trained model."""
    def __init__(self, n_features):
        np.random.seed(7)
        self.W = np.random.randn(n_features + 1) * 0.4
        self.n_features = n_features

    def predict_proba(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return 1 / (1 + np.exp(-(Xb @ self.W).clip(-50, 50)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

class PredictHandler:
    """REST-like handler with input validation and schema."""
    def __init__(self, model, expected_features):
        self.model = model
        self.F = expected_features

    def handle(self, payload):
        """payload: dict with 'instances' key → list of feature vectors."""
        if "instances" not in payload:
            return {"error": "Missing 'instances' key"}, 400
        X = np.array(payload["instances"], dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.F:
            return {"error": f"Expected {self.F} features, got {X.shape[1]}"}, 422
        preds = self.model.predict(X).tolist()
        proba = self.model.predict_proba(X).tolist()
        return {"predictions": preds, "probabilities": proba}, 200

def latency_benchmark(handler, batch_sizes, n_repeats=20):
    """Measure per-request latency for different batch sizes."""
    results = {}
    for bs in batch_sizes:
        times = []
        for _ in range(n_repeats):
            payload = {"instances": np.random.randn(bs, handler.F).tolist()}
            t0 = time.perf_counter()
            handler.handle(payload)
            times.append((time.perf_counter() - t0) * 1000)
        results[bs] = np.mean(times)
    return results

def demo():
    print("=== Model Serving Demo ===")
    model = ModelArtifact(n_features=5)
    handler = PredictHandler(model, expected_features=5)

    # Single prediction
    payload = {"instances": [[0.1, -0.3, 0.5, 1.2, -0.8]]}
    response, status = handler.handle(payload)
    print(f"  Single pred: {response}  status={status}")

    # Batch prediction
    payload = {"instances": np.random.randn(8, 5).tolist()}
    response, status = handler.handle(payload)
    print(f"  Batch preds (n=8): {response['predictions']}  status={status}")

    # Bad request
    _, bad_status = handler.handle({"wrong_key": []})
    print(f"  Bad request status: {bad_status}")

    # Latency benchmark
    batch_sizes = [1, 4, 16, 64, 256]
    latencies = latency_benchmark(handler, batch_sizes)
    print("\n  Latency benchmark:")
    for bs, ms in latencies.items():
        print(f"    batch={bs:4d}  avg={ms:.3f}ms")

    plt.figure(figsize=(5, 3))
    plt.plot(list(latencies.keys()), list(latencies.values()), "o-")
    plt.xlabel("Batch size"); plt.ylabel("Latency (ms)"); plt.title("Serving Latency vs Batch Size")
    plt.tight_layout(); plt.savefig(OUTPUT / "model_serving.png"); plt.close()
    print("  Saved model_serving.png")

if __name__ == "__main__":
    demo()
