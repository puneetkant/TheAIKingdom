"""
Working Example 2: Ranking Metrics — NDCG, MAP, Hit Rate
=========================================================
Manual NDCG/MAP implementation, simulated ranking evaluation.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def dcg(relevances, k=None):
    """Discounted Cumulative Gain."""
    r = np.asarray(relevances[:k], dtype=float)
    if len(r) == 0:
        return 0.0
    return np.sum(r / np.log2(np.arange(2, len(r)+2)))

def ndcg(relevances, ideal_relevances, k=None):
    """Normalized DCG."""
    ideal = sorted(ideal_relevances, reverse=True)
    i = dcg(ideal, k)
    return dcg(relevances, k) / i if i > 0 else 0.0

def average_precision(relevances):
    """Average Precision at K."""
    hits, total = 0, 0
    ap = 0.0
    for i, r in enumerate(relevances, 1):
        if r:
            hits += 1
            ap += hits / i
    return ap / max(total, hits) if hits > 0 else 0.0

def demo_manual_metrics():
    print("=== Manual Ranking Metrics ===")
    # System A: good ranking
    a = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    # System B: bad ranking
    b = [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
    ideal = sorted(a, reverse=True)

    for name, ranks in [("System A", a), ("System B", b)]:
        print(f"\n  {name}: {ranks}")
        for k in [3, 5, 10]:
            nd = ndcg(ranks, ideal, k=k)
            ap = average_precision(ranks[:k])
            print(f"    NDCG@{k}={nd:.4f}  AP@{k}={ap:.4f}")

def demo_regression_as_ranking():
    print("\n=== Regression -> Ranking (Cal Housing) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Rank by predicted score; check if actual high-value items are ranked high
    n = len(y_test)
    pred_rank = np.argsort(-y_pred)
    actual_top20 = set(np.where(y_test >= np.percentile(y_test, 80))[0])
    for k in [20, 50, 100]:
        hit = len(actual_top20 & set(pred_rank[:k])) / len(actual_top20)
        print(f"  Recall of true top-20% in top-{k} predictions: {hit:.2%}")

if __name__ == "__main__":
    demo_manual_metrics()
    demo_regression_as_ranking()
