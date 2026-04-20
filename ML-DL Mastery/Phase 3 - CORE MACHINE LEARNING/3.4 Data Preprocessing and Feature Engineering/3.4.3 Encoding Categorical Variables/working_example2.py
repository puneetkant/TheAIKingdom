"""
Working Example 2: Encoding Categorical Variables — OHE, Ordinal, Target Encoding
===================================================================================
One-hot encoding, ordinal encoding, target encoding with cross-validation.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import mean_squared_error
    import warnings; warnings.filterwarnings("ignore")
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def demo_encoding_manual():
    print("=== Manual Encoding Demo ===")
    colours = ["red", "green", "blue", "red", "green"]
    categories = sorted(set(colours))
    # OHE manual
    ohe = {c: i for i, c in enumerate(categories)}
    for col in colours:
        vec = [0] * len(categories)
        vec[ohe[col]] = 1
        print(f"  {col:6s} -> {vec}")

    # Ordinal (size)
    sizes = ["small", "medium", "large", "small", "large"]
    order = {"small": 0, "medium": 1, "large": 2}
    print("\n  Ordinal encoding (small<medium<large):")
    for s in sizes:
        print(f"    {s:7s} -> {order[s]}")

def demo_column_transformer():
    print("\n=== ColumnTransformer with Mixed Types ===")
    # Synthetic housing data
    np.random.seed(42)
    n = 500
    X = {
        "rooms":      np.random.randint(1, 8, n).astype(float),
        "age":        np.random.randint(1, 50, n).astype(float),
        "zone":       np.random.choice(["urban", "suburban", "rural"], n),
        "condition":  np.random.choice(["poor", "fair", "good", "excellent"], n),
    }
    import pandas as pd
    df = pd.DataFrame(X)
    y = (df["rooms"] * 50000 + df["age"] * -1000 +
         df["zone"].map({"urban": 200000, "suburban": 100000, "rural": 0}) +
         df["condition"].map({"poor": 0, "fair": 20000, "good": 40000, "excellent": 80000}) +
         np.random.normal(0, 10000, n))

    num_feats = ["rooms", "age"]
    cat_feats = ["zone", "condition"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_feats),
    ])
    pipe = Pipeline([("prep", preprocessor), ("model", Ridge(1.0))])
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
    print(f"  OHE + StandardScaler + Ridge: RMSE={rmse:.2f}")

def demo_target_encoding():
    print("\n=== Target Encoding (manual mean encoding) ===")
    np.random.seed(42)
    zones = np.random.choice(["urban", "suburban", "rural"], 1000)
    y = np.where(zones=="urban", 500000.0, np.where(zones=="suburban", 300000.0, 150000.0))
    y += np.random.normal(0, 20000, 1000)

    # Train split for target encoding (to avoid leakage)
    train_mask = np.arange(1000) < 800
    zone_means = {z: y[train_mask & (zones==z)].mean() for z in ["urban","suburban","rural"]}
    print("  Zone target means:", {k: f"{v:.0f}" for k,v in zone_means.items()})

def demo_cardinality_effect():
    """Show how high-cardinality categories affect OHE dimensionality."""
    print("\n=== High-Cardinality Effect ===")
    np.random.seed(7)
    cardinalities = [2, 5, 10, 50, 100, 500]
    n_rows = 1000
    for k in cardinalities:
        ohe_dim = k  # OHE adds k columns
        # Target encoding always adds 1 column
        print(f"  cardinality={k:4d}: OHE cols={ohe_dim:4d}  TargetEnc cols=1  "
              f"OHE memory ratio={ohe_dim:.0f}x")


def demo_embedding_size_heuristic():
    """Show the common sqrt(cardinality) or min(50, k//2) embedding size rules."""
    print("\n=== Embedding Size Heuristics ===")
    cardinalities = [2, 5, 10, 20, 50, 100, 500, 1000]
    print(f"  {'Cardinality':>14s} {'OHE':>6s} {'sqrt rule':>10s} {'k//2 rule':>10s} {'min50 rule':>11s}")
    for k in cardinalities:
        sqrt_rule  = max(2, int(round(k ** 0.5)))
        half_rule  = max(2, k // 2)
        min50_rule = min(50, max(2, k // 2))
        print(f"  {k:>14d} {k:>6d} {sqrt_rule:>10d} {half_rule:>10d} {min50_rule:>11d}")


if __name__ == "__main__":
    demo_encoding_manual()
    demo_column_transformer()
    demo_target_encoding()
    demo_cardinality_effect()
    demo_embedding_size_heuristic()
