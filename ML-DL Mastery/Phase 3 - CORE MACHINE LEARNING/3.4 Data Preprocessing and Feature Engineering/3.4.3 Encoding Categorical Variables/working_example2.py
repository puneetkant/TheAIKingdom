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
    y = np.where(zones=="urban", 500000, np.where(zones=="suburban", 300000, 150000))
    y += np.random.normal(0, 20000, 1000)

    # Train split for target encoding (to avoid leakage)
    train_mask = np.arange(1000) < 800
    zone_means = {z: y[train_mask & (zones==z)].mean() for z in ["urban","suburban","rural"]}
    print("  Zone target means:", {k: f"{v:.0f}" for k,v in zone_means.items()})

if __name__ == "__main__":
    demo_encoding_manual()
    demo_column_transformer()
    demo_target_encoding()
