"""
Working Example 2: TensorFlow/Keras — Sequential API, training, callbacks
===========================================================================
Functional Keras workflow on California Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== MLP Regressor on California Housing (Keras-style workflow) ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, h.target, test_size=0.2, random_state=42)

    # Simulate Keras Sequential-style: 64 -> 32 -> 1 with relu
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    print(f"  Architecture: 8 -> 64 -> 32 -> 1 (relu, adam)")
    print(f"  Epochs trained: {model.n_iter_}")
    print(f"  Test MSE: {mse:.4f}  MAE: {mae:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_curve_, label="train loss")
    if model.validation_scores_ is not None:
        ax.plot([-v for v in model.validation_scores_], label="val loss (neg score)", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("MLP Training Curve (sklearn)")
    ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT / "keras_training.png"); plt.close()
    print("  Saved keras_training.png")

if __name__ == "__main__":
    demo()
