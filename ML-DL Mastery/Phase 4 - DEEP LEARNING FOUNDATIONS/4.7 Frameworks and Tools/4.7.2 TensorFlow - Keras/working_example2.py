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
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise SystemExit("pip install tensorflow scikit-learn matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== TensorFlow / Keras MLP ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, h.target, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(8,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    cb = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    hist = model.fit(X_tr, y_tr, validation_split=0.1, epochs=100, batch_size=64,
                     callbacks=cb, verbose=0)

    val_mse = model.evaluate(X_te, y_te, verbose=0)
    print(f"  Test MSE: {val_mse[0]:.4f}  MAE: {val_mse[1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("Keras Training")
    ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT / "keras_training.png"); plt.close()
    print("  Saved keras_training.png")

if __name__ == "__main__":
    demo()
