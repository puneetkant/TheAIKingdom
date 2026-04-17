"""
Working Example 2: DL for Time Series — sliding window regression with MLP
===========================================================================
Converts time series to supervised learning problem and trains a MLP.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_series(n=500, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    return (np.sin(2*np.pi*t/30) * 3 + 0.5*t/n +
            np.sin(2*np.pi*t/7) + rng.normal(0, 0.2, n))

def series_to_supervised(ts, look_back=20, horizon=1):
    X, y = [], []
    for i in range(len(ts) - look_back - horizon + 1):
        X.append(ts[i:i+look_back])
        y.append(ts[i+look_back:i+look_back+horizon])
    return np.array(X), np.array(y).squeeze()

def demo():
    print("=== DL for Time Series: MLP Sliding Window ===")
    ts = make_series()
    X, y = series_to_supervised(ts, look_back=20, horizon=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False, test_size=0.2)

    scaler_x = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = scaler_x.transform(X_tr); X_te_s = scaler_x.transform(X_te)
    y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).ravel()

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=0)
    mlp.fit(X_tr_s, y_tr_s)

    y_pred_s = mlp.predict(X_te_s)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(y_te, y_pred)
    print(f"  Samples: {len(X)} | look_back=20 | Test MAE: {mae:.4f}")

    plt.figure(figsize=(10, 3))
    plt.plot(y_te[:100], label="True"); plt.plot(y_pred[:100], label="MLP pred")
    plt.legend(); plt.title("MLP Time Series Forecast (first 100 test steps)")
    plt.tight_layout(); plt.savefig(OUTPUT / "dl_timeseries.png"); plt.close()
    print("  Saved dl_timeseries.png")

if __name__ == "__main__":
    demo()
