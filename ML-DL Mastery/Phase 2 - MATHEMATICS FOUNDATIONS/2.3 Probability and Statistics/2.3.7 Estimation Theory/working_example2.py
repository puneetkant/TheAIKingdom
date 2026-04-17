"""
Working Example 2: Estimation Theory — MLE, MAP, Bias-Variance, Cramér-Rao
============================================================================
Maximum Likelihood Estimation for Gaussian/Bernoulli, MAP with Beta prior,
bias-variance decomposition, Fisher information, Cramér-Rao lower bound.

Run:  python working_example2.py
"""
import math
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_mle_gaussian():
    print("=== MLE for Gaussian parameters ===")
    np.random.seed(42)
    mu_true, sigma_true = 3.0, 1.5
    data = np.random.normal(mu_true, sigma_true, 500)
    # MLE
    mu_hat   = np.mean(data)
    sigma_hat = np.sqrt(np.mean((data - mu_hat)**2))   # biased MLE
    sigma_unb = np.std(data, ddof=1)                   # unbiased
    print(f"  True: μ={mu_true}, σ={sigma_true}")
    print(f"  MLE:  μ̂={mu_hat:.4f}, σ̂={sigma_hat:.4f}  (biased)")
    print(f"  Unbiased σ: {sigma_unb:.4f}")

    # Log-likelihood surface
    mus = np.linspace(2.0, 4.0, 100)
    sigs = np.linspace(0.5, 3.0, 100)
    MU, SIG = np.meshgrid(mus, sigs)
    LL = -500*np.log(SIG) - 0.5*np.sum([(d-MU)**2/SIG**2 for d in data], axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(MU, SIG, LL, 25, cmap="viridis")
    ax.scatter(mu_hat, sigma_hat, color="r", zorder=5, label="MLE")
    ax.scatter(mu_true, sigma_true, color="w", marker="*", s=100, zorder=5, label="true")
    plt.colorbar(cs); ax.legend(); ax.set_xlabel("μ"); ax.set_ylabel("σ")
    ax.set_title("Gaussian Log-Likelihood surface")
    fig.savefig(OUTPUT / "mle_surface.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: mle_surface.png")

def demo_mle_bernoulli():
    print("\n=== MLE for Bernoulli ===")
    np.random.seed(1)
    p_true = 0.35
    for n in [10, 50, 200, 1000]:
        data = np.random.binomial(1, p_true, n)
        p_hat = data.mean()
        se    = math.sqrt(p_hat * (1 - p_hat) / n)
        print(f"  n={n:>5}: p̂={p_hat:.4f}  95%CI=[{max(0,p_hat-1.96*se):.4f},{min(1,p_hat+1.96*se):.4f}]")

def demo_map_beta_prior():
    print("\n=== MAP with Beta Prior ===")
    # Conjugate: p|data ~ Beta(α+k, β+n-k) with prior Beta(α,β)
    alpha, beta = 2, 2          # prior (slightly regularising)
    n = 20; k = 4               # 4 successes in 20 trials
    # MLE
    p_mle = k / n
    # MAP = (alpha+k-1) / (alpha+beta+n-2)
    p_map = (alpha + k - 1) / (alpha + beta + n - 2)
    # Posterior mean
    p_post = (alpha + k) / (alpha + beta + n)
    print(f"  Observed: k={k}, n={n}")
    print(f"  MLE:           p̂={p_mle:.4f}")
    print(f"  MAP:           p̂={p_map:.4f}")
    print(f"  Posterior mean:p̂={p_post:.4f}")

def demo_bias_variance():
    print("\n=== Bias-Variance Decomposition ===")
    np.random.seed(3)
    # Generate data from y = sin(x) + noise
    x_train = np.linspace(0, 2*np.pi, 15)
    x_test  = np.linspace(0, 2*np.pi, 200)
    sigma_n = 0.3
    predictions = {k: [] for k in [1, 3, 9]}

    for _ in range(200):
        y = np.sin(x_train) + np.random.normal(0, sigma_n, len(x_train))
        for k in [1, 3, 9]:
            # Polynomial regression degree k via np.polyfit
            coef = np.polyfit(x_train, y, k)
            y_pred = np.polyval(coef, x_test)
            predictions[k].append(y_pred)

    y_true = np.sin(x_test)
    print(f"  {'Degree':>6}  {'Bias²':>8}  {'Variance':>9}  {'MSE':>8}")
    for k in [1, 3, 9]:
        preds = np.array(predictions[k])
        mean_pred = preds.mean(0)
        bias2 = np.mean((mean_pred - y_true)**2)
        var   = np.mean(preds.var(0))
        mse   = np.mean((preds - y_true)**2)
        print(f"  {k:>6}  {bias2:>8.4f}  {var:>9.4f}  {mse:>8.4f}")

if __name__ == "__main__":
    demo_mle_gaussian()
    demo_mle_bernoulli()
    demo_map_beta_prior()
    demo_bias_variance()
