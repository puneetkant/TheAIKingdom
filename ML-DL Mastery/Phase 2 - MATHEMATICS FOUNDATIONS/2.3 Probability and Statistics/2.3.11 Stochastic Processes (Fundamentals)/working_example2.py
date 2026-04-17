"""
Working Example 2: Stochastic Processes — Markov Chains, Random Walks, Brownian Motion
========================================================================================
Discrete Markov chains, stationary distribution, 1D random walk, Brownian motion sim.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_markov_chain():
    print("=== Discrete Markov Chain ===")
    # Weather model: states 0=Sunny, 1=Cloudy, 2=Rainy
    np.random.seed(0)
    P = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.2, 0.3, 0.5]])

    # Stationary distribution via eigenvector
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1))
    stat = np.real(vecs[:, idx]); stat /= stat.sum()
    print(f"  Stationary distribution: Sunny={stat[0]:.4f}  Cloudy={stat[1]:.4f}  Rainy={stat[2]:.4f}")

    # Verify: P^100 converges to stationary
    Pk = np.linalg.matrix_power(P, 100)
    print(f"  P^100 row 0: {Pk[0].round(4)} (should match stationary)")

    # Simulate chain
    n_steps = 2000; state = 0; counts = [0, 0, 0]
    for _ in range(n_steps):
        state = np.random.choice(3, p=P[state])
        counts[state] += 1
    emp = np.array(counts) / n_steps
    print(f"  Empirical:   Sunny={emp[0]:.4f}  Cloudy={emp[1]:.4f}  Rainy={emp[2]:.4f}")

def demo_random_walk():
    print("\n=== 1D Random Walk ===")
    np.random.seed(1)
    n = 1000; steps = np.random.choice([-1, 1], size=(20, n))
    paths = np.cumsum(steps, axis=1)

    std_expected = n**0.5
    std_observed = paths[:, -1].std()
    print(f"  Theoretical std after {n} steps: {std_expected:.2f}")
    print(f"  Observed std (20 paths):          {std_observed:.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for p in paths[:5]: ax.plot(p, lw=0.6, alpha=0.7)
    ax.fill_between(range(n), -np.sqrt(np.arange(n)), np.sqrt(np.arange(n)),
                    alpha=0.15, color="grey", label="±√t envelope")
    ax.legend(); ax.set_title("Random walks with ±√t envelope")
    fig.savefig(OUTPUT / "random_walk.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: random_walk.png")

def demo_brownian_motion():
    print("\n=== Brownian Motion (Wiener Process) ===")
    np.random.seed(2)
    T, n = 1.0, 1000
    dt = T / n; t = np.linspace(0, T, n+1)
    n_paths = 5

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for _ in range(n_paths):
        dW = np.random.normal(0, dt**0.5, n)
        W  = np.concatenate([[0], np.cumsum(dW)])
        axes[0].plot(t, W, lw=0.7, alpha=0.7)
    axes[0].fill_between(t, -t**0.5, t**0.5, alpha=0.15, color="grey", label="±√t")
    axes[0].legend(); axes[0].set_title("Brownian motion paths")

    # Geometric Brownian Motion (GBM): S(t) = S0 * exp((mu-0.5*sig^2)*t + sig*W)
    S0, mu, sig = 100, 0.1, 0.2
    for _ in range(n_paths):
        dW = np.random.normal(0, dt**0.5, n)
        W  = np.cumsum(dW)
        S  = S0 * np.exp((mu - 0.5*sig**2)*t[1:] + sig*W)
        axes[1].plot(t[1:], S, lw=0.7, alpha=0.7)
    axes[1].set_title(f"Geometric BM  μ={mu}  σ={sig}")

    fig.savefig(OUTPUT / "brownian_motion.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: brownian_motion.png")

if __name__ == "__main__":
    demo_markov_chain()
    demo_random_walk()
    demo_brownian_motion()
