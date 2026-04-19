"""
Working Example: Stochastic Processes (Fundamentals)
Covers random walks, Markov chains, stationary distributions,
Poisson processes, Brownian motion, and hidden Markov models.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_stoch")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Random walk ------------------------------------------------------------
def random_walk():
    print("=== Random Walk ===")
    rng  = np.random.default_rng(0)
    n    = 1000
    p    = 0.5   # symmetric
    steps = rng.choice([-1, 1], size=n, p=[1-p, p])
    pos   = np.concatenate([[0], np.cumsum(steps)])

    print(f"  n={n} steps, p(right)={p}")
    print(f"  Final position: {pos[-1]}")
    print(f"  Expected |X_n| ~= sqrtn = {np.sqrt(n):.2f}")
    print(f"  Var(X_n) = n·p·(1-p)·4 = {n:.0f}")

    # Multiple walks
    M = 500
    finals = np.array([np.sum(rng.choice([-1,1], n)) for _ in range(M)])
    print(f"  {M} walks: mean final pos={finals.mean():.4f}  std={finals.std():.4f}  (theory std=sqrtn={np.sqrt(n):.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10,4))
    for _ in range(10):
        s = np.concatenate([[0], np.cumsum(rng.choice([-1,1], n))])
        ax.plot(s, alpha=0.5, lw=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set(xlabel="Step", ylabel="Position", title="10 Random Walks")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "random_walk.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Saved: {path}")


# -- 2. Discrete-time Markov chain --------------------------------------------
def markov_chain():
    print("\n=== Markov Chain ===")
    # Weather: 0=Sunny, 1=Cloudy, 2=Rainy
    P = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.2, 0.3, 0.5]])
    states = ["Sunny", "Cloudy", "Rainy"]

    # Verify stochastic matrix
    print(f"  Transition matrix P (rows sum to 1: {np.allclose(P.sum(axis=1), 1)}):")
    for i, row in enumerate(P):
        print(f"    {states[i]:<7}: {row}")

    # Chapman-Kolmogorov: n-step probabilities P^n
    for n in [1, 5, 10, 50]:
        Pn = np.linalg.matrix_power(P, n)
        print(f"\n  P^{n} (from Sunny):")
        print(f"    {np.round(Pn[0], 6)}")

    # Stationary distribution pi = piP, Sigmapi=1
    # Solve (Pᵀ - I)pi = 0 with normalisation
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1))
    pi  = vecs[:, idx].real
    pi /= pi.sum()
    print(f"\n  Stationary distribution pi:")
    for s, p in zip(states, pi):
        print(f"    P({s}) = {p:.4f}")
    print(f"  Verify piP = pi: {np.allclose(pi @ P, pi)}")


# -- 3. Absorbing Markov chain -------------------------------------------------
def absorbing_markov():
    print("\n=== Absorbing Markov Chain (Gambler's Ruin) ===")
    # States: 0 (broke, absorbing), 1, 2, 3, 4 (win, absorbing)
    # P(go up) = p=0.5, P(go down) = 1-p
    p   = 0.5
    n   = 5   # target winnings
    # Ruin probability starting from state k: r_k = (q/p)^k - 1 / (q/p)^n - 1 if p!=q
    # For p=0.5 (symmetric): r_k = 1 - k/n
    print(f"  Fair game (p={p}), target={n}")
    for k in range(n+1):
        ruin_prob = 1 - k/n
        print(f"    Start at ${k}: ruin prob = {ruin_prob:.4f}  win prob = {k/n:.4f}")


# -- 4. Poisson process --------------------------------------------------------
def poisson_process():
    print("\n=== Poisson Process (lambda events per unit time) ===")
    rng = np.random.default_rng(1)
    lam = 3.0   # events per unit time

    # Simulate via exponential inter-arrival times
    T   = 10.0  # total time
    arrivals = []
    t = 0
    while t < T:
        t += rng.exponential(1/lam)
        if t < T:
            arrivals.append(t)
    N = len(arrivals)

    print(f"  lambda={lam}/unit  T={T}  arrivals={N}  expected={lam*T}")

    # Count in intervals
    intervals = [(0,1),(1,2),(2,3),(3,4),(4,5)]
    from scipy import stats
    print(f"\n  Counts per interval (Poisson(lambda={lam}) distributed):")
    for a, b in intervals:
        cnt = sum(1 for x in arrivals if a <= x < b)
        print(f"    [{a},{b}): {cnt}  (Poisson pmf={stats.poisson.pmf(cnt, lam):.4f})")

    # Inter-arrival times ~ Exp(lambda)
    if len(arrivals) > 1:
        iat = np.diff(arrivals)
        print(f"\n  Inter-arrival times: mean={iat.mean():.4f}  (theory 1/lambda={1/lam:.4f})")


# -- 5. Brownian motion --------------------------------------------------------
def brownian_motion():
    print("\n=== Brownian Motion (Wiener Process) ===")
    rng  = np.random.default_rng(2)
    T    = 1.0
    n    = 1000
    dt   = T / n
    t    = np.linspace(0, T, n+1)

    # W_t = Sigma sqrtdt · Z_i  (Z_i ~ N(0,1))
    dW   = np.sqrt(dt) * rng.standard_normal(n)
    W    = np.concatenate([[0], np.cumsum(dW)])

    print(f"  T={T}  n={n} steps  dt={dt:.4f}")
    print(f"  E[W_T] ~= {W[-1]:.4f}  (theory 0)")
    print(f"  E[W_T²] ~= {np.mean([np.sum(np.sqrt(dt)*rng.standard_normal(n))**2 for _ in range(1000)]):.4f}  (theory T={T})")

    # Properties
    print(f"\n  Properties:")
    print(f"    W_0 = 0: {W[0]==0}")
    print(f"    Increments W_t - W_s ~ N(0, t-s)")
    # Check increment distribution
    mid = n//2
    inc = W[n] - W[mid]
    expected_var = T/2   # n/2 · dt = T/2
    print(f"    W_1 - W_0.5 = {inc:.4f}  (N(0,{expected_var:.2f}) distributed)")

    # Plot
    fig, ax = plt.subplots(figsize=(10,4))
    for _ in range(15):
        dW_ = np.sqrt(dt) * rng.standard_normal(n)
        W_  = np.concatenate([[0], np.cumsum(dW_)])
        ax.plot(t, W_, alpha=0.5, lw=0.7)
    ax.axhline(0, color='black', lw=1)
    ax.fill_between(t, -2*np.sqrt(t), 2*np.sqrt(t), alpha=0.1, color='red', label='+/-2sqrtt')
    ax.set(xlabel="Time t", ylabel="W_t", title="Brownian Motion (15 paths)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "brownian_motion.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Saved: {path}")


# -- 6. Hidden Markov Model (Viterbi decoding) ---------------------------------
def hidden_markov_model():
    print("\n=== Hidden Markov Model (Viterbi Algorithm) ===")
    # States: 0=Fair die, 1=Loaded die
    # Observations: 1–6
    A = np.array([[0.95, 0.05],
                  [0.10, 0.90]])   # transition
    B = np.array([[1/6]*6,
                  [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])  # emission (loaded favors 6)
    pi = np.array([0.5, 0.5])   # initial state prob

    rng = np.random.default_rng(3)
    T   = 20

    # Generate sequence from fair die (state 0)
    obs = rng.choice(6, T, p=B[0])
    print(f"  Observations (0-indexed die rolls): {obs+1}")

    # Viterbi
    n_states = 2
    delta  = np.zeros((T, n_states))
    psi    = np.zeros((T, n_states), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for s in range(n_states):
            trans = delta[t-1] * A[:, s]
            psi[t, s]   = trans.argmax()
            delta[t, s] = trans.max() * B[s, obs[t]]

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    state_names = ["Fair", "Loaded"]
    print(f"  Viterbi decoded states: {[state_names[s] for s in path]}")
    print(f"  (True: all Fair since we sampled from p=1/6 uniform)")


if __name__ == "__main__":
    random_walk()
    markov_chain()
    absorbing_markov()
    poisson_process()
    brownian_motion()
    hidden_markov_model()
