"""
Working Example 2: Tabular RL — Q-Learning and SARSA on GridWorld
==================================================================
Compares on-policy SARSA vs off-policy Q-Learning.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

N = 5; N_STATES = N * N; N_ACTIONS = 4

def step(s, a):
    if s == N_STATES - 1: return s, 0
    r, c = divmod(s, N)
    if a==0: r=max(0,r-1)
    elif a==1: r=min(N-1,r+1)
    elif a==2: c=max(0,c-1)
    else: c=min(N-1,c+1)
    ns = r*N+c
    return ns, (1 if ns == N_STATES-1 else -0.01)

def eps_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(N_ACTIONS)
    return Q[s].argmax()

def q_learning(episodes=2000, alpha=0.1, gamma=0.9, eps=0.1):
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []
    for _ in range(episodes):
        s = 0; total = 0
        for __ in range(200):
            a = eps_greedy(Q, s, eps)
            ns, r = step(s, a)
            Q[s, a] += alpha * (r + gamma * Q[ns].max() - Q[s, a])
            s = ns; total += r
            if ns == N_STATES - 1: break
        returns.append(total)
    return Q, returns

def sarsa(episodes=2000, alpha=0.1, gamma=0.9, eps=0.1):
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns = []
    for _ in range(episodes):
        s = 0; a = eps_greedy(Q, s, eps); total = 0
        for __ in range(200):
            ns, r = step(s, a)
            na = eps_greedy(Q, ns, eps)
            Q[s, a] += alpha * (r + gamma * Q[ns, na] - Q[s, a])
            s, a = ns, na; total += r
            if ns == N_STATES - 1: break
        returns.append(total)
    return Q, returns

def smooth(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode="valid")

def demo():
    print("=== Q-Learning vs SARSA on GridWorld (5x5) ===")
    Qq, rq = q_learning()
    Qs, rs = sarsa()
    print(f"  Q-Learning final avg return (last 100): {np.mean(rq[-100:]):.3f}")
    print(f"  SARSA      final avg return (last 100): {np.mean(rs[-100:]):.3f}")

    plt.figure(figsize=(8, 4))
    plt.plot(smooth(rq), label="Q-Learning"); plt.plot(smooth(rs), label="SARSA")
    plt.xlabel("Episode"); plt.ylabel("Return (smoothed)"); plt.legend()
    plt.title("Q-Learning vs SARSA — GridWorld")
    plt.tight_layout(); plt.savefig(OUTPUT / "tabular_rl.png"); plt.close()
    print("  Saved tabular_rl.png")

def demo_double_q_learning(episodes=1000, alpha=0.1, gamma=0.9, eps=0.1):
    """Double Q-Learning to reduce maximisation bias."""
    print("\n=== Double Q-Learning ===")
    Q1 = np.zeros((N_STATES, N_ACTIONS))
    Q2 = np.zeros((N_STATES, N_ACTIONS))
    returns_dq = []
    for _ in range(episodes):
        s = 0; total = 0
        for __ in range(200):
            a = eps_greedy(Q1 + Q2, s, eps)
            ns, r = step(s, a)
            if np.random.rand() < 0.5:
                Q1[s, a] += alpha * (r + gamma * Q2[ns, Q1[ns].argmax()] - Q1[s, a])
            else:
                Q2[s, a] += alpha * (r + gamma * Q1[ns, Q2[ns].argmax()] - Q2[s, a])
            s = ns; total += r
            if ns == N_STATES - 1: break
        returns_dq.append(total)
    _, rq = q_learning(episodes, alpha, gamma, eps)
    print(f"  Q-Learning   avg return (last 100): {np.mean(rq[-100:]):.3f}")
    print(f"  Double-Q     avg return (last 100): {np.mean(returns_dq[-100:]):.3f}")


def demo_learning_rate_comparison():
    """Compare different alpha values for convergence speed."""
    print("\n=== Learning Rate Comparison ===")
    for alpha in [0.01, 0.1, 0.5, 0.9]:
        _, rr = q_learning(episodes=1000, alpha=alpha)
        print(f"  alpha={alpha:.2f}: final avg={np.mean(rr[-100:]):.3f}")


if __name__ == "__main__":
    demo()
    demo_double_q_learning()
    demo_learning_rate_comparison()
