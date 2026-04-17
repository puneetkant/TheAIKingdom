"""
Working Example 2: RL Fundamentals — MDP, Bellman equations, policy evaluation
================================================================================
Implements Value Iteration on a simple GridWorld MDP.

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

# 4x4 GridWorld: states 0-15, terminal states 0 and 15
N = 4
N_STATES = N * N
ACTIONS = ["up", "down", "left", "right"]
GAMMA = 0.9

def step(s, a):
    """Deterministic transition. Returns (next_state, reward)."""
    if s == 0 or s == N_STATES - 1:  # terminal
        return s, 0
    r, c = divmod(s, N)
    if a == 0: r = max(0, r-1)
    elif a == 1: r = min(N-1, r+1)
    elif a == 2: c = max(0, c-1)
    elif a == 3: c = min(N-1, c+1)
    ns = r * N + c
    return ns, -1  # -1 reward per step

def value_iteration(gamma=0.9, theta=1e-6, max_iter=1000):
    V = np.zeros(N_STATES)
    for i in range(max_iter):
        delta = 0
        for s in range(N_STATES):
            if s == 0 or s == N_STATES - 1:
                continue
            q_vals = []
            for a_idx in range(len(ACTIONS)):
                ns, r = step(s, a_idx)
                q_vals.append(r + gamma * V[ns])
            new_v = max(q_vals)
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            print(f"  Converged at iteration {i+1}")
            break
    return V

def extract_policy(V):
    pi = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if s == 0 or s == N_STATES - 1:
            continue
        q_vals = [step(s, a)[1] + GAMMA * V[step(s, a)[0]] for a in range(len(ACTIONS))]
        pi[s] = np.argmax(q_vals)
    return pi

def demo():
    print("=== Value Iteration on GridWorld ===")
    V = value_iteration()
    pi = extract_policy(V)
    arrows = ["↑", "↓", "←", "→"]
    print("\n  Value function:")
    print(V.reshape(N, N).round(2))
    print("\n  Greedy policy:")
    policy_grid = [arrows[pi[s]] if s not in (0, N_STATES-1) else ("G" if s==0 else "G")
                   for s in range(N_STATES)]
    for r in range(N):
        print("  " + " ".join(policy_grid[r*N:(r+1)*N]))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(V.reshape(N, N), cmap="Blues")
    for s in range(N_STATES):
        r, c = divmod(s, N)
        ax.text(c, r, f"{V[s]:.1f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax); ax.set_title("Value Function (GridWorld)")
    plt.tight_layout(); plt.savefig(OUTPUT / "rl_value_iteration.png"); plt.close()
    print("\n  Saved rl_value_iteration.png")

if __name__ == "__main__":
    demo()
