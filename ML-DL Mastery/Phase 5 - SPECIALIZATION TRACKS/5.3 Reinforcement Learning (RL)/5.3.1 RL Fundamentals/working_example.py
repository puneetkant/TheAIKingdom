"""
Working Example: Reinforcement Learning Fundamentals
Covers MDP formulation, Bellman equations, policy/value functions,
and exploration vs exploitation.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rl_fundamentals")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. MDP formulation ────────────────────────────────────────────────────────
def mdp_overview():
    print("=== Markov Decision Process (MDP) ===")
    print("  A MDP is defined by (S, A, P, R, γ):")
    print("    S  — finite state space")
    print("    A  — finite action space")
    print("    P(s'|s,a) — transition probability")
    print("    R(s,a,s') — reward function")
    print("    γ ∈ [0,1) — discount factor")
    print()
    print("  Markov property: future depends only on current state")
    print("    P[S_{t+1} | S_t, A_t, ..., S_0, A_0] = P[S_{t+1} | S_t, A_t]")
    print()
    print("  Goal: find policy π(a|s) maximising discounted return:")
    print("    G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...")
    print()
    print("  Episodic vs Continuing tasks:")
    print("    Episodic: terminal state exists (games, robotics tasks)")
    print("    Continuing: infinite horizon (stock trading, control systems)")


# ── 2. GridWorld environment ──────────────────────────────────────────────────
class GridWorld:
    """Simple 4×4 GridWorld with a goal and a trap."""
    def __init__(self, size=4):
        self.size   = size
        self.goal   = (size-1, size-1)
        self.trap   = (size-2, 0)
        self.start  = (0, 0)
        self.state  = self.start
        self.n_states  = size * size
        self.n_actions = 4   # up, down, left, right

    def _to_id(self, s): return s[0] * self.size + s[1]
    def _from_id(self, i): return (i // self.size, i % self.size)

    def reset(self):
        self.state = self.start
        return self._to_id(self.state)

    def step(self, action):
        r, c = self.state
        if   action == 0: r -= 1   # up
        elif action == 1: r += 1   # down
        elif action == 2: c -= 1   # left
        elif action == 3: c += 1   # right
        r = np.clip(r, 0, self.size-1)
        c = np.clip(c, 0, self.size-1)
        self.state = (r, c)
        sid  = self._to_id(self.state)
        done = self.state == self.goal or self.state == self.trap
        if   self.state == self.goal: reward = 1.0
        elif self.state == self.trap: reward = -1.0
        else:                         reward = -0.01  # small step penalty
        return sid, reward, done

    def display(self, V=None):
        symbols = {self.goal: 'G', self.trap: 'T', self.start: 'S'}
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                s = (r, c)
                if s in symbols:
                    row += f" {symbols[s]}  "
                elif V is not None:
                    sid = self._to_id(s)
                    row += f"{V[sid]:+.2f}"
                else:
                    row += " .  "
            print(f"  {row}")


def gridworld_overview():
    print("\n=== GridWorld Environment ===")
    env = GridWorld(4)
    env.display()
    print()
    print("  States: 16  Actions: 4 (up/down/left/right)")
    print("  G=goal(+1), T=trap(-1), other=-0.01 per step")
    print()
    # A random rollout
    sid = env.reset()
    trajectory = [sid]
    for _ in range(10):
        a = np.random.randint(4)
        sid, r, done = env.step(a)
        trajectory.append(sid)
        if done: break
    print(f"  Random rollout states: {trajectory}")


# ── 3. Bellman equations ──────────────────────────────────────────────────────
def bellman_equations():
    print("\n=== Bellman Equations ===")
    print("  State-value function:")
    print("    V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]")
    print()
    print("  Action-value function:")
    print("    Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s',a')]")
    print()
    print("  Bellman optimality (greedy policy):")
    print("    V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]")
    print("    Q*(s,a) = Σ_{s'} P(s'|s,a) [R + γ max_{a'} Q*(s',a')]")
    print()

    # Small policy evaluation on GridWorld
    print("  Policy Evaluation (random policy, γ=0.9):")
    env = GridWorld(4); γ = 0.9
    V   = np.zeros(env.n_states)

    def rand_policy(s): return np.ones(env.n_actions) / env.n_actions

    for _ in range(100):
        V_new = np.zeros_like(V)
        for sid in range(env.n_states):
            s = env._from_id(sid)
            if s == env.goal or s == env.trap:
                V_new[sid] = 1.0 if s == env.goal else -1.0
                continue
            v = 0
            for a in range(env.n_actions):
                env.state = s
                sid2, r, _ = env.step(a)
                v += rand_policy(sid)[a] * (r + γ * V[sid2])
            V_new[sid] = v
        V = V_new

    print(f"\n  V (random policy):")
    env.display(V.reshape(4,4).flatten())


# ── 4. Policy types ───────────────────────────────────────────────────────────
def policy_types():
    print("\n=== Policy Types ===")
    print("  Deterministic: π(s) = a")
    print("  Stochastic:    π(a|s) = P(A=a | S=s)")
    print()
    types = [
        ("Greedy",       "Always take the best known action; no exploration"),
        ("ε-greedy",     "With prob ε choose random; else greedy"),
        ("Boltzmann",    "Softmax of Q-values; temperature τ controls exploration"),
        ("UCB",          "Upper Confidence Bound; optimistic face uncertainty"),
        ("Thompson",     "Bayesian sampling from posterior; bandit problems"),
        ("Entropy-reg.", "Add entropy bonus to reward; soft actor-critic"),
    ]
    for name, desc in types:
        print(f"  {name:<14} {desc}")
    print()

    # ε-greedy demo
    Q = np.array([1.0, 2.5, 1.8, 0.5])
    eps = 0.1
    rng = np.random.default_rng(0)
    actions = []
    for _ in range(1000):
        if rng.uniform() < eps:
            a = rng.integers(4)
        else:
            a = Q.argmax()
        actions.append(a)
    counts = np.bincount(actions, minlength=4)
    print(f"  ε-greedy (ε={eps}) over 1000 steps — action counts: {counts}")
    print(f"  Greedy action (Q={Q}): {Q.argmax()}")
    print(f"  Selected {counts[Q.argmax()]/10:.1f}% of time (expected ≈{(1-eps)*100 + eps*25:.1f}%)")


# ── 5. Exploration strategies ─────────────────────────────────────────────────
def exploration_strategies():
    print("\n=== Exploration vs Exploitation ===")
    print("  Dilemma: exploit current knowledge vs explore for better rewards")
    print()
    print("  K-armed bandit (stateless RL):")
    K   = 5
    rng = np.random.default_rng(1)
    true_means = rng.uniform(-1, 2, K)
    print(f"  True reward means: {np.round(true_means, 3)}")
    print(f"  Optimal arm: {true_means.argmax()} (mean={true_means.max():.3f})")
    print()

    def run_bandit(strategy, steps=500, eps=0.1, c=2.0):
        Q_est = np.zeros(K); N = np.zeros(K); t = 0; total_r = 0
        rewards = []
        for step in range(1, steps+1):
            if strategy == "greedy":
                a = Q_est.argmax()
            elif strategy == "eps-greedy":
                a = rng.integers(K) if rng.uniform() < eps else Q_est.argmax()
            elif strategy == "ucb":
                if step <= K:
                    a = step - 1
                else:
                    a = (Q_est + c * np.sqrt(np.log(step) / (N + 1e-9))).argmax()
            r = rng.normal(true_means[a], 1.0)
            N[a] += 1; Q_est[a] += (r - Q_est[a]) / N[a]
            total_r += r; rewards.append(r)
        return total_r / steps, Q_est

    print(f"  {'Strategy':<14} {'Avg Reward':>12}")
    for strat in ["greedy", "eps-greedy", "ucb"]:
        avg, q_est = run_bandit(strat)
        print(f"  {strat:<14} {avg:>10.4f}   Q_est={np.round(q_est,2)}")

    print()
    print("  Regret = Σ_t [μ* - μ_{a_t}]  (difference from optimal)")
    print("  ε-greedy: O(√T) regret with decaying ε")
    print("  UCB: O(log T) regret — theoretically optimal for stationary bandits")


if __name__ == "__main__":
    mdp_overview()
    gridworld_overview()
    bellman_equations()
    policy_types()
    exploration_strategies()
