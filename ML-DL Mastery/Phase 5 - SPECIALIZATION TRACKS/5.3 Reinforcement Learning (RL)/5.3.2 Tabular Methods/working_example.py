"""
Working Example: Tabular Reinforcement Learning Methods
Covers Monte Carlo, TD(0), SARSA, Q-Learning, and Dynamic Programming.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_tabular_rl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- GridWorld environment (shared) -------------------------------------------
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size-1, size-1)
        self.trap = (size-2, 0)
        self.start = (0, 0)
        self.state = self.start
        self.n_s = size * size
        self.n_a = 4

    def _id(self, s): return s[0]*self.size + s[1]
    def _rc(self, i): return (i//self.size, i%self.size)

    def reset(self):
        self.state = self.start
        return self._id(self.state)

    def step(self, action):
        r, c = self.state
        dr = [-1,1,0,0]; dc = [0,0,-1,1]
        r = np.clip(r + dr[action], 0, self.size-1)
        c = np.clip(c + dc[action], 0, self.size-1)
        self.state = (r, c)
        sid  = self._id(self.state)
        done = self.state in (self.goal, self.trap)
        reward = 1.0 if self.state==self.goal else (-1.0 if self.state==self.trap else -0.01)
        return sid, reward, done

    def render_policy(self, Q):
        ARROWS = ["^","v","<-","->"]
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                s = (r, c)
                if s == self.goal: row += "  G  "
                elif s == self.trap: row += "  T  "
                else:
                    sid = self._id(s)
                    row += f"  {ARROWS[Q[sid].argmax()]}  "
            print(f"  {row}")


# -- 1. Dynamic Programming (Value Iteration) ----------------------------------
def value_iteration():
    print("=== Value Iteration (Dynamic Programming) ===")
    print("  Requires full model P(s'|s,a) — model-based")
    print("  V_{k+1}(s) = max_a Sigma_{s'} P(s'|s,a)[R + gammaV_k(s')]")
    print()
    env = GridWorld(4); gamma = 0.9; theta = 1e-6
    V   = np.zeros(env.n_s)
    Q   = np.zeros((env.n_s, env.n_a))

    n_iter = 0
    while True:
        delta = 0
        for sid in range(env.n_s):
            s = env._rc(sid)
            if s in (env.goal, env.trap): continue
            q_vals = np.zeros(env.n_a)
            for a in range(env.n_a):
                env.state = s
                sid2, r, _ = env.step(a)
                q_vals[a] = r + gamma * V[sid2]
            v_new = q_vals.max()
            delta = max(delta, abs(V[sid] - v_new))
            V[sid] = v_new; Q[sid] = q_vals
        n_iter += 1
        if delta < theta: break

    print(f"  Converged in {n_iter} iterations  (delta<{theta})")
    print(f"  V at start (0,0): {V[env._id(env.start)]:.4f}")
    print(f"  V at goal  (3,3): {V[env._id(env.goal)]:.4f}")
    print("\n  Optimal policy (^v<-->):")
    env.render_policy(Q)
    return Q


# -- 2. Monte Carlo ------------------------------------------------------------
def monte_carlo():
    print("\n=== Monte Carlo Methods ===")
    print("  Model-free; learn from complete episodes")
    print("  V(s) <- V(s) + alpha[G_t - V(s)]  (every-visit MC)")
    print()
    env = GridWorld(4); gamma = 0.9; alpha = 0.05; eps = 0.3
    Q   = np.zeros((env.n_s, env.n_a))
    n_episodes = 2000; returns = []

    rng = np.random.default_rng(0)
    for ep in range(n_episodes):
        s = env.reset()
        traj = []
        for _ in range(100):
            a = rng.integers(env.n_a) if rng.uniform() < eps else Q[s].argmax()
            s2, r, done = env.step(a)
            traj.append((s, a, r))
            s = s2
            if done: break
        # Compute returns and update Q
        G = 0
        for (st, at, rt) in reversed(traj):
            G = rt + gamma * G
            Q[st, at] += alpha * (G - Q[st, at])
        returns.append(G)

    avg10  = np.mean(returns[-100:])
    print(f"  Episodes: {n_episodes}  epsilon={eps}  alpha={alpha}")
    print(f"  Avg return (last 100): {avg10:.4f}")
    print(f"  Q[start, ->]: {Q[env._id(env.start)].round(4)}")
    print("\n  Learned policy:")
    env.render_policy(Q)


# -- 3. Temporal Difference — TD(0) --------------------------------------------
def td_learning():
    print("\n=== TD(0) — Temporal Difference Learning ===")
    print("  Online, model-free; update after each step")
    print("  V(s) <- V(s) + alpha[R + gammaV(s') - V(s)]")
    print("  TD target: R + gammaV(s')  (bootstraps from next value estimate)")
    print()
    env = GridWorld(4); gamma = 0.9; alpha = 0.1; eps = 0.3
    V   = np.zeros(env.n_s)
    rng = np.random.default_rng(1)

    for ep in range(2000):
        s = env.reset()
        for _ in range(100):
            a = rng.integers(env.n_a) if rng.uniform() < eps else 0
            s2, r, done = env.step(a)
            V[s] += alpha * (r + gamma * V[s2] - V[s])
            s = s2
            if done: break

    print(f"  V[start]:   {V[env._id(env.start)]:.4f}")
    print(f"  V[goal]:    {V[env._id(env.goal)]:.4f}")
    print(f"  V[trap]:    {V[env._id(env.trap)]:.4f}")
    print()
    print("  Comparison: MC vs TD")
    print("    MC:  unbiased, high variance, needs full episode")
    print("    TD:  biased (bootstrap), low variance, online (no full episode needed)")


# -- 4. SARSA (on-policy TD) ---------------------------------------------------
def sarsa():
    print("\n=== SARSA (On-Policy TD Control) ===")
    print("  Q(s,a) <- Q(s,a) + alpha[R + gammaQ(s',a') - Q(s,a)]")
    print("  a' is chosen under the CURRENT (epsilon-greedy) policy")
    print()
    env = GridWorld(4); gamma = 0.9; alpha = 0.1; eps = 0.2
    Q   = np.zeros((env.n_s, env.n_a))
    rng = np.random.default_rng(2)

    def choose(s):
        return rng.integers(env.n_a) if rng.uniform() < eps else Q[s].argmax()

    rewards = []
    for ep in range(3000):
        s = env.reset(); a = choose(s); total = 0
        for _ in range(100):
            s2, r, done = env.step(a)
            a2 = choose(s2)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s = s2; a = a2; total += r
            if done: break
        rewards.append(total)

    print(f"  SARSA avg return (last 200): {np.mean(rewards[-200:]):.4f}")
    print("\n  SARSA policy:")
    env.render_policy(Q)


# -- 5. Q-Learning (off-policy TD) --------------------------------------------
def q_learning():
    print("\n=== Q-Learning (Off-Policy TD Control) ===")
    print("  Q(s,a) <- Q(s,a) + alpha[R + gamma max_{a'} Q(s',a') - Q(s,a)]")
    print("  a' is always the GREEDY action (regardless of behaviour policy)")
    print()
    env = GridWorld(4); gamma = 0.9; alpha = 0.1; eps = 0.2
    Q   = np.zeros((env.n_s, env.n_a))
    rng = np.random.default_rng(3)

    rewards = []
    for ep in range(3000):
        s = env.reset(); total = 0
        for _ in range(100):
            a = rng.integers(env.n_a) if rng.uniform() < eps else Q[s].argmax()
            s2, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            s = s2; total += r
            if done: break
        rewards.append(total)

    print(f"  Q-Learning avg return (last 200): {np.mean(rewards[-200:]):.4f}")
    print("\n  Q-Learning policy:")
    env.render_policy(Q)
    print()
    print("  SARSA vs Q-Learning:")
    print("    SARSA:      on-policy; safer; learns value under epsilon-greedy")
    print("    Q-Learning: off-policy; learns optimal Q* regardless of policy")
    print("    Cliff-walk: SARSA prefers safe route; Q-Learning discovers optimal (risky)")

    # Convergence plots
    window = 100
    ql_smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(ql_smooth, label="Q-Learning"); ax.set_xlabel("Episode")
    ax.set_ylabel("Return"); ax.legend(); ax.set_title("Q-Learning Return")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "qlearning_returns.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Return plot: {path}")


if __name__ == "__main__":
    value_iteration()
    monte_carlo()
    td_learning()
    sarsa()
    q_learning()
