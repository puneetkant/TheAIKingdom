"""
Working Example 2: RL Environments and Tools — Gym-like API from scratch
=========================================================================
Implements a minimal gym-compatible environment + training loop.

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

class BanditEnv:
    """k-armed bandit environment with gym-like API."""
    def __init__(self, k=5, seed=42):
        rng = np.random.default_rng(seed)
        self.q_star = rng.normal(0, 1, k)   # true action values
        self.k = k
    def reset(self): return np.zeros(self.k)  # dummy obs
    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1.0)
        return np.zeros(self.k), reward, False, {"q_star": self.q_star}
    @property
    def observation_space_shape(self): return (self.k,)
    @property
    def action_space_n(self): return self.k

def ucb_agent(env, steps=1000, c=2.0):
    """UCB1 agent."""
    Q = np.zeros(env.k); N = np.zeros(env.k); total = 0; rewards = []
    env.reset()
    for t in range(1, steps+1):
        with np.errstate(divide="ignore", invalid="ignore"):
            ucb = Q + c * np.sqrt(np.log(t) / (N + 1e-9))
        a = ucb.argmax(); _, r, _, _ = env.step(a)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]; total += r
        rewards.append(total / t)
    return rewards

def eps_greedy_agent(env, steps=1000, eps=0.1):
    """Epsilon-greedy agent."""
    Q = np.zeros(env.k); N = np.zeros(env.k); total = 0; rewards = []
    env.reset()
    for t in range(1, steps+1):
        a = np.random.randint(env.k) if np.random.rand() < eps else Q.argmax()
        _, r, _, _ = env.step(a)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]; total += r
        rewards.append(total / t)
    return rewards

def demo():
    print("=== RL Environments — k-armed Bandit ===")
    env = BanditEnv(k=10)
    print(f"  True values q*: {env.q_star.round(2)}")
    print(f"  Best arm: {env.q_star.argmax()} (q* = {env.q_star.max():.2f})")

    r_ucb  = ucb_agent(env)
    r_eps  = eps_greedy_agent(env)

    plt.figure(figsize=(8, 4))
    plt.plot(r_ucb, label="UCB (c=2)"); plt.plot(r_eps, label="epsilon-greedy (epsilon=0.1)")
    plt.axhline(env.q_star.max(), ls="--", color="gray", label="Optimal")
    plt.xlabel("Step"); plt.ylabel("Avg reward"); plt.legend()
    plt.title("k-Armed Bandit: UCB vs epsilon-Greedy")
    plt.tight_layout(); plt.savefig(OUTPUT / "rl_environments.png"); plt.close()
    print(f"  UCB  final avg: {r_ucb[-1]:.3f}")
    print(f"  epsilon-gr final avg: {r_eps[-1]:.3f}")
    print("  Saved rl_environments.png")

def thompson_sampling_agent(env, steps=1000):
    """Thompson sampling via Beta distribution posterior."""
    alpha_ts = np.ones(env.k); beta_ts = np.ones(env.k)
    rewards = []; total = 0
    env.reset()
    for t in range(1, steps+1):
        samples = np.random.beta(alpha_ts, beta_ts)
        a = samples.argmax()
        _obs, r, _done, _info = env.step(a)
        r_norm = np.clip((r + 3) / 6, 0, 1)
        alpha_ts[a] += r_norm; beta_ts[a] += (1 - r_norm)
        total += r; rewards.append(total / t)
    return rewards


def demo_bandit_comparison():
    """Compare UCB, epsilon-greedy, and Thompson sampling."""
    print("\n=== Bandit Algorithm Comparison ===")
    env = BanditEnv(k=10, seed=7)
    r_ucb = ucb_agent(env)
    r_eps = eps_greedy_agent(env)
    r_ts  = thompson_sampling_agent(env)
    print(f"  UCB               final avg: {r_ucb[-1]:.3f}")
    print(f"  Epsilon-greedy    final avg: {r_eps[-1]:.3f}")
    print(f"  Thompson sampling final avg: {r_ts[-1]:.3f}")
    print(f"  Optimal (q*)              : {env.q_star.max():.3f}")


def demo_environment_api():
    """Demonstrate the gym-like reset/step/done lifecycle."""
    print("\n=== Environment API Lifecycle ===")
    env = BanditEnv(k=5)
    obs = env.reset()
    print(f"  obs_shape={env.observation_space_shape}  n_actions={env.action_space_n}")
    for t in range(3):
        action = np.random.randint(env.action_space_n)
        next_obs, reward, done, info = env.step(action)
        print(f"  step {t+1}: action={action}  reward={reward:.3f}  done={done}")


if __name__ == "__main__":
    demo()
    demo_bandit_comparison()
    demo_environment_api()
