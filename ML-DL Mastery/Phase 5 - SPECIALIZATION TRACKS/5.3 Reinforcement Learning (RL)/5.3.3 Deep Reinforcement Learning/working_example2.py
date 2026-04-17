"""
Working Example 2: Deep RL — DQN key components from scratch
=============================================================
Implements replay buffer, target network update, and Q-network training
on a simple CartPole-like MDP (numpy-only, no gym required).

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

# ---------- Simple environment: 2-state continous MDP ----------
class PoleEnv:
    """Simplified 1D pole balancing — state: (angle, vel), action: 0 or 1."""
    def reset(self):
        self.s = np.random.uniform(-0.05, 0.05, 2)
        return self.s.copy()
    def step(self, a):
        force = 1.0 if a == 1 else -1.0
        self.s[1] += force * 0.1 - self.s[0] * 0.05
        self.s[0] += self.s[1] * 0.1
        done = abs(self.s[0]) > 1.0
        return self.s.copy(), -1.0 if done else 0.1, done

# ---------- Replay buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = []; self.cap = capacity
    def push(self, transition):
        if len(self.buf) >= self.cap: self.buf.pop(0)
        self.buf.append(transition)
    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)

# ---------- Minimal linear Q-network (numpy) ----------
class LinearQ:
    def __init__(self, in_dim=2, n_actions=2, lr=0.01):
        self.W = np.zeros((n_actions, in_dim)); self.b = np.zeros(n_actions)
        self.lr = lr
    def predict(self, s):
        return self.W @ s + self.b
    def update(self, s, a, target):
        q = self.predict(s); error = target - q[a]
        self.W[a] += self.lr * error * s; self.b[a] += self.lr * error
    def copy_from(self, other):
        self.W = other.W.copy(); self.b = other.b.copy()

def train(episodes=500, batch_size=32, gamma=0.95, target_update=50, eps_start=1.0, eps_end=0.05):
    env = PoleEnv()
    online = LinearQ(); target = LinearQ()
    buf = ReplayBuffer()
    returns = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - ep / episodes)
        s = env.reset(); total = 0
        for _ in range(200):
            if np.random.rand() < eps or len(buf) < batch_size:
                a = np.random.randint(2)
            else:
                a = online.predict(s).argmax()
            ns, r, done = env.step(a)
            buf.push((s, a, r, ns, done)); s = ns; total += r
            if len(buf) >= batch_size:
                batch = buf.sample(batch_size)
                for bs, ba, br, bns, bdone in batch:
                    t_val = br if bdone else br + gamma * target.predict(bns).max()
                    online.update(bs, ba, t_val)
            if done: break
        if ep % target_update == 0: target.copy_from(online)
        returns.append(total)
    return returns

def demo():
    print("=== DQN Components Demo (numpy linear Q) ===")
    returns = train()
    smooth = np.convolve(returns, np.ones(20)/20, mode="valid")
    print(f"  Final avg return (last 50): {np.mean(returns[-50:]):.2f}")
    plt.figure(figsize=(7, 3))
    plt.plot(returns, alpha=0.3, label="raw"); plt.plot(smooth, label="smoothed")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.legend()
    plt.title("DQN Linear Q — PoleEnv")
    plt.tight_layout(); plt.savefig(OUTPUT / "deep_rl_dqn.png"); plt.close()
    print("  Saved deep_rl_dqn.png")

if __name__ == "__main__":
    demo()
