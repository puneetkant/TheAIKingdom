"""
Working Example: Deep Reinforcement Learning
Covers DQN, Policy Gradient (REINFORCE), Actor-Critic (A2C),
PPO concepts, and practical training tips.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_deep_rl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def relu(z): return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# -- Simple cartpole-like environment -----------------------------------------
class CartPoleSimple:
    """Simplified 1D balance task — state: (pos, vel) pole angle."""
    def __init__(self):
        self.rng   = np.random.default_rng(0)
        self.state = None
        self.max_steps = 200
        self.steps = 0

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        """Simplified dynamics — pole stays up if action opposes velocity."""
        a   = 1 if action == 1 else -1
        self.state[0] += self.state[1] * 0.02         # pos += vel
        self.state[1] += a * 0.5 - self.state[0]*0.1  # vel += force
        self.state[2] += self.state[3] * 0.02         # angle
        self.state[3] += self.state[1]*0.1 - self.state[2]*0.3
        self.state = np.clip(self.state, -3, 3)
        self.steps += 1
        done   = abs(self.state[2]) > 0.5 or self.steps >= self.max_steps
        reward = 1.0 if not done or self.steps >= self.max_steps else 0.0
        return self.state.copy(), reward, done


# -- Minimal neural network ----------------------------------------------------
class MLP:
    def __init__(self, in_dim, hidden, out_dim, rng=None):
        rng = rng or np.random.default_rng(0)
        s = np.sqrt(2 / in_dim)
        self.W1 = rng.standard_normal((in_dim, hidden)) * s
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, out_dim)) * np.sqrt(2/hidden)
        self.b2 = np.zeros(out_dim)
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        h = relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def apply_grads(self, grads, lr):
        for p, g in zip(self.params, grads):
            p -= lr * np.clip(g, -1, 1)


# -- 1. DQN --------------------------------------------------------------------
def dqn_theory():
    print("=== Deep Q-Network (DQN) ===")
    print("  Mnih et al. (2015) — learns to play Atari from pixels")
    print()
    print("  Core idea: approximate Q(s,a) with neural network theta")
    print("  Loss: L(theta) = E[(r + gamma max_{a'} Q(s',a';theta^-) - Q(s,a;theta))²]")
    print()
    print("  Two key innovations:")
    print("  1. Experience Replay:")
    print("     Store transitions (s,a,r,s') in buffer D")
    print("     Sample random mini-batches -> break temporal correlation")
    print()
    print("  2. Target Network theta^-:")
    print("     Separate network for computing targets")
    print("     Update theta^- <- theta every C steps -> stable targets")
    print()
    print("  DQN variants:")
    variants = [
        ("Double DQN",     "Decouple action selection from Q evaluation; less overestimate"),
        ("Dueling DQN",    "V(s) + A(s,a) — separate advantage stream"),
        ("PER",            "Prioritised Experience Replay; sample high-TD-error"),
        ("Rainbow",        "Combines 6 improvements: DDQN+Dueling+PER+Noisy+Multi-step+C51"),
        ("C51 / Distrib.", "Distributional RL; model full return distribution"),
        ("Noisy Net",      "Noisy linear layers for implicit exploration"),
    ]
    for v, d in variants:
        print(f"  {v:<16} {d}")


class ReplayBuffer:
    def __init__(self, capacity, rng):
        self.buffer = []
        self.capacity = capacity
        self.rng = rng

    def add(self, s, a, r, s2, done):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s.copy(), a, r, s2.copy(), done))

    def sample(self, batch_size):
        idx = self.rng.integers(len(self.buffer), size=batch_size)
        S, A, R, S2, D = zip(*[self.buffer[i] for i in idx])
        return (np.array(S), np.array(A), np.array(R, dtype=float),
                np.array(S2), np.array(D, dtype=float))


def dqn_demo():
    print("\n--- DQN Demo (CartPoleSimple, numpy-only) ---")
    env = CartPoleSimple(); rng = np.random.default_rng(42)
    n_actions = 2; state_dim = 4; gamma = 0.99; lr = 0.005
    eps = 1.0; eps_min = 0.05; eps_decay = 0.995
    buf = ReplayBuffer(2000, rng); bs = 32

    q_net    = MLP(state_dim, 32, n_actions, rng)
    q_target = MLP(state_dim, 32, n_actions, rng)
    q_target.W1[:] = q_net.W1; q_target.b1[:] = q_net.b1
    q_target.W2[:] = q_net.W2; q_target.b2[:] = q_net.b2

    episode_returns = []
    for ep in range(200):
        s = env.reset(); total_r = 0
        for _ in range(200):
            if rng.uniform() < eps:
                a = rng.integers(n_actions)
            else:
                a = q_net.forward(s).argmax()
            s2, r, done = env.step(a)
            buf.add(s, a, r, s2, done); s = s2; total_r += r
            if done: break

        if len(buf.buffer) >= bs:
            S, A, R, S2, D = buf.sample(bs)
            Q_next = q_target.forward(S2).max(1)
            targets = R + gamma * Q_next * (1 - D)
            Q_pred  = q_net.forward(S)
            # Gradient for chosen actions
            dL = Q_pred.copy()
            for i, ai in enumerate(A):
                dL[i, ai] = Q_pred[i, ai] - targets[i]
            dL /= bs
            # Backprop (2-layer)
            h1 = relu(S @ q_net.W1 + q_net.b1)
            dW2 = h1.T @ dL; db2 = dL.sum(0)
            dh1 = (dL @ q_net.W2.T) * (h1 > 0)
            dW1 = S.T @ dh1; db1 = dh1.sum(0)
            q_net.apply_grads([dW1, db1, dW2, db2], lr)

        if ep % 10 == 0:   # update target
            q_target.W1[:] = q_net.W1; q_target.b1[:] = q_net.b1
            q_target.W2[:] = q_net.W2; q_target.b2[:] = q_net.b2

        eps = max(eps_min, eps * eps_decay)
        episode_returns.append(total_r)

    avg = np.mean(episode_returns[-50:])
    print(f"  Episodes: 200  Avg return (last 50): {avg:.1f}  (max possible: 200)")

    window = 20
    smooth = np.convolve(episode_returns, np.ones(window)/window, mode="valid")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(episode_returns, alpha=0.3, label="raw")
    ax.plot(smooth, label=f"{window}-ep avg")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return"); ax.legend()
    ax.set_title("DQN on CartPoleSimple")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dqn_returns.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 2. Policy Gradient / REINFORCE -------------------------------------------
def policy_gradient():
    print("\n=== Policy Gradient (REINFORCE) ===")
    print("  Williams (1992); directly optimise E[G_t]")
    print()
    print("  Policy gradient theorem:")
    print("    ∇J(theta) = E[G_t · ∇ log pi_theta(a_t|s_t)]")
    print()
    print("  REINFORCE update:")
    print("    theta <- theta + alpha · G_t · ∇ log pi_theta(a_t|s_t)")
    print()
    print("  Variance reduction: subtract baseline b(s)")
    print("    theta <- theta + alpha · (G_t - b(s_t)) · ∇ log pi")
    print("    Advantage A_t = G_t - V(s_t)  (Actor-Critic)")
    print()

    env = CartPoleSimple(); rng = np.random.default_rng(5)
    pi_net = MLP(4, 32, 2, rng); lr = 0.02; gamma = 0.99
    ep_returns = []

    for ep in range(300):
        s = env.reset(); log_probs = []; rewards = []; h_cache = []
        for _ in range(200):
            logits = pi_net.forward(s)
            probs  = softmax(logits)
            a      = rng.choice(2, p=probs)
            log_probs.append(np.log(probs[a] + 1e-9))
            h_cache.append((s.copy(), a, probs.copy()))
            s, r, done = env.step(a)
            rewards.append(r)
            if done: break

        # Compute returns
        G = 0; returns_ep = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns_ep.insert(0, G)
        returns_ep = np.array(returns_ep)
        baseline = returns_ep.mean()
        advs = returns_ep - baseline

        # Gradient accumulation
        dW1 = np.zeros_like(pi_net.W1); db1 = np.zeros_like(pi_net.b1)
        dW2 = np.zeros_like(pi_net.W2); db2 = np.zeros_like(pi_net.b2)
        for (si, ai, pi), adv in zip(h_cache, advs):
            h1   = relu(si @ pi_net.W1 + pi_net.b1)
            dlog = -pi.copy(); dlog[ai] += 1   # ∇ log pi
            g    = -adv * dlog  # gradient direction
            dW2 += np.outer(h1, g)
            db2 += g
            dh1  = (g @ pi_net.W2.T) * (h1 > 0)
            dW1 += np.outer(si, dh1)
            db1 += dh1

        n = len(h_cache)
        pi_net.apply_grads([dW1/n, db1/n, dW2/n, db2/n], lr)
        ep_returns.append(sum(rewards))

    avg = np.mean(ep_returns[-50:])
    print(f"  REINFORCE trained for 300 episodes  Avg (last 50): {avg:.1f}")


# -- 3. Actor-Critic / A2C overview -------------------------------------------
def actor_critic():
    print("\n=== Actor-Critic Methods ===")
    print("  Actor: policy pi_theta(a|s)  <- optimised by policy gradient")
    print("  Critic: value V_phi(s)    <- bootstraps advantage estimates")
    print()
    print("  A2C update:")
    print("    delta_t = R_t + gammaV(s_{t+1}) - V(s_t)   (TD error = advantage estimate)")
    print("    theta <- theta + alpha · delta_t · ∇ log pi_theta(a_t|s_t)   (actor)")
    print("    phi <- phi + beta · delta_t · ∇V_phi(s_t)              (critic)")
    print()
    print("  A3C (Async A3C): multiple workers update global network asynchronously")
    print("  A2C: synchronous version; simpler implementation")
    print()
    print("  Key algorithms in this family:")
    algs = [
        ("A2C / A3C", "Advantage Actor-Critic; on-policy; entropy bonus"),
        ("PPO",       "Proximal Policy Optimisation; clip ratio; most widely used"),
        ("TRPO",      "Trust Region; KL constraint; harder to implement"),
        ("SAC",       "Soft AC; off-policy; maximum entropy; continuous actions"),
        ("TD3",       "Twin Delayed DDPG; addresses overestimation in continuous RL"),
        ("DDPG",      "Deterministic PG + replay buffer; continuous action spaces"),
    ]
    for a, d in algs:
        print(f"  {a:<12} {d}")


# -- 4. PPO --------------------------------------------------------------------
def ppo_overview():
    print("\n=== PPO (Proximal Policy Optimisation) ===")
    print("  Schulman et al. (2017) — widely used; stable, simple")
    print()
    print("  Clip objective:")
    print("    r_t(theta) = pi_theta(a_t|s_t) / pi_{theta_old}(a_t|s_t)")
    print("    L^CLIP = E[min(r_t · A_t, clip(r_t, 1-epsilon, 1+epsilon) · A_t)]")
    print("    epsilon typically 0.1-0.2  (how far new policy can stray)")
    print()
    print("  Full loss:")
    print("    L = L^CLIP - c1·L_VF + c2·S[pi_theta]")
    print("    L_VF = (V_theta(s_t) - V_t^target)²   (value loss)")
    print("    S[pi]  = -Sigma pi log pi                 (entropy bonus)")
    print()
    print("  Training loop:")
    steps = [
        "1. Collect T timesteps of experience with pi_{theta_old}",
        "2. Compute advantages (GAE-lambda: generalised advantage estimation)",
        "3. For K epochs: mini-batch updates with clip objective",
        "4. theta_old <- theta",
    ]
    for s in steps:
        print(f"  {s}")
    print()
    print("  GAE: Â_t = Sigma_{l=0}^inf (gammalambda)^l delta_{t+l}  (smoothed advantage)")
    print()
    print("  PPO is default choice for:")
    print("    Continuous control (MuJoCo, robotics)")
    print("    Game playing (OpenAI Five, AlphaStar)")
    print("    RLHF (reinforcement learning from human feedback in LLMs)")


if __name__ == "__main__":
    dqn_theory()
    dqn_demo()
    policy_gradient()
    actor_critic()
    ppo_overview()
