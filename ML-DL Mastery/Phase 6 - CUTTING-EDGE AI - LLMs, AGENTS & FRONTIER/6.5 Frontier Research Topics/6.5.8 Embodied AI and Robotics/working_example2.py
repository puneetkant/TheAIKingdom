"""
Working Example 2: Embodied AI and Robotics
Grid-world navigation with policy gradient simulation.
Run: python working_example2.py
"""
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["^", "v", "<-", "->"]


class GridWorld:
    def __init__(self, size=6, goal=(5, 5), obstacles=None):
        self.size = size
        self.goal = goal
        self.obstacles = set(obstacles or [(2, 2), (2, 3), (3, 2), (4, 4)])
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            self.pos = (nr, nc)
        done = self.pos == self.goal
        reward = 10.0 if done else -0.1
        return self.pos, reward, done


def softmax_policy(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def episode(env, theta, rng, max_steps=50):
    """Run one episode with softmax policy; return log-probs, rewards, states."""
    state = env.reset()
    n_states = env.size * env.size
    log_probs, rewards, states = [], [], []

    for _ in range(max_steps):
        s_idx = state[0] * env.size + state[1]
        logits = theta[s_idx]
        probs = softmax_policy(logits)
        action = rng.choice(4, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        next_state, reward, done = env.step(action)
        log_probs.append(log_prob); rewards.append(reward); states.append(state)
        state = next_state
        if done:
            break

    return log_probs, rewards, states


def policy_gradient_update(theta, log_probs, rewards, lr=0.05, gamma=0.95):
    """REINFORCE update with discount."""
    T = len(rewards)
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    # Baseline: mean return
    baseline = returns.mean()
    # Gradient ascent (add, because we want to maximise)
    for t, (lp, g) in enumerate(zip(log_probs, returns)):
        theta += lr * (g - baseline)  # simplified: same update to all states
    return theta


def demo():
    print("=== Embodied AI and Robotics: Grid-World Navigation ===")
    rng = np.random.default_rng(42)
    env = GridWorld(size=6, goal=(5, 5))
    n_states = env.size * env.size
    theta = rng.standard_normal((n_states, 4)) * 0.1

    n_episodes = 200
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        env.reset()
        log_probs, rewards, states = episode(env, theta, rng, max_steps=80)
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))
        theta = policy_gradient_update(theta.copy(), log_probs, rewards, lr=0.02)
        if ep % 40 == 0:
            print(f"  Episode {ep:3d}: reward={total_reward:.2f}, length={len(rewards)}")

    # Show learned policy on grid
    grid_size = env.size
    policy_grid = np.argmax(
        np.array([softmax_policy(theta[s]) for s in range(n_states)]), axis=1
    ).reshape(grid_size, grid_size)

    # Smooth reward curve
    def moving_avg(x, w=20):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Training reward curve
    axes[0][0].plot(episode_rewards, alpha=0.3, color="steelblue")
    if len(episode_rewards) >= 20:
        axes[0][0].plot(range(19, n_episodes), moving_avg(episode_rewards, 20),
                         color="steelblue", lw=2, label="MA-20")
    axes[0][0].set(xlabel="Episode", ylabel="Total Reward", title="Policy Gradient Training Reward")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # Episode length
    axes[0][1].plot(episode_lengths, alpha=0.3, color="tomato")
    if len(episode_lengths) >= 20:
        axes[0][1].plot(range(19, n_episodes), moving_avg(episode_lengths, 20),
                         color="tomato", lw=2, label="MA-20")
    axes[0][1].set(xlabel="Episode", ylabel="Steps", title="Episode Length")
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    # Policy grid (arrow for each cell)
    OBSTACLE = -1
    grid_display = np.full((grid_size, grid_size), 0.5)
    for obs in env.obstacles:
        grid_display[obs] = 0.0
    grid_display[env.goal] = 1.0
    axes[1][0].imshow(grid_display, cmap="RdYlGn", vmin=0, vmax=1)
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in env.obstacles:
                axes[1][0].text(c, r, "X", ha="center", va="center", fontsize=12, fontweight="bold")
            elif (r, c) == env.goal:
                axes[1][0].text(c, r, "G", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
            else:
                axes[1][0].text(c, r, ACTION_NAMES[policy_grid[r, c]], ha="center", va="center", fontsize=14)
    axes[1][0].set(title="Learned Policy Grid", xticks=[], yticks=[])

    # Action distribution of last episode
    env.reset()
    lp, rw, _ = episode(env, theta, rng, max_steps=80)
    action_hist = np.zeros(4)
    # rerun to collect actions
    state = env.reset()
    for _ in range(80):
        s_idx = state[0] * env.size + state[1]
        probs = softmax_policy(theta[s_idx])
        action = rng.choice(4, p=probs)
        action_hist[action] += 1
        next_state, _, done = env.step(action)
        state = next_state
        if done:
            break
    axes[1][1].bar(ACTION_NAMES, action_hist, color="mediumseagreen")
    axes[1][1].set(ylabel="Count", title="Action Distribution (last episode)")
    axes[1][1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "embodied_ai.png", dpi=100)
    plt.close()
    print("  Saved embodied_ai.png")


if __name__ == "__main__":
    demo()
