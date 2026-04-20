"""
Working Example 2: Reinforcement Learning from Human Feedback (RLHF)
Demonstrates reward model training with Bradley-Terry pairwise preferences
and a PPO-style KL-penalised objective.
Run: python working_example2.py
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


def bradley_terry_prob(r_w, r_l):
    """P(w > l) under Bradley-Terry model."""
    return 1.0 / (1.0 + np.exp(r_l - r_w))


def demo():
    print("=== RLHF: Reward Model + PPO KL Penalty ===")
    rng = np.random.default_rng(42)
    n_pairs = 500

    # Synthetic pairwise preference data
    true_rewards = rng.normal(0, 1, 200)   # 200 responses
    win_idx = rng.integers(0, 200, n_pairs)
    lose_idx = rng.integers(0, 200, n_pairs)
    same = win_idx == lose_idx
    lose_idx[same] = (lose_idx[same] + 1) % 200
    # Preference determined by true reward with noise
    prefer_win = true_rewards[win_idx] > true_rewards[lose_idx]

    # Estimate reward for each response (simple linear proxy)
    estimated_rewards = np.zeros(200)
    for _ in range(300):  # gradient steps
        for i in range(n_pairs):
            w, l = win_idx[i], lose_idx[i]
            p = bradley_terry_prob(estimated_rewards[w], estimated_rewards[l])
            grad = 1 - p if prefer_win[i] else -p
            estimated_rewards[w] += 0.01 * grad
            estimated_rewards[l] -= 0.01 * grad

    corr = np.corrcoef(true_rewards, estimated_rewards)[0, 1]
    print(f"  Correlation between true & estimated rewards: {corr:.3f}")

    # PPO KL penalty objective: r(x,y) - beta * KL(pi || pi_ref)
    betas = np.linspace(0, 2, 100)
    mean_reward = 1.5  # after RL training
    kl_div = 0.8       # typical KL from reference
    objective = mean_reward - betas * kl_div

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # True vs estimated reward
    axes[0].scatter(true_rewards, estimated_rewards, alpha=0.4, s=15, color="steelblue")
    mn, mx = true_rewards.min(), true_rewards.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
    axes[0].set(xlabel="True Reward", ylabel="Estimated Reward",
                title=f"Reward Model Calibration (r={corr:.2f})")
    axes[0].grid(True, alpha=0.3)

    # Bradley-Terry: win probability surface
    r_w = np.linspace(-3, 3, 100)
    r_l = np.linspace(-3, 3, 100)
    RW, RL = np.meshgrid(r_w, r_l)
    P = bradley_terry_prob(RW, RL)
    im = axes[1].contourf(r_w, r_l, P, levels=20, cmap="RdYlGn")
    axes[1].set(xlabel="Reward(winner)", ylabel="Reward(loser)",
                title="Bradley-Terry P(w>l)")
    plt.colorbar(im, ax=axes[1])

    # PPO KL penalty trade-off
    axes[2].plot(betas, objective, color="tomato", lw=2)
    axes[2].axhline(0, color="gray", linestyle="--", lw=1)
    axes[2].set(xlabel="KL Penalty beta", ylabel="PPO Objective",
                title="PPO: Reward - beta·KL(pi || pi_ref)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "rlhf_demo.png", dpi=100)
    plt.close()
    print("  Saved rlhf_demo.png")


def demo_reward_hacking():
    """Show reward hacking: optimising a proxy reward can diverge from true reward."""
    print("\n=== Reward Hacking ===")
    rng = np.random.default_rng(1)
    steps = np.arange(1, 101)
    # True reward improves then plateaus; proxy reward keeps climbing (hacking)
    true_reward  = 1.5 * (1 - np.exp(-0.05 * steps)) + rng.normal(0, 0.05, len(steps))
    proxy_reward = 2.5 * (1 - np.exp(-0.03 * steps)) + rng.normal(0, 0.05, len(steps))
    gap = proxy_reward - true_reward
    print(f"  Gap at step 50:  {gap[49]:.3f}")
    print(f"  Gap at step 100: {gap[99]:.3f}")
    plt.figure(figsize=(6, 4))
    plt.plot(steps, true_reward,  lw=2, color="steelblue", label="True Reward")
    plt.plot(steps, proxy_reward, lw=2, color="tomato",    label="Proxy (RM) Reward")
    plt.fill_between(steps, true_reward, proxy_reward, alpha=0.15, color="red", label="Hacking Gap")
    plt.xlabel("Training Steps"); plt.ylabel("Reward")
    plt.title("Reward Hacking: Proxy vs True Reward")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "reward_hacking.png", dpi=100); plt.close()
    print("  Saved reward_hacking.png")


def demo_kl_beta_sweep():
    """Show how varying beta changes the KL-reward tradeoff in PPO-RLHF."""
    print("\n=== KL Beta Sweep ===")
    betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    kl_div = 0.8
    mean_reward = 1.5
    for b in betas:
        obj = mean_reward - b * kl_div
        print(f"  beta={b:.2f}: objective={obj:.3f}")
    objectives = [mean_reward - b * kl_div for b in betas]
    plt.figure(figsize=(5, 3))
    plt.plot(betas, objectives, "o-", color="#9b59b6", lw=2)
    plt.axhline(0, color="gray", linestyle="--", lw=1)
    plt.xlabel("KL Penalty beta"); plt.ylabel("PPO Objective")
    plt.title("Beta Sweep: Reward vs KL Penalty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "kl_beta_sweep.png", dpi=100); plt.close()
    print("  Saved kl_beta_sweep.png")


if __name__ == "__main__":
    demo()
    demo_reward_hacking()
    demo_kl_beta_sweep()
