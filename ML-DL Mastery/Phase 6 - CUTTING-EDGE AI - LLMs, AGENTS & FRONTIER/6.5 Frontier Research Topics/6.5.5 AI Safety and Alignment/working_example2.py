"""
Working Example 2: AI Safety and Alignment
Reward hacking simulation, KL divergence from reference policy,
and safety constraint analysis.
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


def kl_divergence(p, q):
    """KL(p || q) in nats."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def rlhf_reward_hacking(n_steps=200, beta=0.2, rng=None):
    """
    Simulate reward hacking: agent maximises proxy reward, 
    true reward starts high then decays as policy diverges.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    proxy_rewards, true_rewards, kl_divs = [], [], []
    policy = np.ones(10) / 10  # uniform reference
    for t in range(n_steps):
        # Proxy reward grows, true reward peaks then falls (Goodhart's law)
        proxy = 0.5 + 0.5 * (t / n_steps) + rng.normal(0, 0.05)
        kl = min(t / n_steps * 3, 3.0) + rng.normal(0, 0.05)
        true_reward = 0.8 * np.exp(-0.3 * kl) + rng.normal(0, 0.05)
        kl_penalised_reward = proxy - beta * kl
        proxy_rewards.append(proxy)
        true_rewards.append(true_reward)
        kl_divs.append(kl)
    return np.array(proxy_rewards), np.array(true_rewards), np.array(kl_divs)


def constitutional_filter(output_scores, safety_threshold=0.5):
    """Accept outputs above safety threshold."""
    accepted = output_scores >= safety_threshold
    return accepted, accepted.mean()


def demo():
    print("=== AI Safety and Alignment ===")
    rng = np.random.default_rng(42)

    proxy_r, true_r, kl_d = rlhf_reward_hacking(n_steps=300, beta=0.2, rng=rng)
    print(f"  Peak true reward at step: {true_r.argmax()}")
    print(f"  Final proxy reward: {proxy_r[-1]:.3f}")
    print(f"  Final true reward:  {true_r[-1]:.3f}")
    print(f"  Final KL divergence: {kl_d[-1]:.3f}")

    # Constitutional AI: filtering unsafe outputs
    n_outputs = 200
    harm_scores = rng.beta(2, 5, n_outputs)  # mostly low harm
    safety_scores = 1 - harm_scores
    accepted, accept_rate = constitutional_filter(safety_scores, 0.6)
    print(f"\n  Constitutional AI: accept rate = {accept_rate:.2f}")
    print(f"  Mean harm (accepted): {harm_scores[accepted].mean():.3f}")
    print(f"  Mean harm (rejected): {harm_scores[~accepted].mean():.3f}")

    # Beta sweep: KL penalty strength
    betas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    final_true_rewards = []
    for b in betas:
        _, tr, _ = rlhf_reward_hacking(n_steps=300, beta=b, rng=rng)
        final_true_rewards.append(tr.max())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    steps = np.arange(len(proxy_r))
    axes[0][0].plot(steps, proxy_r, color="tomato", lw=2, label="Proxy Reward")
    axes[0][0].plot(steps, true_r, color="steelblue", lw=2, label="True Reward")
    axes[0][0].axvline(true_r.argmax(), color="green", linestyle="--", alpha=0.5, label="True peak")
    axes[0][0].set(xlabel="Training Step", ylabel="Reward",
                   title="Reward Hacking (Goodhart's Law)")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    axes[0][1].plot(steps, kl_d, color="purple", lw=2)
    axes[0][1].set(xlabel="Training Step", ylabel="KL(pi || pi_ref)",
                   title="KL Divergence from Reference Policy")
    axes[0][1].grid(True, alpha=0.3)

    axes[1][0].hist(safety_scores[accepted], bins=20, alpha=0.7, color="mediumseagreen",
                     label=f"Accepted ({accept_rate:.0%})")
    axes[1][0].hist(safety_scores[~accepted], bins=20, alpha=0.7, color="tomato",
                     label=f"Rejected ({1-accept_rate:.0%})")
    axes[1][0].axvline(0.6, color="k", linestyle="--", label="Threshold=0.6")
    axes[1][0].set(xlabel="Safety Score", ylabel="Count",
                   title="Constitutional AI: Output Filtering")
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(True, alpha=0.3)

    axes[1][1].plot(betas, final_true_rewards, "o-", color="steelblue", lw=2)
    axes[1][1].set(xlabel="KL Penalty beta", ylabel="Peak True Reward",
                   title="KL Penalty beta vs True Reward")
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "ai_safety.png", dpi=100)
    plt.close()
    print("  Saved ai_safety.png")


if __name__ == "__main__":
    demo()
