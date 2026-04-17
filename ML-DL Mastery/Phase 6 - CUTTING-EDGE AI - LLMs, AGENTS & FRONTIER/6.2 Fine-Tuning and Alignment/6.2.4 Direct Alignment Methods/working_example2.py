"""
Working Example 2: Direct Alignment Methods (DPO)
Demonstrates the DPO loss formula and compares DPO vs SFT loss curves
on synthetic preference data.
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


def dpo_loss(log_pi_w, log_pi_l, log_ref_w, log_ref_l, beta=0.1):
    """DPO loss: -log σ(β * (log(π(w)/π_ref(w)) - log(π(l)/π_ref(l))))."""
    ratio = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
    return -np.log(1 / (1 + np.exp(-ratio)))


def demo():
    print("=== Direct Alignment Methods: DPO ===")
    rng = np.random.default_rng(42)
    n_pairs = 1000
    beta = 0.1

    # Synthetic: reference model log-probs
    log_ref_w = rng.normal(-2, 0.5, n_pairs)  # preferred responses
    log_ref_l = rng.normal(-3, 0.5, n_pairs)  # rejected responses

    # Training: simulate policy improving over steps
    steps = np.arange(1, 101)
    dpo_losses, sft_losses = [], []
    for step in steps:
        # Policy improves: preferred gets higher, rejected lower
        alpha = step / 100
        log_pi_w = log_ref_w + alpha * rng.normal(0.5, 0.1, n_pairs)
        log_pi_l = log_ref_l - alpha * rng.normal(0.3, 0.1, n_pairs)
        dpo_l = dpo_loss(log_pi_w, log_pi_l, log_ref_w, log_ref_l, beta).mean()
        # SFT loss: just NLL on preferred
        sft_l = -log_pi_w.mean() + rng.normal(0, 0.02)
        dpo_losses.append(dpo_l)
        sft_losses.append(sft_l)

    print(f"  DPO final loss: {dpo_losses[-1]:.4f}")
    print(f"  SFT final loss: {sft_losses[-1]:.4f}")

    # DPO margin: implicit reward difference
    margins = np.linspace(-3, 3, 200)
    loss_by_margin = -np.log(1 / (1 + np.exp(-beta * margins)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    axes[0].plot(steps, dpo_losses, color="steelblue", lw=2, label="DPO Loss")
    axes[0].plot(steps, sft_losses, color="tomato", lw=2, label="SFT NLL")
    axes[0].set(xlabel="Training Steps", ylabel="Loss",
                title="DPO vs SFT Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # DPO loss as function of implicit reward margin
    axes[1].plot(margins, loss_by_margin, color="darkorange", lw=2)
    axes[1].set(xlabel="Implicit Reward Margin β(r_w − r_l)",
                ylabel="DPO Loss",
                title="DPO Loss vs Reward Margin")
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].grid(True, alpha=0.3)

    # Reference vs policy reward
    final_alpha = 1.0
    log_pi_w_f = log_ref_w + final_alpha * 0.5
    log_pi_l_f = log_ref_l - final_alpha * 0.3
    axes[2].scatter(log_ref_w[:200], log_pi_w_f[:200], alpha=0.4, s=15,
                    color="green", label="Preferred (w)")
    axes[2].scatter(log_ref_l[:200], log_pi_l_f[:200], alpha=0.4, s=15,
                    color="red", label="Rejected (l)")
    mn = min(log_ref_w.min(), log_ref_l.min())
    mx = max(log_ref_w.max(), log_ref_l.max())
    axes[2].plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
    axes[2].set(xlabel="log π_ref", ylabel="log π_DPO",
                title="Policy vs Reference Log-Probs")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "dpo_demo.png", dpi=100)
    plt.close()
    print("  Saved dpo_demo.png")


if __name__ == "__main__":
    demo()
