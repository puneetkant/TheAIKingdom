"""
Working Example 2: Ethics and Responsible AI
Fairness metrics (demographic parity, equalized odds, calibration),
disparate impact analysis, and model card template.
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


def generate_loan_data(n=1000, rng=None):
    """
    Simulate loan approval data with two demographic groups (A, B).
    Group B has a historically biased approval rate.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    group = rng.choice([0, 1], n, p=[0.5, 0.5])  # 0=A, 1=B
    credit_score = rng.normal(650 + group * 30, 80, n).clip(300, 850)

    # Ground truth: approve if credit >= 620
    y_true = (credit_score >= 620).astype(int)

    # Biased model: lower threshold applied to group B
    bias_factor = np.where(group == 1, -25, 0)
    y_pred = (credit_score + bias_factor >= 620).astype(int)

    return group, y_true, y_pred, credit_score


def demographic_parity(group, y_pred):
    """DP: difference in positive prediction rates between groups."""
    rate_a = y_pred[group == 0].mean()
    rate_b = y_pred[group == 1].mean()
    return rate_a, rate_b, abs(rate_a - rate_b)


def equalized_odds(group, y_true, y_pred):
    """
    EO: difference in TPR and FPR between groups.
    """
    def tpr_fpr(g):
        mask = (group == g)
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        return tpr, fpr

    tpr_a, fpr_a = tpr_fpr(0)
    tpr_b, fpr_b = tpr_fpr(1)
    return (tpr_a, fpr_a, tpr_b, fpr_b,
            abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))


def calibration_by_group(group, y_true, y_pred_prob, bins=10):
    """Calibration curve per group."""
    results = {}
    for g in [0, 1]:
        mask = (group == g)
        probs = y_pred_prob[mask]
        truths = y_true[mask]
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_means, bin_fracs = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (probs >= lo) & (probs < hi)
            if in_bin.sum() > 0:
                bin_means.append(probs[in_bin].mean())
                bin_fracs.append(truths[in_bin].mean())
        results[g] = (bin_means, bin_fracs)
    return results


def demo():
    print("=== Ethics and Responsible AI: Fairness Metrics ===")
    rng = np.random.default_rng(42)

    group, y_true, y_pred, credit_score = generate_loan_data(n=2000, rng=rng)
    print(f"\n  Dataset: n={len(group)}, Group A: {(group==0).sum()}, Group B: {(group==1).sum()}")
    print(f"  Overall approval rate: {y_pred.mean():.3f}")

    # Demographic Parity
    rate_a, rate_b, dp_gap = demographic_parity(group, y_pred)
    print(f"\n  Demographic Parity:")
    print(f"    Group A approval: {rate_a:.3f}")
    print(f"    Group B approval: {rate_b:.3f}")
    print(f"    DP Gap: {dp_gap:.3f} {'[!] UNFAIR (>0.05)' if dp_gap > 0.05 else '[OK]'}")

    # Equalized Odds
    tpr_a, fpr_a, tpr_b, fpr_b, tpr_gap, fpr_gap = equalized_odds(group, y_true, y_pred)
    print(f"\n  Equalized Odds:")
    print(f"    TPR A: {tpr_a:.3f}, TPR B: {tpr_b:.3f}, gap: {tpr_gap:.3f}")
    print(f"    FPR A: {fpr_a:.3f}, FPR B: {fpr_b:.3f}, gap: {fpr_gap:.3f}")

    # Soft probabilities (logistic-like from credit score)
    y_pred_prob = 1 / (1 + np.exp(-(credit_score - 620) / 40))
    cal = calibration_by_group(group, y_true, y_pred_prob)

    # Credit score distribution
    bins = np.linspace(300, 850, 50)
    hist_a, _ = np.histogram(credit_score[group == 0], bins)
    hist_b, _ = np.histogram(credit_score[group == 1], bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Disparate impact ratio
    di = min(rate_a, rate_b) / (max(rate_a, rate_b) + 1e-10)
    print(f"\n  Disparate Impact Ratio: {di:.3f} {'[!] <0.8 threshold violated' if di < 0.8 else '[OK] >0.8'}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Credit score distributions
    axes[0][0].plot(bin_centers, hist_a, color="steelblue", lw=2, label="Group A")
    axes[0][0].plot(bin_centers, hist_b, color="tomato", lw=2, linestyle="--", label="Group B")
    axes[0][0].axvline(620, color="k", linestyle=":", alpha=0.6, label="Threshold 620")
    axes[0][0].set(xlabel="Credit Score", ylabel="Count",
                   title="Credit Score Distribution by Group")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # Demographic Parity
    axes[0][1].bar(["Group A", "Group B"], [rate_a, rate_b],
                    color=["steelblue", "tomato"], alpha=0.8)
    axes[0][1].axhline(rate_a, color="steelblue", linestyle="--", alpha=0.5)
    axes[0][1].set(ylabel="Approval Rate", title=f"Demographic Parity (gap={dp_gap:.3f})")
    axes[0][1].set_ylim(0, 1)
    axes[0][1].grid(True, axis="y", alpha=0.3)
    for i, (label, val) in enumerate(zip(["Group A", "Group B"], [rate_a, rate_b])):
        axes[0][1].text(i, val + 0.01, f"{val:.3f}", ha="center")

    # Equalized Odds bars
    metrics = ["TPR", "FPR"]
    vals_a = [tpr_a, fpr_a]
    vals_b = [tpr_b, fpr_b]
    x = np.arange(2)
    axes[1][0].bar(x - 0.2, vals_a, 0.4, label="Group A", color="steelblue", alpha=0.8)
    axes[1][0].bar(x + 0.2, vals_b, 0.4, label="Group B", color="tomato", alpha=0.8)
    axes[1][0].set(xticks=x, xticklabels=metrics,
                   ylabel="Rate", title="Equalized Odds")
    axes[1][0].legend()
    axes[1][0].grid(True, axis="y", alpha=0.3)

    # Calibration curves
    axes[1][1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for g, color, label in [(0, "steelblue", "Group A"), (1, "tomato", "Group B")]:
        bm, bf = cal[g]
        axes[1][1].plot(bm, bf, "o-", color=color, lw=2, label=label)
    axes[1][1].set(xlabel="Mean Predicted Prob", ylabel="Fraction Positive",
                   title="Calibration Curves by Group")
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "responsible_ai.png", dpi=100)
    plt.close()
    print("\n  Saved responsible_ai.png")

    print("\n  === Model Card (Summary) ===")
    print("  Model: Loan Approval Classifier v1")
    print("  Training data: Simulated 2000-sample loan dataset")
    print(f"  Overall accuracy: {(y_pred == y_true).mean():.3f}")
    print(f"  Demographic Parity gap: {dp_gap:.3f}")
    print(f"  Disparate Impact ratio: {di:.3f}")
    print("  Known limitations: biased threshold per group")
    print("  Recommended mitigation: threshold equalization or reweighting")


if __name__ == "__main__":
    demo()
