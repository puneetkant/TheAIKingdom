"""
Working Example 2: Key LLM Families
Parameter count comparison and capability overview of major LLM families.
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


def demo():
    print("=== Key LLM Families: Parameter Count Comparison ===")

    models = [
        "GPT-3\n(175B)", "GPT-4\n(~1.8T est.)", "LLaMA-2\n(70B)",
        "LLaMA-3\n(70B)", "Mistral\n(7B)", "Mixtral\n(8x7B)",
        "Gemma\n(7B)", "Phi-3\n(3.8B)", "Falcon\n(40B)", "Claude-3\n(~500B est.)"
    ]
    params_b = [175, 1800, 70, 70, 7, 46.7, 7, 3.8, 40, 500]
    open_source = [False, False, True, True, True, True, True, True, True, False]
    colors = ["#e74c3c" if not o else "#2ecc71" for o in open_source]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: parameter counts
    bars = axes[0].bar(range(len(models)), params_b, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, fontsize=7)
    axes[0].set(ylabel="Parameters (Billions)", title="LLM Parameter Counts")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#2ecc71", label="Open Source"),
              Patch(color="#e74c3c", label="Closed Source")]
    axes[0].legend(handles=legend, fontsize=8)

    # Log-scale comparison
    sorted_idx = np.argsort(params_b)
    sorted_models = [models[i].replace("\n", " ") for i in sorted_idx]
    sorted_params = [params_b[i] for i in sorted_idx]
    axes[1].barh(sorted_models, sorted_params, color="#3498db", edgecolor="white")
    axes[1].set(xlabel="Parameters (Billions, log scale)", title="Sorted by Size")
    axes[1].set_xscale("log")
    axes[1].grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "llm_families.png", dpi=100)
    plt.close()
    print("  Saved llm_families.png")

    print("\n  Summary table:")
    header = f"{'Model':<20} {'Params (B)':>12} {'Open Source':>12}"
    print("  " + header)
    print("  " + "-" * len(header))
    for m, p, o in zip(models, params_b, open_source):
        m_clean = m.replace("\n", " ")
        print(f"  {m_clean:<20} {p:>12.1f} {'Yes' if o else 'No':>12}")


def demo_efficiency_ratio():
    """Performance-per-parameter: smaller models can punch above their weight."""
    print("\n=== Efficiency Ratio ===")
    models_short = ["GPT-3", "GPT-4", "LLaMA-2", "LLaMA-3", "Mistral",
                    "Mixtral", "Gemma", "Phi-3", "Falcon", "Claude-3"]
    params_b = [175, 1800, 70, 70, 7, 46.7, 7, 3.8, 40, 500]
    # Synthetic benchmark scores (0-100) representing overall capability
    scores   = [72, 92, 74, 80, 70, 79, 68, 69, 67, 88]
    ratios   = [s / p for s, p in zip(scores, params_b)]
    idx = np.argsort(ratios)[::-1]
    print("  Score-per-billion-parameter ranking:")
    for i in idx:
        print(f"    {models_short[i]:<12} score={scores[i]:3d}  params={params_b[i]:6.1f}B  ratio={ratios[i]:.3f}")
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(models_short)), [ratios[i] for i in idx],
            color="#9b59b6", edgecolor="white")
    plt.xticks(range(len(models_short)), [models_short[i] for i in idx], rotation=30, ha="right", fontsize=8)
    plt.ylabel("Score / Billion Params")
    plt.title("LLM Efficiency: Score per Billion Parameters")
    plt.tight_layout()
    plt.savefig(OUTPUT / "llm_efficiency.png", dpi=100); plt.close()
    print("  Saved llm_efficiency.png")


def demo_open_vs_closed_trend():
    """Show the trend of open-source models closing the gap with closed models."""
    print("\n=== Open vs Closed Model Trend ===")
    years = [2020, 2021, 2022, 2023, 2024]
    closed_score = [65, 72, 80, 90, 93]
    open_score   = [30, 42, 58, 76, 85]
    gap = [c - o for c, o in zip(closed_score, open_score)]
    for y, g in zip(years, gap):
        print(f"  {y}: gap = {g} points")
    plt.figure(figsize=(6, 4))
    plt.plot(years, closed_score, "o-", color="#e74c3c", lw=2, label="Closed Source")
    plt.plot(years, open_score,   "o-", color="#2ecc71", lw=2, label="Open Source")
    plt.fill_between(years, open_score, closed_score, alpha=0.15, color="gray")
    plt.xlabel("Year"); plt.ylabel("Benchmark Score")
    plt.title("Open vs Closed LLM Capability Gap")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "open_closed_trend.png", dpi=100); plt.close()
    print("  Saved open_closed_trend.png")


if __name__ == "__main__":
    demo()
    demo_efficiency_ratio()
    demo_open_vs_closed_trend()
