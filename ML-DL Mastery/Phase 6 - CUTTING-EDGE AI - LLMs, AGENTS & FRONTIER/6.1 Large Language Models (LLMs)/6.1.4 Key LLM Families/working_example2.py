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


if __name__ == "__main__":
    demo()
