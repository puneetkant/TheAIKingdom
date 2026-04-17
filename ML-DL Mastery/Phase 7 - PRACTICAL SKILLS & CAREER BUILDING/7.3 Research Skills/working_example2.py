"""
Working Example 2: Research Skills
Paper tracker, citation network analysis, and ablation study template.
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


def simulate_citation_network(n_papers=20, max_refs=6, rng=None):
    """
    Simulate a citation DAG: each paper i cites a random subset of {0..i-1}.
    Returns adjacency matrix and citation counts.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    adj = np.zeros((n_papers, n_papers), dtype=int)
    for i in range(1, n_papers):
        n_refs = rng.integers(1, min(i + 1, max_refs + 1))
        refs = rng.choice(i, n_refs, replace=False)
        for r in refs:
            adj[i, r] = 1  # paper i cites paper r

    citation_counts = adj.sum(axis=0)  # in-degree = # times cited
    return adj, citation_counts


def ablation_scores(baseline, components, drop_pcts):
    """
    Simulate ablation study: removing each component drops performance by drop_pct.
    Returns dict of ablation results.
    """
    results = {"full": baseline}
    for comp, pct in zip(components, drop_pcts):
        results[f"w/o {comp}"] = baseline * (1 - pct)
    return results


def demo():
    print("=== Research Skills: Paper Tracker & Citation Analysis ===")
    rng = np.random.default_rng(42)

    # Paper tracker
    papers = [
        {"title": "Attention Is All You Need", "year": 2017, "venue": "NeurIPS", "read": True, "citations": 85000},
        {"title": "BERT", "year": 2018, "venue": "NAACL", "read": True, "citations": 60000},
        {"title": "GPT-3", "year": 2020, "venue": "NeurIPS", "read": True, "citations": 30000},
        {"title": "LoRA", "year": 2021, "venue": "ICLR", "read": True, "citations": 8000},
        {"title": "InstructGPT", "year": 2022, "venue": "NeurIPS", "read": False, "citations": 5000},
        {"title": "LLaMA", "year": 2023, "venue": "arXiv", "read": True, "citations": 12000},
        {"title": "Mamba", "year": 2023, "venue": "arXiv", "read": False, "citations": 3000},
        {"title": "Mixtral MoE", "year": 2024, "venue": "arXiv", "read": False, "citations": 1200},
    ]
    print(f"\n  Paper Tracker: {len(papers)} papers")
    read_count = sum(1 for p in papers if p["read"])
    print(f"  Read: {read_count}/{len(papers)}")

    # Citation network
    adj, cit_counts = simulate_citation_network(n_papers=15, max_refs=5, rng=rng)
    print(f"\n  Citation network: {adj.shape[0]} papers")
    print(f"  Most cited: paper {cit_counts.argmax()} ({cit_counts.max()} citations)")
    print(f"  Mean citations: {cit_counts.mean():.2f}")

    # Ablation study
    baseline_acc = 0.873
    components = ["Attention", "LayerNorm", "Dropout", "Positional Enc", "Feed-Forward"]
    drop_pcts = [0.15, 0.05, 0.03, 0.08, 0.12]
    ablation = ablation_scores(baseline_acc, components, drop_pcts)
    print(f"\n  Ablation study:")
    for name, score in ablation.items():
        print(f"    {name:25s}: {score:.4f}")

    # Reading progress over time (simulate)
    months = np.arange(1, 13)
    papers_read = np.cumsum(rng.integers(2, 8, 12))
    papers_added = np.cumsum(rng.integers(5, 15, 12))
    reading_pct = np.minimum(papers_read / papers_added, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Citation count distribution
    axes[0][0].bar([p["title"][:15] for p in papers], [p["citations"] for p in papers],
                    color=["mediumseagreen" if p["read"] else "tomato" for p in papers])
    axes[0][0].set(ylabel="Citations", title="Paper Citations (green=read)")
    axes[0][0].tick_params(axis="x", rotation=30)
    axes[0][0].set_yscale("log")
    axes[0][0].grid(True, axis="y", alpha=0.3)

    # Citation network heatmap
    im = axes[0][1].imshow(adj, cmap="Blues", aspect="auto")
    axes[0][1].set(title="Citation Network (paper i cites j)",
                   xlabel="Cited paper (j)", ylabel="Citing paper (i)")
    plt.colorbar(im, ax=axes[0][1])

    # Ablation chart
    names = list(ablation.keys())
    scores_ab = list(ablation.values())
    colors_ab = ["mediumseagreen"] + ["steelblue"] * (len(names) - 1)
    axes[1][0].bar(names, scores_ab, color=colors_ab)
    axes[1][0].set(ylabel="Accuracy", title="Ablation Study")
    axes[1][0].set_ylim(0.7, 0.92)
    axes[1][0].tick_params(axis="x", rotation=25)
    axes[1][0].grid(True, axis="y", alpha=0.3)
    for i, (n, s) in enumerate(zip(names, scores_ab)):
        axes[1][0].text(i, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)

    # Reading progress
    axes[1][1].plot(months, reading_pct * 100, "o-", color="steelblue", lw=2, label="% Read")
    axes[1][1].plot(months, (papers_read / papers_read[-1]) * 100, "s--", color="tomato",
                     lw=2, label="Papers read (cum)")
    axes[1][1].set(xlabel="Month", ylabel="Percent / Count proxy",
                   title="Research Reading Progress")
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "research_skills.png", dpi=100)
    plt.close()
    print("  Saved research_skills.png")


if __name__ == "__main__":
    demo()
