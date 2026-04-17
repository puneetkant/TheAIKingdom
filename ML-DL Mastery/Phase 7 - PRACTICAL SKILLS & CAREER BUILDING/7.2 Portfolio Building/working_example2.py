"""
Working Example 2: Portfolio Building
Project impact metrics, documentation quality scoring,
and portfolio score calculator.
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


def score_project(readme_len, n_code_files, has_demo, has_results, stars, n_commits, n_tests):
    """
    Composite portfolio project score (0-100).
    Components:
    - Documentation quality (README length)
    - Code coverage (# files, tests)
    - Completeness (demo, results)
    - Community engagement (stars, commits)
    """
    doc_score = min(readme_len / 1000, 1.0) * 25          # max 25
    code_score = min(n_code_files / 10, 1.0) * 20          # max 20
    test_score = min(n_tests / 5, 1.0) * 10                # max 10
    complete_score = (has_demo + has_results) * 10          # max 20
    engagement = min(np.log1p(stars) / 5, 1.0) * 15 + \
                 min(n_commits / 50, 1.0) * 10              # max 25
    total = doc_score + code_score + test_score + complete_score + engagement
    return min(total, 100), {
        "Documentation": doc_score,
        "Code Coverage": code_score,
        "Tests": test_score,
        "Completeness": complete_score,
        "Engagement": engagement,
    }


def demo():
    print("=== Portfolio Building ===")

    projects = [
        {"name": "Image Classifier", "readme_len": 800, "n_files": 5, "has_demo": 1,
         "has_results": 1, "stars": 12, "commits": 30, "n_tests": 3},
        {"name": "NLP Sentiment", "readme_len": 1200, "n_files": 8, "has_demo": 1,
         "has_results": 1, "stars": 45, "commits": 80, "n_tests": 6},
        {"name": "RL Game Agent", "readme_len": 500, "n_files": 3, "has_demo": 0,
         "has_results": 0, "stars": 3, "commits": 10, "n_tests": 0},
        {"name": "Time Series", "readme_len": 950, "n_files": 6, "has_demo": 1,
         "has_results": 1, "stars": 22, "commits": 45, "n_tests": 4},
        {"name": "Kaggle Top-10%", "readme_len": 1500, "n_files": 12, "has_demo": 1,
         "has_results": 1, "stars": 120, "commits": 150, "n_tests": 8},
    ]

    scores = []
    components_all = []
    for p in projects:
        score, components = score_project(
            p["readme_len"], p["n_files"], p["has_demo"],
            p["has_results"], p["stars"], p["commits"], p["n_tests"]
        )
        scores.append(score)
        components_all.append(components)
        print(f"  {p['name']:20s}: {score:.1f}/100")

    # Portfolio total
    portfolio_score = np.mean(scores)
    print(f"\n  Portfolio mean score: {portfolio_score:.1f}/100")

    # Improvement impact simulation
    dimensions = ["Documentation", "Code Coverage", "Tests", "Completeness", "Engagement"]
    improvement_effects = []
    for dim in dimensions:
        # Simulate +25% improvement in this dimension
        effect = np.mean([c[dim] * 1.25 - c[dim] for c in components_all])
        improvement_effects.append(effect)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Project scores bar chart
    colors = plt.cm.RdYlGn(np.array(scores) / 100)
    axes[0][0].barh([p["name"] for p in projects], scores, color=colors)
    axes[0][0].axvline(70, color="red", linestyle="--", alpha=0.5, label="Good threshold (70)")
    axes[0][0].set(xlabel="Score (/100)", title="Portfolio Project Scores")
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(True, axis="x", alpha=0.3)

    # Stacked bar: component breakdown
    x = np.arange(len(projects))
    bottoms = np.zeros(len(projects))
    component_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]
    for dim, color in zip(dimensions, component_colors):
        vals = [c[dim] for c in components_all]
        axes[0][1].bar(x, vals, bottom=bottoms, label=dim, color=color, alpha=0.85)
        bottoms += np.array(vals)
    axes[0][1].set(xticks=x, xticklabels=[p["name"][:8] for p in projects],
                   ylabel="Score", title="Component Breakdown")
    axes[0][1].legend(fontsize=7, loc="upper left")
    axes[0][1].tick_params(axis="x", rotation=20)
    axes[0][1].grid(True, axis="y", alpha=0.3)

    # Improvement impact
    axes[1][0].bar(dimensions, improvement_effects, color="mediumseagreen")
    axes[1][0].set(xlabel="Dimension", ylabel="Score Gain per Project",
                   title="Impact of 25% Improvement per Dimension")
    axes[1][0].tick_params(axis="x", rotation=20)
    axes[1][0].grid(True, axis="y", alpha=0.3)

    # Stars vs Score scatter
    stars = [p["stars"] for p in projects]
    axes[1][1].scatter(stars, scores, s=100, color="steelblue", zorder=5)
    for p, s in zip(projects, scores):
        axes[1][1].annotate(p["name"][:8], (p["stars"], s), textcoords="offset points",
                             xytext=(5, 3), fontsize=7)
    axes[1][1].set(xlabel="GitHub Stars", ylabel="Portfolio Score",
                   title="Stars vs Portfolio Score")
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "portfolio_building.png", dpi=100)
    plt.close()
    print("  Saved portfolio_building.png")


if __name__ == "__main__":
    demo()
