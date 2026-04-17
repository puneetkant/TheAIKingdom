"""
Working Example 2: Staying Current
Simulates arXiv paper volume growth, a personalised reading schedule,
and topic trend tracking.
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


def simulate_arxiv_growth(start_year=2015, end_year=2024, rng=None):
    """
    Simulate monthly arXiv CS.AI + CS.LG paper counts based on approximate
    historical growth (doubling roughly every 2 years).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    years = np.arange(start_year, end_year + 1)
    # Exponential baseline: ~500 papers/month in 2015, doubling every 2 yr
    baseline = 500 * 2 ** ((years - 2015) / 2)
    noise = rng.normal(0, baseline * 0.08)
    return years, np.maximum(baseline + noise, 0)


def reading_schedule(papers_per_week, hours_per_paper, hours_per_week):
    """
    Given a reading budget, compute how many papers can be read per week
    and how many remain unread from total published.
    """
    readable = hours_per_week / hours_per_paper
    backlog_pct = max(0, (papers_per_week - readable) / papers_per_week * 100)
    return readable, backlog_pct


def topic_trend(topics, n_years=9, rng=None):
    """Simulate topic interest scores over years."""
    if rng is None:
        rng = np.random.default_rng(42)
    years = np.arange(2015, 2015 + n_years)
    trends = {}
    for topic, (start, growth) in topics.items():
        t = np.arange(n_years)
        curve = start * np.exp(growth * t) + rng.normal(0, start * 0.1, n_years)
        trends[topic] = np.clip(curve, 0, None)
    return years, trends


def demo():
    print("=== Staying Current with AI Research ===")
    rng = np.random.default_rng(42)

    years, paper_counts = simulate_arxiv_growth(rng=rng)
    print(f"\n  arXiv paper volume (simulated):")
    for y, c in zip(years, paper_counts):
        print(f"    {y}: {c:,.0f} papers/month")

    # Reading schedule
    curr_papers_per_week = int(paper_counts[-1] / 4)
    readable, backlog = reading_schedule(curr_papers_per_week, 0.5, 10)
    print(f"\n  Current estimated papers/week (2024): {curr_papers_per_week:,}")
    print(f"  With 10 hr/week @ 0.5 hr/paper: can read {readable:.0f} papers/week")
    print(f"  Backlog rate: {backlog:.1f}%")

    # Topic trends
    topics = {
        "Transformers":  (2, 0.50),
        "Diffusion":     (0.1, 0.90),
        "RL":            (1, 0.25),
        "LLMs":          (0.2, 0.85),
        "GNNs":          (0.5, 0.35),
    }
    trend_years, trends = topic_trend(topics, rng=rng)

    # Weekly reading plan
    reading_plan = {
        "Monday":    "Scan arXiv digest (30 min)",
        "Tuesday":   "Deep read 1 paper (60 min)",
        "Wednesday": "Blog post / tutorial (45 min)",
        "Thursday":  "Re-implement key paper (90 min)",
        "Friday":    "Twitter/X + LinkedIn scan (20 min)",
        "Weekend":   "Long paper + project coding (120 min)",
    }
    print("\n  Weekly reading plan:")
    for day, task in reading_plan.items():
        print(f"    {day:12s}: {task}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Paper volume growth
    axes[0][0].plot(years, paper_counts / 1000, "o-", color="steelblue", lw=2)
    axes[0][0].fill_between(years, paper_counts / 1000, alpha=0.15, color="steelblue")
    axes[0][0].set(xlabel="Year", ylabel="Papers/Month (×1000)",
                   title="Simulated arXiv AI+ML Paper Volume")
    axes[0][0].grid(True, alpha=0.3)

    # Topic trends
    for topic, vals in trends.items():
        axes[0][1].plot(trend_years, vals / vals.max(), lw=2, label=topic)
    axes[0][1].set(xlabel="Year", ylabel="Normalised Popularity",
                   title="AI Topic Popularity Trends")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(True, alpha=0.3)

    # Reading budget breakdown
    hours = {"Deep Read\n1 paper": 1, "Blog/Tutorial": 0.75, "Re-implementation": 1.5,
             "Digest Scan": 0.5, "Social Media": 0.33}
    axes[1][0].pie(list(hours.values()), labels=list(hours.keys()),
                    autopct="%1.0f%%", startangle=90,
                    colors=plt.cm.Set2.colors[:len(hours)])
    axes[1][0].set_title("Weekly Time Budget Distribution")

    # Cumulative readable papers
    total_readable = np.array([reading_schedule(int(c / 4), 0.5, 10)[0] * 52
                                for c in paper_counts])
    total_published = paper_counts * 12
    axes[1][1].plot(years, total_published / 1e3, lw=2, color="tomato", label="Published/yr (×1k)")
    axes[1][1].plot(years, total_readable / 1e3, lw=2, color="mediumseagreen",
                    linestyle="--", label="Readable/yr (×1k)")
    axes[1][1].set(xlabel="Year", ylabel="Papers (×1000)",
                   title="Published vs Readable per Year")
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "staying_current.png", dpi=100)
    plt.close()
    print("\n  Saved staying_current.png")


if __name__ == "__main__":
    demo()
