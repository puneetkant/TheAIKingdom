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


def demo_conference_calendar():
    """Show top AI conference submission deadlines and their prestige tiers."""
    print("\n=== AI Conference Calendar ===")
    conferences = [
        {"name": "NeurIPS",  "tier": 1, "acceptance": 25, "month_submit": 5,  "month_notify": 9},
        {"name": "ICML",     "tier": 1, "acceptance": 27, "month_submit": 1,  "month_notify": 5},
        {"name": "ICLR",     "tier": 1, "acceptance": 30, "month_submit": 10, "month_notify": 1},
        {"name": "CVPR",     "tier": 1, "acceptance": 25, "month_submit": 11, "month_notify": 3},
        {"name": "EMNLP",    "tier": 2, "acceptance": 22, "month_submit": 6,  "month_notify": 9},
        {"name": "AAAI",     "tier": 2, "acceptance": 23, "month_submit": 8,  "month_notify": 12},
        {"name": "COLM",     "tier": 2, "acceptance": 30, "month_submit": 2,  "month_notify": 5},
    ]
    print(f"  {'Conference':10s} {'Tier':6s} {'Acceptance %':14s} {'Submit':8s} {'Notify':8s}")
    for c in conferences:
        print(f"  {c['name']:10s} {'Top' if c['tier']==1 else 'Strong':6s} {c['acceptance']:14d}%"
              f" {c['month_submit']:8d} {c['month_notify']:8d}")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if c["tier"]==1 else "#3498db" for c in conferences]
    bars = ax.bar([c["name"] for c in conferences],
                  [c["acceptance"] for c in conferences], color=colors, edgecolor="white")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="Tier 1"), Patch(color="#3498db", label="Tier 2")])
    ax.set(ylabel="Acceptance Rate (%)", title="AI Conference Acceptance Rates")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "conference_calendar.png", dpi=100); plt.close()
    print("  Saved conference_calendar.png")


def demo_knowledge_decay():
    """Show how fast ML knowledge becomes outdated (half-life model)."""
    print("\n=== Knowledge Decay in ML ===")
    topics = {
        "Deep Learning basics": 5.0,    # half-life in years
        "Transformer details":  2.0,
        "Specific LLM APIs":    0.5,
        "Python fundamentals":  20.0,
        "Optimization theory":  15.0,
        "Latest SOTA benchmarks": 0.25,
    }
    t = np.linspace(0, 5, 100)
    plt.figure(figsize=(8, 4))
    for topic, half_life in topics.items():
        relevance = 0.5 ** (t / half_life)
        plt.plot(t, relevance * 100, lw=2, label=f"{topic} (t1/2={half_life}yr)")
        print(f"  {topic:32s}: {relevance[-1]*100:.1f}% relevance after 5 years")
    plt.xlabel("Years since learned"); plt.ylabel("Relevance (%)")
    plt.title("ML Knowledge Half-Life")
    plt.legend(fontsize=7); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "knowledge_decay.png", dpi=100); plt.close()
    print("  Saved knowledge_decay.png")


if __name__ == "__main__":
    demo()
    demo_conference_calendar()
    demo_knowledge_decay()
