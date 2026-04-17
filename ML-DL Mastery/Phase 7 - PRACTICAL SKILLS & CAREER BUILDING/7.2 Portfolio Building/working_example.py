"""
Working Example: Portfolio Building
Covers project selection, GitHub best practices, writing about ML work,
and showcasing technical skills effectively.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_portfolio")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Project selection ──────────────────────────────────────────────────────
def project_selection():
    print("=== Portfolio Building ===")
    print()
    print("  Portfolio project tiers (by impact):")
    tiers = [
        ("Tier 1 (BEST)",  "Novel applied research; paper + code; real-world problem"),
        ("Tier 2",         "Competition top % placement with writeup + clean code"),
        ("Tier 3",         "End-to-end ML system deployed with documented results"),
        ("Tier 4",         "Original dataset + analysis + baseline models"),
        ("Tier 5 (OK)",    "Tutorial reimplementation with genuine extension"),
        ("Avoid",          "Titanic/MNIST/CIFAR-10 without meaningful twist"),
    ]
    for t, d in tiers:
        print(f"  {t:<18} {d}")
    print()
    print("  High-impact project ideas (2025):")
    ideas = [
        "Fine-tune an open LLM on a niche domain (legal, medical, coding style)",
        "Build a RAG system over a novel corpus with RAGAS evaluation",
        "Create a multi-agent system solving a concrete business problem",
        "Reproduce a recent paper (with credit) and extend it",
        "Build an end-to-end ML system with monitoring/drift detection",
        "Win or place top 10% in an active Kaggle competition",
        "Open-source a useful ML tool/library with documentation",
    ]
    for idea in ideas:
        print(f"  • {idea}")


# ── 2. GitHub best practices ──────────────────────────────────────────────────
def github_best_practices():
    print("\n=== GitHub Repository Best Practices ===")
    print()
    print("  Repository checklist:")
    checklist = [
        ("README.md",          "Problem, approach, results, how to run (with screenshots)"),
        ("requirements.txt",   "Pin exact versions for reproducibility"),
        ("Notebooks cleaned",  "Run all cells; clear obvious; add markdown explanations"),
        ("results/",           "Pre-computed metrics; graphs; model card"),
        ("data/README.md",     "Describe data source, size, access instructions"),
        ("tests/",             "At least data loading + model inference tests"),
        (".gitignore",         "Exclude data files, model weights, __pycache__"),
        ("LICENSE",            "MIT or Apache 2.0 for open source"),
        ("CONTRIBUTING.md",    "If you want others to contribute"),
    ]
    for c, d in checklist:
        print(f"  {'✓ ' + c:<24} {d}")
    print()
    print("  README structure:")
    readme_sections = [
        "## Problem Statement (1-2 sentences; why it matters)",
        "## Demo / Results (GIF or screenshot first; grab attention)",
        "## Approach (3-5 bullet points; key technical choices)",
        "## Results (table comparing your approach vs baseline)",
        "## Installation (copy-paste commands that actually work)",
        "## Usage (minimal working example)",
        "## Citation / References",
    ]
    for s in readme_sections:
        print(f"  {s}")


# ── 3. Writing about ML work ──────────────────────────────────────────────────
def writing_about_ml():
    print("\n=== Writing About ML Work ===")
    print()
    print("  Blog post structure (Medium/Substack/personal):")
    structure = [
        ("Hook",         "Striking result or problem; grab in first sentence"),
        ("Problem",      "Why does this matter? Who has this problem?"),
        ("Background",   "Minimal necessary context; link to papers"),
        ("Approach",     "What you tried; key decisions; why you chose this"),
        ("Results",      "Concrete numbers; graphs; ablations"),
        ("Lessons",      "What didn't work; what you'd do differently"),
        ("Call to action","GitHub link; follow for more; open question"),
    ]
    for s, d in structure:
        print(f"  {s:<16} {d}")
    print()
    print("  Common mistakes:")
    mistakes = [
        "Showing accuracy without baseline ('94% accuracy!' on MNIST)",
        "Too much theory, not enough results",
        "No visualisations (graphs >>> tables for readability)",
        "Reproducibility: not sharing code/data/model weights",
        "Writing for ML experts when your audience is hiring managers",
    ]
    for m in mistakes:
        print(f"  ✗ {m}")
    print()
    print("  Platforms:")
    platforms = [
        ("Medium",      "Largest ML audience; Medium Partner Program for $"),
        ("Substack",    "Newsletter format; build subscriber base"),
        ("Towards Data Science", "Curated; Medium publication; high reach"),
        ("Hugging Face Spaces", "Demo + model card; interactive"),
        ("arXiv",       "If it's genuinely research; preprint server"),
        ("Personal site","GitHub Pages / Notion / Hugo; full control"),
    ]
    for p, d in platforms:
        print(f"  {p:<26} {d}")


# ── 4. Project showcase template ─────────────────────────────────────────────
def showcase_template():
    print("\n=== Project Showcase Template ===")
    print()

    template = """
## [Project Name] — [One-line description]

### Problem
[1-2 sentences: what problem, who has it, why it's hard]

### My Approach
- [Key technical choice 1 + why]
- [Key technical choice 2 + why]
- [Key technical choice 3 + why]

### Results
| Metric    | Baseline  | My Model  | Improvement |
|-----------|-----------|-----------|-------------|
| Accuracy  | 72.3%     | 87.1%     | +14.8%      |
| Latency   | 340ms     | 28ms      | 12x faster  |

### What I Learned
- [Technical lesson 1]
- [Failure that taught me something]
- [What I'd do differently next time]

### Links
- GitHub: github.com/yourname/project
- Demo: spaces.huggingface.co/yourname/project
- Blog post: medium.com/@yourname/project
"""
    for line in template.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    project_selection()
    github_best_practices()
    writing_about_ml()
    showcase_template()
