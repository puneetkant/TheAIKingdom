"""
Working Example: Research Skills
Covers how to read papers efficiently, find research, stay current,
and contribute to the ML research community.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_research")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. How to read ML papers --------------------------------------------------
def how_to_read_papers():
    print("=== Research Skills ===")
    print()
    print("  The Three-Pass Method (Keshav 2007, adapted for ML):")
    print()
    passes = [
        ("Pass 1 (5-10 min)",
         ["Read: title, abstract, intro, section headings, conclusion",
          "Question: What problem? What's the key idea? New vs related?",
          "Decision: read further? cite? implement?"]),
        ("Pass 2 (30-60 min)",
         ["Read figures and tables carefully (often the whole story)",
          "Read methods; understand architecture/algorithm",
          "Note: equations, key results, limitations",
          "Skip: proofs, appendices (unless relevant)"]),
        ("Pass 3 (2-4 hours)",
         ["Re-implement from scratch (best understanding)",
          "Critically evaluate every claim",
          "Check citations; understand prior work",
          "Think: what would I do differently?"]),
    ]
    for name, steps in passes:
        print(f"  {name}:")
        for s in steps:
            print(f"    • {s}")
        print()


# -- 2. Paper discovery --------------------------------------------------------
def paper_discovery():
    print("=== Paper Discovery ===")
    print()
    print("  Primary sources:")
    sources = [
        ("arXiv.org",        "cs.LG / cs.AI / cs.CL; preprint server; daily papers"),
        ("Papers with Code",  "Papers + GitHub repos + benchmarks; trending"),
        ("Semantic Scholar",  "Free; citation graph; AI-powered search"),
        ("Google Scholar",    "Citation counts; author pages; alerts"),
        ("Connected Papers",  "Visual citation graph; find related work"),
        ("huggingface.co/papers", "AI paper digest; model + paper links"),
    ]
    for s, d in sources:
        print(f"  {s:<26} {d}")
    print()
    print("  Conference venues (tier 1):")
    conferences = [
        ("NeurIPS",   "December; flagship ML conference"),
        ("ICML",      "July; theoretical + applied ML"),
        ("ICLR",      "May; deep learning; open review"),
        ("CVPR",      "June; computer vision"),
        ("ACL/EMNLP", "NLP; July/December"),
        ("ICCV/ECCV", "Vision; biennial"),
    ]
    for c, d in conferences:
        print(f"  {c:<10} {d}")
    print()
    print("  Staying on top of arXiv:")
    print("    1. Follow @karpathy, @ylecun, @AnthropicAI on X/LinkedIn")
    print("    2. Subscribe to daily arXiv digest services (e.g. arxiv-sanity)")
    print("    3. AK (@_akhaliq) curates top daily papers on HuggingFace")
    print("    4. Use Google Scholar alerts for specific topics/authors")


# -- 3. Reproducing papers -----------------------------------------------------
def reproducing_papers():
    print("\n=== Reproducing Papers ===")
    print()
    print("  Why reproduce?")
    print("  • Deepest form of understanding")
    print("  • Builds credibility and portfolio")
    print("  • Reveals what actually matters (ablations)")
    print()
    print("  Reproduction checklist:")
    steps = [
        "1. Read paper 3 times (abstract, method, results)",
        "2. Find official code (GitHub, Papers With Code)",
        "3. Start with smallest possible experiment",
        "4. Match paper exactly first, then explore variations",
        "5. Document differences between your results and paper",
        "6. Write a blog post: 'Notes on reproducing [Paper]'",
    ]
    for s in steps:
        print(f"  {s}")
    print()
    print("  Common reproduction failures:")
    failures = [
        ("Different random seeds",    "Always report mean±std over 3-5 runs"),
        ("Data preprocessing gap",    "Often underdescribed; check supplements"),
        ("Hyperparameter secrets",    "Authors often tune more than reported"),
        ("Hardware differences",      "BF16 vs FP32 can change results"),
        ("Undisclosed techniques",    "Learning rate warmup, gradient clip, etc."),
    ]
    for f, d in failures:
        print(f"  {f:<26} {d}")


# -- 4. Writing research -------------------------------------------------------
def writing_research():
    print("\n=== Writing Research (Papers) ===")
    print()
    print("  ML paper structure:")
    sections = [
        ("Abstract",    "4-sentence formula: motivation, method, results, conclusion"),
        ("Introduction","Problem, why it's hard, our contribution (bullet list)"),
        ("Related Work","Honest comparison; make clear why you're different"),
        ("Method",      "Your algorithm/architecture; clear enough to reproduce"),
        ("Experiments", "Baselines; ablations; error analysis"),
        ("Conclusion",  "Honest limitations; future work; broad impact"),
    ]
    for s, d in sections:
        print(f"  {s:<16} {d}")
    print()
    print("  Figure quality:")
    fig_tips = [
        "Every paper needs 1 great teaser figure (intro/abstract)",
        "Use vectorised formats (PDF/SVG) for crisp plots",
        "Consistent colour palette and font sizes",
        "Caption: describe what it shows, not just what it is",
        "Ablation table: rows = components, columns = metric",
    ]
    for t in fig_tips:
        print(f"  • {t}")
    print()
    print("  LaTeX / tools:")
    tools = [
        ("Overleaf",     "Collaborative LaTeX editor; free tier sufficient"),
        ("arXiv Latex",  "Use arxiv template (article class; natbib)"),
        ("TikZ",         "High-quality figures in LaTeX"),
        ("Matplotlib",   "Python plots; set figsize and DPI; save as PDF"),
        ("Grammarly",    "Grammar check before submission"),
    ]
    for t, d in tools:
        print(f"  {t:<16} {d}")


if __name__ == "__main__":
    how_to_read_papers()
    paper_discovery()
    reproducing_papers()
    writing_research()
