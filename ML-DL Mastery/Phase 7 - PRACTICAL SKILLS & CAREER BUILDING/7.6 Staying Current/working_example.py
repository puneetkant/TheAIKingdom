"""
Working Example: Staying Current in AI/ML
Covers how to follow research, build a learning system,
and maintain expertise in a fast-moving field.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_staying_current")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Information sources ----------------------------------------------------
def information_sources():
    print("=== Staying Current in AI/ML ===")
    print()
    print("  Daily sources:")
    daily = [
        ("arXiv cs.LG/cs.CL/cs.CV",  "Primary research; new papers daily"),
        ("Hugging Face Daily Papers",  "Curated top papers; model links"),
        ("X/Twitter @_akhaliq",       "Best ML paper curator; daily digest"),
        ("X @karpathy",               "Andrej Karpathy; high-signal commentary"),
        ("X @AnthropicAI / @OpenAI",  "Official announcements"),
        ("AI newsletters",            "The Batch (Andrew Ng), Import AI (Jack Clark)"),
    ]
    for s, d in daily:
        print(f"  {s:<34} {d}")
    print()
    print("  Weekly sources:")
    weekly = [
        ("Papers with Code Trending",  "Top papers + code; sorted by stars"),
        ("Reddit r/MachineLearning",   "Community discussion; new papers"),
        ("AI Supremacy newsletter",    "Comprehensive weekly roundup"),
        ("Nathan.ai newsletter",       "Non-technical; AI industry news"),
        ("Last Week in AI (podcast)",  "Audio summary of the week"),
    ]
    for s, d in weekly:
        print(f"  {s:<34} {d}")


# -- 2. Learning system --------------------------------------------------------
def learning_system():
    print("\n=== Building a Personal Learning System ===")
    print()
    print("  The 4-layer filter:")
    layers = [
        ("Layer 1: Scan",   "Daily: titles + abstracts (15 min); 95% filtered out"),
        ("Layer 2: Read",   "Weekly: full read of 3-5 relevant papers (2h)"),
        ("Layer 3: Study",  "Monthly: deep dive into 1 seminal paper (4h+)"),
        ("Layer 4: Build",  "Quarterly: implement 1 paper or build 1 project"),
    ]
    for l, d in layers:
        print(f"  {l:<18} {d}")
    print()
    print("  Note-taking systems for ML:")
    systems = [
        ("Obsidian",     "Knowledge graph; bidirectional links; local Markdown"),
        ("Notion",       "Database + wiki; team-shareable"),
        ("Roam Research","Networked thought; daily notes"),
        ("Zotero",       "Reference manager; PDF annotation; citation export"),
        ("Logseq",       "Open-source Roam; local first; free"),
    ]
    for s, d in systems:
        print(f"  {s:<16} {d}")
    print()
    print("  Paper note template:")
    template = [
        "## Paper: [Title] (Year)",
        "### 1-sentence summary",
        "### Key contribution (what's new?)",
        "### Method (how does it work?)",
        "### Results (numbers that matter)",
        "### Limitations (honest critique)",
        "### Connection to my work",
        "### Open questions",
    ]
    for t in template:
        print(f"  {t}")


# -- 3. Podcasts, courses, communities -----------------------------------------
def podcasts_and_communities():
    print("\n=== Podcasts, Courses, and Communities ===")
    print()
    print("  Podcasts:")
    podcasts = [
        ("Lex Fridman",       "Long-form; AI/CS/science; very popular"),
        ("80,000 Hours",      "AI safety; career; deep conversations"),
        ("The TWIML AI Podcast","Practical ML; Sam Charrington"),
        ("Gradient Dissent",  "Weights & Biases; practitioners"),
        ("Practical AI",      "Developer-focused; weekly"),
    ]
    for p, d in podcasts:
        print(f"  {p:<24} {d}")
    print()
    print("  Online communities:")
    communities = [
        ("Hugging Face Discord",  "50k+ members; NLP/LLMs; active"),
        ("EleutherAI Discord",    "Open-source LLM research; technical"),
        ("r/MachineLearning",     "Reddit; paper discussions; career"),
        ("ML Collective",         "Independent researchers; workshops"),
        ("MLOps Community",       "Slack; production ML practitioners"),
        ("Interconnects (substack)","Nathan Lambert; RL/alignment research"),
    ]
    for c, d in communities:
        print(f"  {c:<26} {d}")
    print()
    print("  Free courses to reference:")
    courses = [
        ("fast.ai",          "Jeremy Howard; practical deep learning; excellent"),
        ("CS229 (Stanford)", "Andrew Ng; ML fundamentals; free online"),
        ("CS231n (Stanford)","ConvNets; computer vision; free"),
        ("Deeplearning.ai",  "Andrew Ng; specialisations; paid but worthwhile"),
        ("Andrej Karpathy",  "YouTube; build nanoGPT from scratch; legendary"),
    ]
    for c, d in courses:
        print(f"  {c:<22} {d}")


# -- 4. Personal knowledge base ------------------------------------------------
def knowledge_base():
    print("\n=== Personal Knowledge Base Workflow ===")
    print()
    print("  Weekly review ritual (30 min, Sunday):")
    review = [
        "1. Process paper queue: triage notes from week",
        "2. Update 'Current Research Landscape' doc",
        "3. Note 1 thing I learned well this week",
        "4. Note 1 gap to address next week",
        "5. Tag connections between new and old notes",
    ]
    for r in review:
        print(f"  {r}")
    print()
    print("  Annual review:")
    annual = [
        "What were the 5 biggest AI developments?",
        "What skills became less valuable?",
        "What skills became more valuable?",
        "What's in my portfolio vs what I want?",
        "Set 3 specific technical goals for next year",
    ]
    for a in annual:
        print(f"  • {a}")
    print()
    print("  Warning signs you're falling behind:")
    warnings = [
        "You can't name the top models released in the last 6 months",
        "You've never used the library everyone is talking about",
        "Your go-to answer hasn't changed in 2 years",
        "Conference names don't sound familiar",
    ]
    for w in warnings:
        print(f"  [!] {w}")


if __name__ == "__main__":
    information_sources()
    learning_system()
    podcasts_and_communities()
    knowledge_base()
