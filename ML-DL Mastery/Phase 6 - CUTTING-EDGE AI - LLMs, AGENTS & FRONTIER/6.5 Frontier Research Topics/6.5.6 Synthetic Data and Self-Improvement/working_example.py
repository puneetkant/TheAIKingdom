"""
Working Example: Synthetic Data and Self-Improvement
Covers data generation techniques, self-play, distillation pipelines,
and frontier self-improvement methods.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Why synthetic data ─────────────────────────────────────────────────────
def synthetic_data_overview():
    print("=== Synthetic Data and Self-Improvement ===")
    print()
    print("  Why synthetic data?")
    reasons = [
        ("Data scarcity",    "Real data exhausted; especially reasoning / code tasks"),
        ("Quality control",  "Filter to high quality; better than web crawl"),
        ("Privacy",          "Generate data without real personal information"),
        ("Diversity",        "Sample rare scenarios systematically"),
        ("Labelling cost",   "LLM annotation << human annotation cost"),
    ]
    for r, d in reasons:
        print(f"  {r:<18} {d}")
    print()
    print("  Notable synthetic data papers:")
    papers = [
        ("Alpaca",         "GPT-3 generates 52k instruction-following examples"),
        ("Orca",           "GPT-4 explanations for reasoning steps"),
        ("Phi-1",          "Textbooks + exercises; tiny but capable"),
        ("Magpie",         "LLaMA generates its own instructions from BOS prompt"),
        ("WizardLM",       "Evol-Instruct: prompt evolves into harder versions"),
        ("MetaMath",       "Math augmentation: rephrase, reverse, FOBAR questions"),
        ("Nemotron-4",     "NVIDIA; 98% synthetic training data"),
    ]
    for p, d in papers:
        print(f"  {p:<14} {d}")


# ── 2. Instruction evolution ──────────────────────────────────────────────────
def instruction_evolution():
    print("\n=== Instruction Evolution (Evol-Instruct) ===")
    print()
    print("  WizardLM: take simple instructions → evolve into harder, more complex ones")
    print()
    print("  Evolution operations:")
    ops = [
        ("Add constraints",    "Add additional requirements to the original instruction"),
        ("Deepening",          "Ask about more specific aspects of the topic"),
        ("Concretising",       "Replace general concepts with specific examples"),
        ("Increase reasoning", "Add steps requiring multi-step reasoning"),
        ("Breadth",            "Create a sibling task (new, related instruction)"),
    ]
    for op, d in ops:
        print(f"  {op:<22} {d}")
    print()

    # Simulate evolution chain
    print("  Simulated evolution chain (2 steps):")
    v0 = "Write a Python function to sort a list."
    v1 = "Write a Python function to sort a list of dictionaries by a given key, handling ties stably."
    v2 = ("Write a Python function that sorts a list of employee records (dicts with 'name', 'salary', 'dept') "
          "by salary descending, breaking ties alphabetically by name, "
          "without using Python's built-in sort stability — implement your own merge sort.")
    for i, ver in enumerate([v0, v1, v2]):
        print(f"  v{i}: {ver[:100]}")


# ── 3. Self-play and iterative self-improvement ───────────────────────────────
def self_improvement():
    print("\n=== Self-Play and Iterative Self-Improvement ===")
    print()
    print("  SPIN (Self-Play Fine-Tuning):")
    print("    Generator tries to fool Discriminator (same initial model)")
    print("    Discriminator distinguishes real vs generated responses")
    print("    Both improve iteratively without new human labels")
    print()
    print("  STaR (Self-Taught Reasoner):")
    print("    1. Generate reasoning trace → check answer")
    print("    2. Correct: add to training; Incorrect: provide hint → re-try")
    print("    3. Fine-tune on correct traces → stronger model for next round")
    print()

    # Simulate STaR improvement curve
    rng = np.random.default_rng(0)
    rounds = 6
    base_acc = 0.30
    print(f"  Simulated STaR accuracy over iterations:")
    print(f"  {'Round':<8} {'Accuracy':>10} {'Δ':>8}")
    prev = 0.0
    for r in range(rounds):
        gain = (0.70 - base_acc) * (1 - np.exp(-0.7 * r)) + rng.normal(0, 0.01)
        acc  = base_acc + gain
        delta = acc - prev if r > 0 else 0
        print(f"  {r:<8} {acc:>9.1%} {delta:>+7.1%}")
        prev = acc
    print()
    print("  Rejection sampling (RS) fine-tuning:")
    print("    Generate K solutions per problem; keep only correct ones")
    print("    Fine-tune on correct solutions; iterate")


# ── 4. Distillation pipelines ─────────────────────────────────────────────────
def distillation_pipelines():
    print("\n=== Distillation and Data Pipelines ===")
    print()
    print("  Frontier-to-small distillation:")
    print("    Large teacher (GPT-4o) generates diverse high-quality completions")
    print("    Small student fine-tuned on teacher's outputs")
    print("    Often beats same-size model trained on human data alone")
    print()
    pipelines = [
        ("φ-1, Phi-2, Phi-3",    "Microsoft; textbook quality + ChatGPT outputs"),
        ("Gemma-2",              "Google; knowledge distilled from Gemini Pro"),
        ("Zephyr",               "dDPO + UltraFeedback (GPT-4 preferences)"),
        ("OpenHermes",           "Teknium; GPT-4 tool-use + reasoning traces"),
        ("Dolphin",              "Uncensored Mixtral distilled from GPT-4"),
    ]
    print(f"  {'Model':<22} {'Notes'}")
    for m, d in pipelines:
        print(f"  {m:<22} {d}")
    print()
    print("  Quality filters:")
    filters = [
        "Perplexity filtering (remove fluency outliers)",
        "Deduplication (MinHash / exact hash; >10% duplicates common)",
        "Classifier-based quality scoring (trained on human judgements)",
        "IFD score (instruction-following difficulty; keep hard ones)",
        "Response length distribution normalisation",
    ]
    for f in filters:
        print(f"  ✓ {f}")


if __name__ == "__main__":
    synthetic_data_overview()
    instruction_evolution()
    self_improvement()
    distillation_pipelines()
