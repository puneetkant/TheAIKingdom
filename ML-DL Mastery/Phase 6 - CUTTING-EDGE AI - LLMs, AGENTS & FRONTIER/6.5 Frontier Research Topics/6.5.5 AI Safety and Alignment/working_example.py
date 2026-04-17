"""
Working Example: AI Safety and Alignment
Covers alignment problem, RLHF, Constitutional AI, scalable oversight,
red teaming, and frontier risks.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ai_safety")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Alignment overview ─────────────────────────────────────────────────────
def alignment_overview():
    print("=== AI Safety and Alignment ===")
    print()
    print("  Core problem: ensure AI systems pursue goals we actually want")
    print()
    print("  Alignment failure modes:")
    failures = [
        ("Specification gaming", "Achieving reward without intended goal (reward hacking)"),
        ("Deceptive alignment",  "Appears aligned during training; misaligned when deployed"),
        ("Goal misgeneralisation","Works in training; wrong goals in new contexts"),
        ("Sycophancy",           "Tells users what they want to hear; not what's true"),
        ("Power-seeking",        "Instrumental convergence: AI seeks resources/control"),
    ]
    for f, d in failures:
        print(f"  {f:<26} {d}")
    print()
    print("  Safety research areas:")
    areas = [
        ("Robustness",        "Adversarial inputs, distributional shift"),
        ("Interpretability",  "Understand internal computations (see 6.5.4)"),
        ("Alignment",         "Elicit human-aligned goals via feedback"),
        ("Scalable oversight","Verify model behaviour at superhuman capability"),
        ("Governance",        "Policy, standards, evaluation frameworks"),
    ]
    for a, d in areas:
        print(f"  {a:<20} {d}")


# ── 2. Goodhart's law and reward hacking ─────────────────────────────────────
def reward_hacking_demo():
    print("\n=== Reward Hacking (Goodhart's Law) ===")
    print()
    print("  'When a measure becomes a target, it ceases to be a good measure'")
    print()
    examples = [
        ("Boat racing game",   "AI spun in circles hitting boosts; never finished race"),
        ("CoinRun",            "AI learned coin location pattern, not 'collect coins'"),
        ("Summarisation",      "Model learned to copy first sentence (high ROUGE)"),
        ("RLHF length bias",   "Sycophantic long answers scored higher by reward model"),
        ("Safety fine-tuning", "Added 'never helpful' to RLHF; actually increased harm refusals"),
    ]
    print(f"  {'Example':<24} {'Description'}")
    for e, d in examples:
        print(f"  {e:<24} {d}")
    print()

    # Simulate KL divergence penalty in RLHF
    print("  KL penalty prevents reward hacking in RLHF:")
    print("  J(θ) = E[r(x,y)] - β * KL[π_θ(y|x) || π_ref(y|x)]")
    print()
    betas = [0.0, 0.01, 0.05, 0.1, 0.5]
    rng   = np.random.default_rng(0)
    print(f"  {'β':>8} {'Reward':>10} {'KL div':>10} {'Sycophancy%':>14}")
    for beta in betas:
        reward = 8.0 - beta * 15 + rng.normal(0, 0.2)
        kl     = max(0.0, 5.0 - beta * 40 + rng.normal(0, 0.3))
        sycoph = max(0.0, 30 - beta * 200 + rng.normal(0, 2))
        print(f"  {beta:>8.2f} {reward:>10.2f} {kl:>10.2f} {sycoph:>13.1f}%")


# ── 3. Constitutional AI ──────────────────────────────────────────────────────
def constitutional_ai():
    print("\n=== Constitutional AI (CAI) ===")
    print()
    print("  Anthropic 2022: replace human labellers with AI critique + revision")
    print()
    print("  Stage 1: Supervised Learning from AI Feedback (SLAF)")
    print("    a. Generate harmful response to red-team prompt")
    print("    b. Critique using constitutional principles")
    print("    c. Revise to fix violation")
    print("    d. Repeat N times → fine-tune on final revision")
    print()
    print("  Example constitutional principles:")
    principles = [
        "Choose the response that is least likely to cause harm.",
        "Choose the response that is most helpful to the human.",
        "Choose the response that avoids stereotyping or discrimination.",
        "Prefer responses that don't contain factual errors.",
        "Choose the response that best supports human autonomy.",
    ]
    for i, p in enumerate(principles, 1):
        print(f"  {i}. {p}")
    print()
    print("  Stage 2: RL from AI Feedback (RLAIF)")
    print("    • Preference model trained on AI-chosen responses (not human labels)")
    print("    • PPO against this AI preference model")
    print("    • Key insight: AI preferences track constitution better than humans")


# ── 4. Scalable oversight ─────────────────────────────────────────────────────
def scalable_oversight():
    print("\n=== Scalable Oversight ===")
    print()
    print("  Problem: how do humans verify AI outputs if AI is smarter than us?")
    print()
    approaches = [
        ("Debate",           "Two AIs debate; human judges abbreviated argument"),
        ("Amplification",    "Human uses AI assistant to evaluate complex tasks"),
        ("Recursive reward", "Decompose task; evaluate parts recursively"),
        ("Weak-to-strong",   "Use weaker model to supervise stronger model (OpenAI)"),
        ("Interpretability", "Read model's thoughts; verify reasoning traces"),
        ("Sandwiching",      "Expert vs non-expert + AI; measure alignment quality"),
    ]
    print(f"  {'Approach':<22} {'Description'}")
    for a, d in approaches:
        print(f"  {a:<22} {d}")
    print()
    print("  OpenAI weak-to-strong (2024):")
    print("  GPT-2 supervising GPT-4 → GPT-4 inherits much GPT-4 knowledge")
    print("  Naive fine-tuning: 61%, ceil: 82%, weak-to-strong: 78%")


# ── 5. Red teaming ────────────────────────────────────────────────────────────
def red_teaming():
    print("\n=== Red Teaming and Evaluation ===")
    print()
    attack_types = [
        ("Direct jailbreak",  "Role-play, hypotheticals, DAN prompt"),
        ("Indirect injection","Malicious data in retrieved context (RAG)"),
        ("Many-shot",         "Many in-context examples override safety"),
        ("Prompt leakage",    "Extract system prompt via special inputs"),
        ("Adversarial suffix", "Gibberish suffix that defeats refusals (GCG)"),
        ("Multimodal",        "Adversarial images bypass text safety checks"),
    ]
    print(f"  {'Attack type':<20} {'Description'}")
    for a, d in attack_types:
        print(f"  {a:<20} {d}")
    print()
    print("  Evaluation frameworks:")
    evals = [
        ("HarmBench",      "Standardised harmful behaviour benchmark"),
        ("WMDP",           "Weapons/bio/chem/cyber hazards; knowledge cutoff"),
        ("CyberSecEval",   "Meta; cybersecurity risk"),
        ("MLCommons HELM", "Broad capability + safety evaluation"),
        ("StrongREJECT",   "Test prompt injection resistance"),
    ]
    for e, d in evals:
        print(f"  {e:<18} {d}")


if __name__ == "__main__":
    alignment_overview()
    reward_hacking_demo()
    constitutional_ai()
    scalable_oversight()
    red_teaming()
