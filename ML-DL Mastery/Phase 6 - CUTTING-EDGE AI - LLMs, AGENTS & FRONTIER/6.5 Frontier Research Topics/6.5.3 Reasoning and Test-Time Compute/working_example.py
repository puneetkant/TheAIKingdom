"""
Working Example: Reasoning and Test-Time Compute
Covers chain-of-thought reasoning, self-consistency, MCTS for LLMs,
o1/o3 architecture, and test-time scaling laws.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_reasoning")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Test-time compute overview ---------------------------------------------
def test_time_overview():
    print("=== Reasoning and Test-Time Compute ===")
    print()
    print("  Traditional scaling: more params + more data")
    print("  New scaling axis: more compute at INFERENCE time")
    print("  'Thinking longer' -> better answers for hard problems")
    print()
    print("  Two orthogonal scaling curves:")
    print("  1. Pre-training compute  -> base capability")
    print("  2. Test-time compute     -> reasoning depth")
    print()
    approaches = [
        ("Chain-of-thought",    "Prompt: think step by step -> linear reasoning trace"),
        ("Self-consistency",    "K diverse CoTs -> majority vote; reliable but Kxcost"),
        ("Best-of-N sampling",  "Sample N answers; pick highest reward model score"),
        ("MCTS / tree search",  "Branch + evaluate; backtrack; finds optimal path"),
        ("Process rewards",     "PRM: score each reasoning step, not just final answer"),
        ("Extended thinking",   "o1/o3: RL-trained to generate internal CoT"),
        ("Budget forcing",      "Control thinking token budget; accuracy vs cost tradeoff"),
    ]
    print(f"  {'Approach':<22} {'Description'}")
    for a, d in approaches:
        print(f"  {a:<22} {d}")


# -- 2. Self-consistency -------------------------------------------------------
def self_consistency_demo():
    print("\n=== Self-Consistency Demo ===")
    print()
    print("  Sample K reasoning paths -> take majority vote on final answer")
    print()

    rng = np.random.default_rng(42)

    # Simulate a math problem with error rate
    def sample_answer(true_answer: int, error_rate: float):
        if rng.random() < error_rate:
            return true_answer + rng.integers(-3, 4)
        return true_answer

    true_answer = 42
    error_rate  = 0.35

    print(f"  True answer: {true_answer}")
    print(f"  Per-sample error rate: {error_rate:.0%}")
    print()
    print(f"  {'K':<6} {'Greedy accuracy':>18} {'SC@K accuracy':>15}")

    for K in [1, 3, 5, 10, 20, 40]:
        greedy_acc = (1 - error_rate)
        # Run 10000 trials of majority voting
        trials = 10000
        sc_correct = 0
        for _ in range(trials):
            answers = [sample_answer(true_answer, error_rate) for _ in range(K)]
            vals, counts = np.unique(answers, return_counts=True)
            majority = vals[np.argmax(counts)]
            if majority == true_answer:
                sc_correct += 1
        sc_acc = sc_correct / trials
        print(f"  {K:<6} {greedy_acc:>17.1%} {sc_acc:>14.1%}")


# -- 3. Process reward models --------------------------------------------------
def process_reward_models():
    print("\n=== Process Reward Models (PRMs) ===")
    print()
    print("  Outcome Reward Model (ORM): score only final answer")
    print("  Process Reward Model (PRM): score each reasoning step")
    print()
    print("  Why PRMs matter:")
    print("  - Dense feedback signal speeds up RL training")
    print("  - Enable beam search over reasoning steps")
    print("  - Detect where reasoning went wrong (credit assignment)")
    print()
    print("  PRM800K (OpenAI dataset): 800K step-level labels")
    print("  Math-Shepherd: automatic PRM labels via process verification")
    print()

    # Simulate a reasoning trace with step scores
    trace = [
        ("Step 1: Let x = total apples.",                   0.98),
        ("Step 2: Alice has 15, Bob has x - 15.",            0.95),
        ("Step 3: Bob gives 5 to Carol, so Bob has x-20.",   0.82),
        ("Step 4: 3 * (x-20) = x (Carol triples Bob's).",   0.61),  # error here
        ("Step 5: 3x - 60 = x -> 2x = 60 -> x = 30.",        0.40),
        ("Step 6: Answer = 30.",                             0.35),
    ]
    print("  Simulated PRM scores for a reasoning trace:")
    for step, score in trace:
        bar = "#" * int(score * 20)
        flag = " <- suspect" if score < 0.6 else ""
        print(f"  [{score:.2f}] {step[:55]:<55} {bar}{flag}")
    print()
    print("  Best-of-N with PRM:")
    print("  Generate N solutions, use PRM to score each step,")
    print("  pick solution with highest min-step score (pessimistic product)")


# -- 4. o1 / o3 architecture ---------------------------------------------------
def o1_architecture():
    print("\n=== OpenAI o1/o3 — Extended Thinking ===")
    print()
    print("  Key idea: train model with RL to generate long internal reasoning chains")
    print("  before outputting an answer (chain-of-thought as latent computation)")
    print()
    print("  Architecture observations:")
    obs = [
        "Model generates 'thinking tokens' (hidden from user in most interfaces)",
        "Thinking tokens budget is dynamic; scales with problem difficulty",
        "Uses MCTS-like search during training to find correct reasoning paths",
        "RL reward: final answer correctness (not intermediate steps)",
        "System card: o1-mini uses ~1k thinking tokens; o1 uses ~10k+",
    ]
    for o in obs:
        print(f"  • {o}")
    print()
    print("  Test-time compute scaling results:")
    models = [
        ("o1-mini",  "~1-3k thinking tokens; fast; math/code"),
        ("o1",       "~10k tokens; AIME 2024: 74%"),
        ("o1-pro",   "~50-100k tokens; extended; $15/query"),
        ("o3-mini",  "Budget modes: low/med/high thinking"),
        ("o3",       "AIME 2025: 96.7%; SWE-bench: 71.7%"),
    ]
    print(f"  {'Model':<12} {'Notes'}")
    for m, d in models:
        print(f"  {m:<12} {d}")
    print()
    print("  When to use extended thinking:")
    use_cases = [
        "Competition math and olympiad problems",
        "Complex multi-step code debugging",
        "Scientific reasoning requiring many deductions",
        "Tasks where accuracy >>> latency/cost",
    ]
    for uc in use_cases:
        print(f"  [OK] {uc}")
    print()
    print("  NOT worth it for:")
    not_worth = ["Simple factual queries", "Creative writing", "Translation", "Summarisation"]
    for n in not_worth:
        print(f"  [X] {n}")


if __name__ == "__main__":
    test_time_overview()
    self_consistency_demo()
    process_reward_models()
    o1_architecture()
