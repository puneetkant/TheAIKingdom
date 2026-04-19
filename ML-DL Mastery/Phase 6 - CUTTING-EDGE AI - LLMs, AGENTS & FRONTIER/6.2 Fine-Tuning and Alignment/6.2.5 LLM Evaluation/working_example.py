"""
Working Example: LLM Evaluation
Covers benchmarks, human evaluation, automated LLM-as-judge,
safety evaluation, and evals best practices.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_llm_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Benchmark overview -----------------------------------------------------
def benchmark_overview():
    print("=== LLM Evaluation Methods ===")
    print()
    print("  Benchmark taxonomy:")
    print()
    benchmarks = [
        ("Knowledge",      "MMLU (57 subjects), ARC, TriviaQA, NaturalQuestions"),
        ("Reasoning",      "GSM8K, MATH, BIG-Bench Hard, LogiQA"),
        ("Coding",         "HumanEval, MBPP, SWE-bench, LiveCodeBench"),
        ("Instruction",    "MT-Bench, AlpacaEval 2, IFEval (strict)"),
        ("Safety",         "ToxiGen, BBQ (bias), TruthfulQA, HarmBench"),
        ("Long context",   "RULER, ZeroSCROLLS, HELMET, L-Eval"),
        ("Multimodal",     "MMMU, MMBench, DocVQA, ChartQA"),
        ("Agentic",        "AgentBench, WebArena, OSWorld, SWE-bench"),
    ]
    for cat, examples in benchmarks:
        print(f"  {cat:<14} {examples}")


# -- 2. Automated metrics ------------------------------------------------------
def automated_metrics():
    print("\n=== Automated Evaluation Metrics ===")
    print()

    # Exact match
    def exact_match(pred, gold):
        return int(pred.strip().lower() == gold.strip().lower())

    # Contains match
    def contains_match(pred, gold):
        return int(gold.strip().lower() in pred.strip().lower())

    # ROUGE-1 (simplified)
    def rouge1(pred, gold):
        p_toks = set(pred.lower().split())
        g_toks = set(gold.lower().split())
        overlap = len(p_toks & g_toks)
        if not overlap: return 0.0
        prec = overlap / len(p_toks)
        rec  = overlap / len(g_toks)
        return 2 * prec * rec / (prec + rec)

    examples = [
        ("Paris",      "Paris"),
        ("The capital of France is Paris.", "Paris"),
        ("Berlin",     "Paris"),
    ]
    print(f"  {'Prediction':<38} {'Gold':<8} {'EM'} {'Contains'} {'ROUGE-1'}")
    for pred, gold in examples:
        em = exact_match(pred, gold)
        cm = contains_match(pred, gold)
        r1 = rouge1(pred, gold)
        print(f"  {pred:<38} {gold:<8} {em}   {cm}        {r1:.3f}")
    print()

    print("  Other automated metrics:")
    metrics = [
        ("BERTScore",       "Semantic similarity via BERT embeddings"),
        ("BLEU",            "N-gram precision; translation; less used for generation"),
        ("Perplexity",      "Inverse probability on held-out text; measures fluency"),
        ("Pass@k",          "Code: pass at least 1 of k samples; HumanEval"),
        ("F1 (span)",       "QA: token F1 between prediction and gold span"),
    ]
    for m, d in metrics:
        print(f"  {m:<20} {d}")


# -- 3. LLM-as-judge -----------------------------------------------------------
def llm_as_judge():
    print("\n=== LLM-as-Judge Evaluation ===")
    print()
    print("  Use a capable LLM (e.g. GPT-4) to evaluate responses")
    print()
    print("  Formats:")
    formats = [
        ("Pairwise",   "Which response A or B is better? -> win rate"),
        ("Absolute",   "Rate on 1-10 scale -> mean score"),
        ("Rubric",     "Structured criteria; helpfulness/accuracy/safety"),
        ("Reference",  "Compare to gold reference; factual alignment"),
    ]
    for f, d in formats:
        print(f"  {f:<12} {d}")
    print()
    print("  Example judge prompt (MT-Bench):")
    judge_prompt = '''
[System]
You are an impartial AI assistant evaluator. Rate the assistant's response
on a scale of 1-10 based on helpfulness, accuracy, and clarity.

[User Question]
What is the difference between supervised and unsupervised learning?

[Assistant Response]
Supervised learning uses labelled data to train models to make predictions.
Unsupervised learning finds patterns in unlabelled data.

[Evaluation]
Score: 7/10
Reason: Correct but superficial. Could include examples and key algorithms.
'''
    print(judge_prompt)
    print("  Biases to be aware of:")
    biases = [
        ("Position bias",   "Prefers answer A if shown first"),
        ("Verbosity bias",  "Prefers longer, more detailed responses"),
        ("Self-enhancement","GPT-4 prefers GPT-4-like responses"),
        ("Calibration",     "May not use full 1-10 scale"),
    ]
    for b, d in biases:
        print(f"  {b:<18} {d}")


# -- 4. Human evaluation -------------------------------------------------------
def human_evaluation():
    print("\n=== Human Evaluation ===")
    print()
    print("  Chatbot Arena (LMSYS): crowd-sourced ELO rating")
    print("    Users submit prompts; see two anonymous responses; vote better")
    print("    Gold standard for user preference; hard to game")
    print()

    # Simulate ELO calculation
    def elo_update(r_a, r_b, result, K=32):
        """result: 1=A wins, 0=B wins, 0.5=tie"""
        E_a = 1 / (1 + 10**((r_b - r_a)/400))
        r_a_new = r_a + K * (result - E_a)
        r_b_new = r_b + K * ((1-result) - (1-E_a))
        return r_a_new, r_b_new

    models = {"GPT-4o": 1200, "Claude-3.5-Sonnet": 1190,
              "LLaMA-3.1-70B": 1150, "Mistral-7B": 1060}
    # Simulate 20 battles
    rng = np.random.default_rng(0)
    model_names = list(models.keys())
    ratings = dict(models)
    for _ in range(20):
        a, b = rng.choice(model_names, 2, replace=False)
        r_a, r_b = ratings[a], ratings[b]
        # Better-rated model wins with higher probability
        p_a_wins = 1 / (1 + 10**((r_b-r_a)/400))
        result = float(rng.random() < p_a_wins)
        ratings[a], ratings[b] = elo_update(r_a, r_b, result)

    print("  Simulated ELO after 20 battles:")
    for m, r in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"    {m:<26} ELO = {r:.0f}")

    print()
    print("  Evaluation best practices:")
    practices = [
        "Use held-out prompts; never evaluate on training distribution",
        "Report confidence intervals; evaluation is noisy",
        "Evaluate on task-relevant slices, not just aggregate",
        "Include adversarial examples and edge cases",
        "Track over time; models improve; baselines shift",
    ]
    for p in practices:
        print(f"  • {p}")


if __name__ == "__main__":
    benchmark_overview()
    automated_metrics()
    llm_as_judge()
    human_evaluation()
