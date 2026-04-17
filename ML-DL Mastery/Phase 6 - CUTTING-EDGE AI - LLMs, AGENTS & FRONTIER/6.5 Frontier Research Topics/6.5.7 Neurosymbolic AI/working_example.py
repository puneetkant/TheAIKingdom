"""
Working Example: Neurosymbolic AI
Covers the integration of neural networks with symbolic reasoning,
knowledge graphs, theorem proving, and constraint satisfaction.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_neurosym")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Overview ───────────────────────────────────────────────────────────────
def neurosym_overview():
    print("=== Neurosymbolic AI ===")
    print()
    print("  Goal: combine statistical learning (neural) with explicit reasoning (symbolic)")
    print()
    print("  Pros of neural:  generalise, handle noise, learn from data")
    print("  Pros of symbolic: interpretable, logically consistent, sample-efficient")
    print()
    print("  Integration spectrum:")
    levels = [
        ("Symbolic-neuro",      "Neural net inside a symbolic system (NSIL, NTP)"),
        ("Neuro-symbolic",      "Symbolic engine inside neural (AlphaGeometry)"),
        ("Neuro-then-symbolic", "Neural output → symbolic verification"),
        ("Neuro+symbolic",      "Parallel neural and symbolic streams (DNN+KB)"),
    ]
    for l, d in levels:
        print(f"  {l:<24} {d}")
    print()
    print("  Notable systems:")
    systems = [
        ("DeepMind AlphaGeometry", "LLM + symbolic geometry solver; olympiad proofs"),
        ("AlphaTensor",            "RL discovers matrix multiplication algorithms"),
        ("Neural Theorem Proving", "Lean / Isabelle + LLM proof search"),
        ("DeepProbLog",            "Probabilistic logic + neural predicates"),
        ("Scallop",                "Differentiable Datalog; logic as neural layer"),
        ("NeSy-EBM",               "Energy-based model satisfying symbolic constraints"),
        ("Graph Neural Theorem",   "GNN over proof state graphs"),
    ]
    for s, d in systems:
        print(f"  {s:<28} {d}")


# ── 2. Symbolic reasoning with Python ─────────────────────────────────────────
def symbolic_reasoning_demo():
    print("\n=== Symbolic Reasoning Demo ===")
    print()
    print("  Forward chaining (rule-based reasoning):")
    print()

    class KnowledgeBase:
        def __init__(self):
            self.facts = set()
            self.rules = []

        def add_fact(self, *facts):
            for f in facts:
                self.facts.add(f)

        def add_rule(self, conditions, conclusion):
            self.rules.append((set(conditions), conclusion))

        def reason(self, max_iter=20):
            """Forward chaining until fixpoint."""
            changed = True
            iterations = 0
            while changed and iterations < max_iter:
                changed = False
                for conditions, conclusion in self.rules:
                    if conditions.issubset(self.facts) and conclusion not in self.facts:
                        self.facts.add(conclusion)
                        print(f"    Derived: {conclusion}")
                        changed = True
                iterations += 1

    kb = KnowledgeBase()
    kb.add_fact("has_feathers", "can_fly", "warm_blooded", "lays_eggs")
    kb.add_rule(["has_feathers", "lays_eggs"],            "is_bird")
    kb.add_rule(["is_bird", "warm_blooded"],              "is_vertebrate")
    kb.add_rule(["is_bird", "can_fly"],                   "is_flying_bird")
    kb.add_rule(["is_vertebrate", "lays_eggs"],           "is_oviparous_vertebrate")

    print("  Facts:", sorted(kb.facts))
    print()
    print("  Inference chain:")
    kb.reason()
    print()
    print("  Final facts:", sorted(kb.facts))


# ── 3. LLM + symbolic verifier ────────────────────────────────────────────────
def llm_plus_verifier():
    print("\n=== LLM + Symbolic Verifier Pattern ===")
    print()
    print("  Pattern: LLM generates candidate solution → symbolic verifier checks correctness")
    print("  Retry until verified or budget exhausted")
    print()

    # Simulate code generation + unit test verification
    def symbolic_verify(code: str, tests: list) -> tuple[bool, str]:
        """Run unit tests as symbolic verifier."""
        namespace = {}
        try:
            exec(code, namespace)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        for test_code, expected in tests:
            try:
                result = eval(test_code, namespace)
                if result != expected:
                    return False, f"FAIL: {test_code} → {result!r}, expected {expected!r}"
            except Exception as e:
                return False, f"ERROR: {test_code} → {e}"
        return True, "All tests passed"

    problem = "Write a function is_prime(n) returning True if n is prime."
    tests = [
        ("is_prime(2)", True),
        ("is_prime(3)", True),
        ("is_prime(4)", False),
        ("is_prime(17)", True),
        ("is_prime(1)", False),
    ]

    # Simulate multiple LLM attempts
    attempts = [
        # Attempt 1: wrong (misses n=2)
        "def is_prime(n):\n    for i in range(2, n): return n % i != 0\n    return True",
        # Attempt 2: wrong edge case (missing n<2)
        "def is_prime(n):\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        # Attempt 3: correct
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
    ]

    print(f"  Problem: {problem}")
    for i, code in enumerate(attempts):
        ok, msg = symbolic_verify(code, tests)
        status = "PASS" if ok else "FAIL"
        print(f"  Attempt {i+1}: {status} — {msg}")
        if ok:
            break


# ── 4. Neural theorem proving ─────────────────────────────────────────────────
def neural_theorem_proving():
    print("\n=== Neural Theorem Proving ===")
    print()
    print("  Proof assistants (symbolic):")
    print("    Lean 4, Isabelle, Coq, Rocq — formally verify mathematical proofs")
    print()
    print("  Neural guidance:")
    print("    LLM proposes next proof tactic → symbolic checker verifies")
    print("    If invalid → backtrack; if valid → continue")
    print()
    print("  Key results:")
    results = [
        ("GPT-f (OpenAI)",       "GPT-3 for proof completion in Metamath; 25% problems"),
        ("AlphaProof",           "DeepMind; LLM + Lean; IMO 2024: 4/6 problems"),
        ("AlphaGeometry",        "DeepMind; LLM + DDAR solver; IMO geometry gold"),
        ("LLM+Lean (MIT)",       "Draft-Sketch-Prove; informal → formal translation"),
        ("COPRA",                "Contextual online proof retrieval augmentation"),
    ]
    for r, d in results:
        print(f"  {r:<24} {d}")
    print()
    print("  Lean 4 example (illustrative):")
    lean_ex = '''
theorem even_plus_even (m n : Nat) (hm : Even m) (hn : Even n) : Even (m + n) := by
  obtain ⟨k, hk⟩ := hm
  obtain ⟨l, hl⟩ := hn
  exact ⟨k + l, by linarith⟩
'''
    for line in lean_ex.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    neurosym_overview()
    symbolic_reasoning_demo()
    llm_plus_verifier()
    neural_theorem_proving()
