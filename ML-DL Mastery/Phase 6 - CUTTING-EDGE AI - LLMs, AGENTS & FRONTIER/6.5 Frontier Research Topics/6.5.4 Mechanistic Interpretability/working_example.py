"""
Working Example: Mechanistic Interpretability
Covers circuits, features, superposition, attention patterns,
and tools for understanding what LLMs compute internally.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_interp")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Overview ───────────────────────────────────────────────────────────────
def interpretability_overview():
    print("=== Mechanistic Interpretability ===")
    print()
    print("  Goal: understand HOW transformers compute, not just WHAT they output")
    print("  Find algorithms implemented in weights → predict model behaviour")
    print()
    print("  Key concepts:")
    concepts = [
        ("Features",       "Directions in activation space representing concepts"),
        ("Circuits",       "Subgraph of weights implementing a computation"),
        ("Superposition",  "Representing more features than dimensions (compressed)"),
        ("Attention heads","Each head often implements specific algorithm"),
        ("MLP neurons",    "Often encode specific factual associations"),
        ("Residual stream","Cumulative information passed between layers"),
    ]
    for c, d in concepts:
        print(f"  {c:<18} {d}")
    print()
    print("  Leading researchers: Anthropic Interpretability Team, Neel Nanda, EleutherAI")


# ── 2. Superposition ──────────────────────────────────────────────────────────
def superposition_demo():
    print("\n=== Superposition Demo ===")
    print()
    print("  Hypothesis: neural nets store more features than dimensions")
    print("  by using nearly-orthogonal directions in high-dim space")
    print()
    print("  Johnson-Lindenstrauss: n features in d dims with error ε if:")
    print("    n ≤ O(e^(ε² d) )   → exponentially many near-orthogonal directions")
    print()

    # Show how many near-orthogonal vectors fit in d dims
    rng = np.random.default_rng(0)
    d   = 50
    threshold = 0.1  # max |cosine sim| to count as near-orthogonal

    vectors = []
    n_accepted = 0
    for _ in range(10000):
        v = rng.normal(0, 1, d)
        v /= np.linalg.norm(v)
        if all(abs(float(v @ u)) < threshold for u in vectors):
            vectors.append(v)
            n_accepted += 1
        if n_accepted >= 200:
            break

    print(f"  In d={d} dims, near-orthogonal vectors (|cos|<{threshold}): {n_accepted}+")
    print(f"  That's {n_accepted/d:.1f}x more features than dimensions!")
    print()
    print("  Consequence: individual neurons are polysemantic")
    print("  (respond to many unrelated concepts; hard to interpret directly)")


# ── 3. Attention pattern analysis ─────────────────────────────────────────────
def attention_patterns():
    print("\n=== Attention Pattern Analysis ===")
    print()
    print("  Known attention head types (from circuit analysis):")
    head_types = [
        ("Previous token", "Attends to position t-1; copies info"),
        ("Induction",      "Copy-paste: finds and repeats previous occurrences"),
        ("Name mover",     "Copies subject entity to output position"),
        ("Backup mover",   "Secondary path for same operation"),
        ("Negative",       "Attends away from (suppresses) a token"),
        ("BOS attention",  "Attends to beginning-of-sequence (info dumping)"),
    ]
    for ht, d in head_types:
        print(f"  {ht:<18} {d}")
    print()

    # Simulate a simple attention pattern
    tokens = ["The", "cat", "sat", "on", "the", "mat", "the", "cat"]
    T = len(tokens)
    rng = np.random.default_rng(42)
    attn = np.zeros((T, T))

    # Simulate induction head: detect "the cat" repeat
    for i in range(T):
        for j in range(T):
            if i < j:
                if tokens[i] == tokens[j]:
                    attn[j, i] += 3.0
                if (i > 0 and j > 0 and
                        tokens[i-1] == tokens[j-1]):
                    attn[j, i] += 2.0
            attn[j, i] += rng.normal(0, 0.5)
    attn = np.clip(attn, 0, None)
    attn = attn / (attn.sum(1, keepdims=True) + 1e-9)

    print("  Simulated induction head attention matrix:")
    print("  (rows = query token, cols = key token; darker = higher attention)")
    header = "        " + " ".join(f"{t:>6}" for t in tokens)
    print(f"  {header}")
    for i, row_tok in enumerate(tokens):
        row = " ".join("█" * int(attn[i, j] * 8) + "·" * (8 - int(attn[i, j] * 8))
                       if int(attn[i, j] * 8) else "  " for j in range(T))
        print(f"  {row_tok:>6}  {' '.join(f'{attn[i,j]:.2f}' for j in range(T))}")


# ── 4. Sparse autoencoders (SAEs) ─────────────────────────────────────────────
def sparse_autoencoders():
    print("\n=== Sparse Autoencoders (SAEs) ===")
    print()
    print("  Tool for decomposing superposition into interpretable features")
    print("  Train SAE on model activations → discover monosemantic features")
    print()
    print("  SAE architecture:")
    print("    Encoder: x → z = ReLU(W_enc x + b_enc)   [z sparse: most zero]")
    print("    Decoder: z → x̂ = W_dec z + b_dec")
    print("    Loss: ||x - x̂||² + λ ||z||₁  (reconstruction + sparsity)")
    print()
    print("  Anthropic findings (Claude features, 2024):")
    features = [
        "~1M sparse features found in Claude 3 Sonnet residual stream",
        "'Golden Gate' feature: fires on Golden Gate Bridge concept",
        "Multimodal features: concept fires on multiple languages",
        "Safety-relevant features: deception, manipulation concepts found",
        "Feature geometry: related features cluster in representation space",
    ]
    for f in features:
        print(f"  • {f}")
    print()
    print("  SAE tools:")
    tools = [
        ("TransformerLens", "Neel Nanda; hook anywhere in GPT-2/LLaMA"),
        ("SAELens",         "Joseph Bloom; open SAE training + Neuronpedia"),
        ("Baukit",          "Dissect activations; patchable hook framework"),
        ("Neuronpedia",     "Web UI for exploring features interactively"),
    ]
    for t, d in tools:
        print(f"  {t:<18} {d}")


if __name__ == "__main__":
    interpretability_overview()
    superposition_demo()
    attention_patterns()
    sparse_autoencoders()
