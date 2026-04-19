"""
Working Example: Mixture of Experts (MoE)
Covers sparse MoE architecture, routing algorithms, training challenges,
and key models like Mixtral and DeepSeek-MoE.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_moe")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. MoE motivation ---------------------------------------------------------
def moe_motivation():
    print("=== Mixture of Experts (MoE) ===")
    print()
    print("  Key idea: replace FFN with N expert FFNs")
    print("  Activate only K experts per token (sparse MoE)")
    print("  -> more parameters, same compute cost per forward pass")
    print()
    print("  Scaling comparison:")
    print(f"  {'Architecture':<22} {'Total params':>14} {'Active params/token':>20}")
    configs = [
        ("Dense 7B",        "7B",   "7B"),
        ("MoE 8x7B (top-2)", "47B", "~13B"),
        ("Dense 70B",       "70B",  "70B"),
        ("DeepSeek-V3",     "671B", "~37B"),
    ]
    for arch, total, active in configs:
        print(f"  {arch:<22} {total:>14} {active:>20}")
    print()
    print("  Result: 47B MoE can match 70B dense at ~13B compute cost")


# -- 2. Router mechanism -------------------------------------------------------
def router_demo():
    print("\n=== Routing Mechanism ===")
    print()
    print("  Token-choice routing (standard):")
    print("    gate(x) = softmax(x @ W_gate)  [shape: (n_experts,)]")
    print("    top_k   = select K highest gate scores")
    print("    output  = Sigma gate_k * expert_k(x)  for each selected expert")
    print()

    rng = np.random.default_rng(0)
    n_experts = 8
    top_k     = 2
    d_model   = 16

    W_gate = rng.normal(0, 0.1, (d_model, n_experts))

    def route(x):
        logits = x @ W_gate
        scores = np.exp(logits) / np.exp(logits).sum()
        top_idx = np.argsort(-scores)[:top_k]
        top_scores = scores[top_idx]
        top_scores /= top_scores.sum()  # renormalise
        return top_idx, top_scores

    print("  Routing for a batch of 5 tokens:")
    print(f"  {'Token':>6}  {'Expert 1':>10} {'Score':>7}  {'Expert 2':>10} {'Score':>7}")
    for i in range(5):
        x = rng.normal(0, 1, d_model)
        idx, scores = route(x)
        print(f"  tok_{i:>2d}   expert_{idx[0]:<2d}  {scores[0]:.4f}   expert_{idx[1]:<2d}  {scores[1]:.4f}")


# -- 3. Load balancing ---------------------------------------------------------
def load_balancing():
    print("\n=== Load Balancing Problem ===")
    print()
    print("  Without load balancing: routing collapses to a few experts")
    print("  ('rich get richer' — some experts dominate training)")
    print()
    print("  Solutions:")
    solutions = [
        ("Auxiliary loss",   "Penalise imbalanced expert assignment (Shazeer 2017)"),
        ("Noise injection",  "Add Gaussian noise to router logits during training"),
        ("Expert capacity",  "Each expert processes at most capacity_factor * T/N tokens"),
        ("Switch Routing",   "Top-1 only; simpler; capacity drop = dropped tokens"),
        ("Expert-choice",    "Experts pick tokens (not tokens pick experts); auto-balanced"),
        ("DeepSeek aux-free","Bias term in router to steer load; no auxiliary loss needed"),
    ]
    for s, d in solutions:
        print(f"  {s:<20} {d}")
    print()

    # Simulate load imbalance
    rng = np.random.default_rng(1)
    n_experts = 8
    n_tokens  = 100
    top_k     = 2
    loads = np.zeros(n_experts, dtype=int)
    for _ in range(n_tokens):
        # biased routing: experts 0 and 1 attract more
        logits = rng.normal(0, 1, n_experts)
        logits[0] += 1.5
        logits[1] += 1.0
        probs = np.exp(logits) / np.exp(logits).sum()
        chosen = np.argsort(-probs)[:top_k]
        loads[chosen] += 1

    print(f"  Expert load distribution (100 tokens, top-2, biased router):")
    for i, load in enumerate(loads):
        bar = "#" * load
        print(f"    Expert {i}: {load:3d} tokens  {bar}")
    ideal = n_tokens * top_k / n_experts
    print(f"  Ideal: {ideal:.0f} tokens/expert")
    print(f"  Max/min ratio: {loads.max()/loads.min():.1f}x (want ~1.0x)")


# -- 4. Key MoE models ---------------------------------------------------------
def moe_models():
    print("\n=== Key MoE Models ===")
    print()
    models = [
        ("GShard",          "Google 2020; 600B; multilingual NMT; pioneer"),
        ("Switch Transformer","Google 2022; top-1 routing; 1.6T params; simple"),
        ("GLaM",            "Google; 1.2T; top-2; language tasks"),
        ("Mixtral 8x7B",    "Mistral; top-2 of 8; 47B total; 13B active; open"),
        ("Mixtral 8x22B",   "Mistral; 141B total; 39B active; strong benchmark"),
        ("DeepSeek-MoE",    "DeepSeek; fine-grained: 64 experts top-6; shared experts"),
        ("DeepSeek-V2",     "236B / 21B active; MLA attention + MoE; efficient"),
        ("DeepSeek-V3",     "671B / 37B active; multi-token prediction; SOTA open"),
        ("Qwen-MoE",        "Alibaba; 57B / 14B active; A14B competitive"),
        ("Arctic",          "Snowflake; 480B / 17B active; code/SQL focus"),
    ]
    print(f"  {'Model':<22} {'Notes'}")
    for m, d in models:
        print(f"  {m:<22} {d}")
    print()
    print("  DeepSeek MoE innovations:")
    innovations = [
        "Fine-grained experts: 64 small experts vs 8 large (richer routing)",
        "Shared experts: 2 always-active experts + 64 sparse = stability",
        "Aux-free load balancing: bias-based routing adjustment",
        "Multi-token prediction: predict 1 main + 1 speculative token",
    ]
    for inv in innovations:
        print(f"  • {inv}")


if __name__ == "__main__":
    moe_motivation()
    router_demo()
    load_balancing()
    moe_models()
