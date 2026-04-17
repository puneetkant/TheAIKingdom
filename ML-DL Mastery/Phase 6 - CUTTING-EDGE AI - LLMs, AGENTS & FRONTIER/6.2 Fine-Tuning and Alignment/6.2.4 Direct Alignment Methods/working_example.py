"""
Working Example: Direct Alignment Methods
Covers DPO, IPO, ORPO, SimPO, KTO, and preference optimisation theory.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_direct_align")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# ── 1. Why direct alignment? ──────────────────────────────────────────────────
def direct_alignment_intro():
    print("=== Direct Alignment Methods ===")
    print()
    print("  RLHF pain points that motivated direct methods:")
    pain = [
        ("Complexity",   "4 separate models; complex training loop"),
        ("Instability",  "PPO hyperparameter sensitivity"),
        ("Memory",       "Policy + reference + value + reward = 4× model memory"),
        ("Reward hacking","Reward model can be gamed by policy"),
    ]
    for p, d in pain:
        print(f"  {p:<14} {d}")
    print()
    print("  Direct methods: optimise policy directly from preference data")
    print("  No separate reward model needed")


# ── 2. DPO ────────────────────────────────────────────────────────────────────
def dpo_demo():
    print("\n=== DPO (Direct Preference Optimisation) ===")
    print()
    print("  Rafailov et al. 2023 — most widely adopted direct method")
    print()
    print("  Key insight: RLHF's optimal policy is closed-form")
    print("    π*(y|x) ∝ π_ref(y|x) · exp(R*(y,x) / β)")
    print()
    print("  Re-parameterize reward as ratio of policies:")
    print("    R*(y,x) = β log π*(y|x)/π_ref(y|x) + β log Z(x)")
    print()
    print("  DPO loss:")
    print("    L_DPO = -E log σ(β log π(y_w)/π_ref(y_w) - β log π(y_l)/π_ref(y_l))")
    print()

    rng = np.random.default_rng(0)
    n_pairs = 50; beta = 0.1

    # Simulate log-probability ratios
    log_pi_w     = rng.normal(0.5, 1, n_pairs)   # chosen log probs (better)
    log_pi_l     = rng.normal(-0.5, 1, n_pairs)  # rejected
    log_pi_ref_w = rng.normal(0, 1, n_pairs)
    log_pi_ref_l = rng.normal(0, 1, n_pairs)

    chosen_diff   = beta * (log_pi_w - log_pi_ref_w)
    rejected_diff = beta * (log_pi_l - log_pi_ref_l)
    loss = -np.log(sigmoid(chosen_diff - rejected_diff)).mean()

    # What DPO optimises
    print(f"  Simulated DPO loss: {loss:.4f}")
    print()
    print("  DPO variants:")
    variants = [
        ("DPO",     "Original; β controls KL penalty strength"),
        ("IPO",     "Identity PO; regularises to avoid probability collapse"),
        ("cDPO",    "Conservative; soft labels; handles annotation noise"),
        ("TDPO",    "Token-level DPO; better credit assignment"),
        ("online DPO","Sample from current policy; fresh preference pairs"),
    ]
    for v, d in variants:
        print(f"  {v:<12} {d}")


# ── 3. SimPO ──────────────────────────────────────────────────────────────────
def simpo_demo():
    print("\n=== SimPO (Simple Preference Optimisation) ===")
    print()
    print("  Meng et al. 2024 — no reference model needed")
    print()
    print("  Key differences from DPO:")
    print("    1. Length-normalised reward: r(x,y) = 1/|y| Σ log π(y_t|x,y<t)")
    print("    2. Target reward margin γ: prefer y_w over y_l by at least γ")
    print()
    print("  Loss:")
    print("    L_SimPO = -E log σ(β/|y_w| Σlog π(y_w) - β/|y_l| Σlog π(y_l) - γ)")
    print()
    print("  Benefits:")
    benefits = [
        ("No reference model", "Half the memory; simpler training"),
        ("Length normalisation","Prevents preference for longer responses"),
        ("Margin γ",           "Explicit gap encourages clear separation"),
        ("Performance",        "Often outperforms DPO on AlpacaEval/MT-Bench"),
    ]
    for b, d in benefits:
        print(f"  {b:<22} {d}")


# ── 4. ORPO and KTO ──────────────────────────────────────────────────────────
def orpo_kto():
    print("\n=== ORPO and KTO ===")
    print()
    print("  ORPO (Odds Ratio Preference Optimisation, Hong et al. 2024):")
    print("    Combines SFT loss and preference optimisation in one step")
    print("    No reference model needed")
    print()
    print("    Loss = SFT_loss + λ · OR_loss")
    print("    OR_loss = -log σ(log(odds_ratio(y_w)) - log(odds_ratio(y_l)))")
    print("    odds_ratio(y) = P(y) / (1-P(y))  where P = per-token probability")
    print()
    print("  KTO (Kahneman-Tversky Optimisation, Ethayarajh et al. 2024):")
    print("    No preference pairs needed — just (prompt, response, label) triples")
    print("    Based on prospect theory: humans are loss-averse")
    print()
    print("    Loss = E_good[1-σ(R-z_ref)] + E_bad[1-σ(z_ref-R)]")
    print("    z_ref = KL[π || π_ref]  (reference point)")
    print()

    print("  Method comparison:")
    comp = [
        ("RLHF/PPO",   "Needs pairs",  "Ref model",  "Best quality; most effort"),
        ("DPO",        "Needs pairs",  "Ref model",  "Simple; widely used"),
        ("SimPO",      "Needs pairs",  "None",       "Reference-free; strong"),
        ("ORPO",       "Needs pairs",  "None",       "SFT+alignment in one pass"),
        ("KTO",        "Binary labels","Ref model",  "Works with unpaired data"),
    ]
    print(f"  {'Method':<10} {'Data':<14} {'Ref model':<12} {'Notes'}")
    for m, data, ref, notes in comp:
        print(f"  {m:<10} {data:<14} {ref:<12} {notes}")


if __name__ == "__main__":
    direct_alignment_intro()
    dpo_demo()
    simpo_demo()
    orpo_kto()
