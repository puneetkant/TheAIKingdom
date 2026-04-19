"""
Working Example: Ethics and Responsible AI
Covers fairness, bias, privacy, transparency, accountability,
and frameworks for building ethical AI systems.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ethics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Principles overview ----------------------------------------------------
def ethics_overview():
    print("=== Ethics and Responsible AI ===")
    print()
    print("  Core principles (EU AI Act, NIST, IEEE):")
    principles = [
        ("Fairness",         "Non-discrimination; equitable outcomes across groups"),
        ("Transparency",     "Explainable decisions; model cards; documentation"),
        ("Privacy",          "Data minimisation; consent; differential privacy"),
        ("Accountability",   "Human oversight; audit trails; clear responsibility"),
        ("Safety",           "Robustness; fail-safe design; red teaming"),
        ("Beneficence",      "Net positive impact; consider all stakeholders"),
        ("Sustainability",   "Environmental impact; carbon cost of training"),
    ]
    for p, d in principles:
        print(f"  {p:<18} {d}")


# -- 2. Fairness metrics -------------------------------------------------------
def fairness_metrics():
    print("\n=== Fairness Metrics Demo ===")
    print()
    print("  Scenario: loan approval model; protected attribute = gender")
    print()

    rng = np.random.default_rng(42)

    # Simulate predictions vs true labels for two groups
    n = 200
    group_A_y    = rng.integers(0, 2, n)   # Group A true labels
    group_B_y    = rng.integers(0, 2, n)   # Group B true labels
    group_A_pred = (group_A_y + rng.integers(0, 2, n)) % 2  # noisier for A
    group_B_pred = group_B_y.copy()                           # cleaner for B

    def metrics(y, y_hat):
        TP = ((y == 1) & (y_hat == 1)).sum()
        TN = ((y == 0) & (y_hat == 0)).sum()
        FP = ((y == 0) & (y_hat == 1)).sum()
        FN = ((y == 1) & (y_hat == 0)).sum()
        accuracy   = (TP + TN) / len(y)
        approval   = y_hat.mean()
        tpr        = TP / (TP + FN + 1e-9)  # True Positive Rate (recall)
        fpr        = FP / (FP + TN + 1e-9)  # False Positive Rate
        return accuracy, approval, tpr, fpr

    acc_A, appr_A, tpr_A, fpr_A = metrics(group_A_y, group_A_pred)
    acc_B, appr_B, tpr_B, fpr_B = metrics(group_B_y, group_B_pred)

    print(f"  {'Metric':<22} {'Group A':>10} {'Group B':>10} {'Ratio':>10}")
    print(f"  {'-'*55}")
    for name, vA, vB in [
        ("Accuracy",          acc_A,  acc_B),
        ("Approval rate",     appr_A, appr_B),
        ("True positive rate", tpr_A, tpr_B),
        ("False positive rate",fpr_A, fpr_B),
    ]:
        ratio = vA / (vB + 1e-9)
        print(f"  {name:<22} {vA:>10.3f} {vB:>10.3f} {ratio:>10.2f}")
    print()
    print("  Fairness definitions:")
    definitions = [
        ("Demographic parity",     "Approval rate equal across groups (approval ratio ~1)"),
        ("Equalized opportunity",  "True positive rate equal across groups (TPR ratio ~1)"),
        ("Equalized odds",         "Both TPR and FPR equal across groups"),
        ("Calibration",            "P(y=1|score=s) equal across groups"),
    ]
    print("  Note: most fairness criteria cannot be simultaneously satisfied!")
    for d, desc in definitions:
        print(f"  {d:<26} {desc}")


# -- 3. Bias sources and mitigations ------------------------------------------
def bias_sources():
    print("\n=== Bias Sources and Mitigations ===")
    print()
    print("  Where bias enters:")
    sources = [
        ("Historical bias",      "Training data reflects past discrimination"),
        ("Representation bias",  "Underrepresented groups in dataset"),
        ("Measurement bias",     "Proxy variables measured differently across groups"),
        ("Aggregation bias",     "One model for all groups ignores subgroup needs"),
        ("Evaluation bias",      "Benchmarks test majority population only"),
        ("Deployment bias",      "Context changes between train and deploy"),
    ]
    for s, d in sources:
        print(f"  {s:<22} {d}")
    print()
    print("  Mitigation strategies:")
    mitigations = [
        ("Pre-processing",  "Reweigh data; oversample minority; fair representations"),
        ("In-processing",   "Constrained optimisation; adversarial debiasing"),
        ("Post-processing", "Threshold adjustment per group; reject option"),
        ("Auditing",        "Regular disparate impact testing; external audit"),
        ("Documentation",   "Datasheets for Datasets; Model Cards"),
    ]
    for m, d in mitigations:
        print(f"  {m:<18} {d}")


# -- 4. Privacy and data governance -------------------------------------------
def privacy_and_governance():
    print("\n=== Privacy and Data Governance ===")
    print()
    print("  Privacy-preserving techniques:")
    techniques = [
        ("Differential Privacy",   "Add calibrated noise; provable privacy guarantee"),
        ("Federated Learning",     "Train on device; only gradients shared; Google Gboard"),
        ("k-Anonymity",            "Each record indistinguishable from k-1 others"),
        ("Homomorphic Encryption", "Compute on encrypted data; very slow"),
        ("SMPC",                   "Secure multi-party computation; split computation"),
        ("Synthetic data",         "Generate new data with same statistics; no real PII"),
    ]
    for t, d in techniques:
        print(f"  {t:<28} {d}")
    print()
    print("  Regulations:")
    regs = [
        ("GDPR (EU)",           "Right to erasure; consent; data minimisation; AI Act"),
        ("CCPA (California)",   "Right to know, delete, opt-out of sale"),
        ("HIPAA (US medical)",  "PHI protection; ML on medical data"),
        ("EU AI Act",           "Risk-based; high-risk AI requires audit and docs"),
        ("NIST AI RMF",         "Framework for AI risk management; voluntary"),
    ]
    for r, d in regs:
        print(f"  {r:<26} {d}")


# -- 5. Environmental impact ---------------------------------------------------
def environmental_impact():
    print("\n=== Environmental Impact of AI ===")
    print()
    print("  CO2 estimates for training large models:")
    models_co2 = [
        ("BERT base",       "~0.65 t CO2"),
        ("GPT-3 175B",      "~552 t CO2"),
        ("BLOOM 176B",      "~25 t CO2 (green energy)"),
        ("LLaMA-3 70B",     "~100-200 t CO2 est."),
        ("GPT-4 (est.)",    ">1000 t CO2 est."),
    ]
    for m, co2 in models_co2:
        print(f"  {m:<24} {co2}")
    print()
    print("  Mitigation:")
    mitigations = [
        "Use renewable-energy data centres (cf. Iceland, Norway)",
        "Report emissions transparently (like BLOOM, LLaMA-3 card)",
        "Prefer efficient architectures (MoE, quantisation)",
        "Don't train from scratch when fine-tuning suffices",
        "Model distillation: smaller model, same capability",
    ]
    for m in mitigations:
        print(f"  • {m}")


if __name__ == "__main__":
    ethics_overview()
    fairness_metrics()
    bias_sources()
    privacy_and_governance()
    environmental_impact()
