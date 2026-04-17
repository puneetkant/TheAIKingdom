"""
Working Example: Interview Preparation
Covers ML interview question types, system design, coding practice,
and how to approach common ML interview scenarios.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_interviews")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Interview types ────────────────────────────────────────────────────────
def interview_types():
    print("=== ML Interview Preparation ===")
    print()
    print("  Interview round types (typical FAANG/startup):")
    rounds = [
        ("Coding (DSA)",       "LeetCode medium; arrays, graphs, DP; 45-60 min"),
        ("ML fundamentals",    "Stats, ML theory, algorithms; whiteboard/oral"),
        ("ML system design",   "Design a real system at scale; open-ended"),
        ("Applied ML coding",  "Implement model/metric from scratch in Python/numpy"),
        ("Research discussion","Explain papers; your work; critiques"),
        ("Behavioural",        "STAR format; leadership, conflict, impact"),
    ]
    for r, d in rounds:
        print(f"  {r:<24} {d}")


# ── 2. Core ML questions with answers ─────────────────────────────────────────
def core_ml_questions():
    print("\n=== Core ML Interview Questions ===")
    print()

    qa = [
        ("Bias-variance tradeoff?",
         "High bias = underfitting (train error high); high variance = overfitting (gap train-val).\n"
         "   Fix bias: more capacity, better features. Fix variance: regularisation, more data."),
        ("Why do we normalise features?",
         "Gradient descent converges faster; equal scales prevent dominance;\n"
         "   distance-based models (KNN, SVM, PCA) require it."),
        ("L1 vs L2 regularisation?",
         "L1 (Lasso): ||w||₁ → sparse weights; useful for feature selection.\n"
         "   L2 (Ridge): ||w||₂² → small non-zero weights; standard for NNs."),
        ("Cross-entropy loss for classification?",
         "L = -Σ y_i log(p_i); minimises KL divergence between true and predicted dist."),
        ("Why batch norm?",
         "Reduces internal covariate shift; allows higher LR; slight regularisation;\n"
         "   stabilises deep network training significantly."),
        ("Attention complexity?",
         "O(L²·d) — quadratic in sequence length L; bottleneck for long sequences."),
        ("Why does Adam work well?",
         "Adaptive LR per parameter (RMSProp) + momentum; robust to hyperparams;\n"
         "   handles sparse gradients; widely used default optimiser."),
    ]
    for q, a in qa:
        print(f"  Q: {q}")
        print(f"     A: {a}")
        print()


# ── 3. Implement from scratch ─────────────────────────────────────────────────
def implement_from_scratch():
    print("=== Common 'Implement From Scratch' Tasks ===")
    print()

    # K-means (common interview ask)
    def kmeans(X, k, n_iter=20, seed=0):
        rng = np.random.default_rng(seed)
        centres = X[rng.choice(len(X), k, replace=False)]
        for _ in range(n_iter):
            dists   = np.linalg.norm(X[:, None] - centres[None], axis=2)
            labels  = dists.argmin(axis=1)
            centres = np.array([X[labels == c].mean(0) for c in range(k)])
        return labels, centres

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([2, 2], 0.5, (30, 2)),
                   rng.normal([7, 7], 0.5, (30, 2)),
                   rng.normal([2, 7], 0.5, (30, 2))])

    labels, centres = kmeans(X, k=3)
    print("  K-means from scratch (numpy only):")
    for c in range(3):
        pts = X[labels == c]
        print(f"    Cluster {c}: {len(pts)} points, centroid = {pts.mean(0).round(2)}")
    print()
    print("  Other common 'implement' asks:")
    tasks = [
        "Logistic regression with gradient descent",
        "Attention mechanism (QKV)",
        "Backpropagation for a 2-layer MLP",
        "Precision / Recall / F1 / AUROC from scratch",
        "Train/test split with stratification",
        "Cosine similarity matrix",
        "Non-maximum suppression (NMS) for object detection",
    ]
    for t in tasks:
        print(f"  • {t}")


# ── 4. ML system design ───────────────────────────────────────────────────────
def ml_system_design():
    print("\n=== ML System Design Framework ===")
    print()
    print("  Framework (use this structure for any system design question):")
    steps = [
        ("1. Clarify",         "Scope, scale, latency, accuracy requirements; ask questions"),
        ("2. ML framing",      "What to predict? Objective? Data availability?"),
        ("3. Data pipeline",   "Collection, labelling, storage, preprocessing"),
        ("4. Feature eng.",    "Online vs offline; feature store; embedding strategy"),
        ("5. Model selection", "Justify: complexity, latency, interpretability"),
        ("6. Training",        "Batch vs online; compute; experiment tracking"),
        ("7. Serving",         "Latency SLA; batch vs real-time; API design"),
        ("8. Monitoring",      "Drift; shadow mode; A/B test; rollback"),
    ]
    for s, d in steps:
        print(f"  {s:<18} {d}")
    print()
    print("  Example problems:")
    problems = [
        "Design YouTube recommendations (candidate gen + ranking + re-ranking)",
        "Design a spam filter for email (real-time; low latency; explainability)",
        "Design a fraud detection system (imbalanced; concept drift; low FP)",
        "Design a search ranking system (query understanding; semantic + BM25)",
        "Design a medical image classifier (FDA; interpretability; uncertainty)",
    ]
    for p in problems:
        print(f"  • {p}")


# ── 5. Behavioural tips ───────────────────────────────────────────────────────
def behavioural_prep():
    print("\n=== Behavioural Interview Prep ===")
    print()
    print("  STAR format: Situation, Task, Action, Result")
    print()
    print("  Must-have stories:")
    stories = [
        "Biggest technical challenge you overcame",
        "Time you had a data/model failure; how you diagnosed and fixed it",
        "Time you disagreed with a team member on technical direction",
        "Most impactful project; quantify the impact",
        "Time you had to learn something quickly",
        "How you handle ambiguous problems with no clear answer",
    ]
    for s in stories:
        print(f"  • {s}")
    print()
    print("  Quantify everything: '17% improvement', '10x faster', 'saved $200k/year'")


if __name__ == "__main__":
    interview_types()
    core_ml_questions()
    implement_from_scratch()
    ml_system_design()
    behavioural_prep()
