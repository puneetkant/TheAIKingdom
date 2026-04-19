"""
Working Example 2: CI/CD for ML — pipeline simulation, test gate, deployment gate
==================================================================================
Simulates a full CI/CD pipeline: lint -> unit test -> train -> evaluate -> promote.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# -- Stages --------------------------------------------------------------------
def stage_lint(code_str):
    """Checks for obvious issues."""
    issues = []
    if "eval(" in code_str: issues.append("eval() detected")
    if len([l for l in code_str.split("\n") if len(l) > 120]) > 0:
        issues.append("Line too long (>120)")
    return issues

def stage_unit_test():
    """Minimal model unit tests."""
    np.random.seed(1)
    # Test 1: predict shape
    W = np.random.randn(5)
    X = np.random.randn(10, 4)
    Xb = np.column_stack([np.ones(10), X])
    preds = (1/(1+np.exp(-(Xb@W).clip(-50,50))) >= 0.5).astype(int)
    assert preds.shape == (10,), "Shape mismatch"
    # Test 2: binary output
    assert set(preds.tolist()).issubset({0,1}), "Non-binary output"
    return True

def stage_train(seed=42):
    np.random.seed(seed)
    N, F = 200, 4
    X = np.random.randn(N, F)
    y = (X[:, 0] + 0.5*X[:, 1] > 0).astype(int)
    split = int(0.8 * N)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    W = np.zeros(F + 1); Xb = np.column_stack([np.ones(len(X_tr)), X_tr])
    for _ in range(100):
        p = 1/(1+np.exp(-(Xb@W).clip(-50,50)))
        W -= 0.2 * (Xb.T @ (p - y_tr)) / len(y_tr)
    Xvb = np.column_stack([np.ones(len(X_val)), X_val])
    val_preds = (1/(1+np.exp(-(Xvb@W).clip(-50,50))) >= 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()
    return W, val_acc

def stage_evaluate(val_acc, threshold=0.70):
    passed = val_acc >= threshold
    return passed, {"val_acc": round(val_acc, 4), "threshold": threshold}

def stage_deploy(model_W, version):
    # Save model artifact
    np.save(OUTPUT / f"model_v{version}.npy", model_W)
    return f"model_v{version}.npy"

# -- Pipeline runner -----------------------------------------------------------
def run_pipeline(version="1.0"):
    print(f"=== CI/CD Pipeline (v{version}) ===\n")
    results = {}

    print("  [1/5] Lint...")
    issues = stage_lint("def train(): pass\n")
    if issues:
        print(f"  [X] Lint failed: {issues}"); return False
    print("  [OK] Lint passed")
    results["lint"] = "pass"

    print("  [2/5] Unit tests...")
    try:
        stage_unit_test()
        print("  [OK] Tests passed")
        results["unit_test"] = "pass"
    except AssertionError as e:
        print(f"  [X] Test failed: {e}"); return False

    print("  [3/5] Training...")
    W, val_acc = stage_train()
    print(f"  [OK] Trained — val_acc={val_acc:.4f}")
    results["train"] = "pass"

    print("  [4/5] Evaluation gate...")
    passed, metrics = stage_evaluate(val_acc, threshold=0.70)
    if not passed:
        print(f"  [X] Eval gate failed: {metrics}"); return False
    print(f"  [OK] Eval gate passed: {metrics}")
    results["eval"] = "pass"

    print("  [5/5] Deploy...")
    artifact = stage_deploy(W, version)
    print(f"  [OK] Deployed: {artifact}")
    results["deploy"] = "pass"

    print(f"\n  Pipeline complete. All stages: {list(results.values())}")

    # Visualise pipeline
    stages = ["lint","unit_test","train","eval","deploy"]
    colours = ["green" if results.get(s) == "pass" else "red" for s in stages]
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, (s, c) in enumerate(zip(stages, colours)):
        ax.barh(0, 1, left=i, color=c, edgecolor="white", height=0.5)
        ax.text(i + 0.5, 0, s, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xlim(0, len(stages)); ax.axis("off"); ax.set_title("CI/CD Pipeline Status")
    plt.tight_layout(); plt.savefig(OUTPUT / "cicd_pipeline.png"); plt.close()
    print("  Saved cicd_pipeline.png")
    return True

if __name__ == "__main__":
    run_pipeline("2.1")
