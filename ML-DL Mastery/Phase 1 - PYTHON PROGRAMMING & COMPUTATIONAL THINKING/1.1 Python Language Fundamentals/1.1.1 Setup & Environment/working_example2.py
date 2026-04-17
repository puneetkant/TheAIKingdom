"""
Working Example 2: Setup & Environment — Real-World Project Bootstrap
=======================================================================
Demonstrates a realistic project setup workflow:
  - Auto-installing missing packages at runtime
  - Downloading a sample dataset from Hugging Face Hub
  - Verifying environment integrity before training
  - Logging environment info to a run artefact

Run:  python working_example2.py
"""
import sys
import os
import subprocess
import json
import platform
import hashlib
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

REQUIRED_PACKAGES = {
    "numpy":    "numpy",
    "pandas":   "pandas",
    "requests": "requests",
    "sklearn":  "scikit-learn",
}


# ── 1. Auto-install missing packages ──────────────────────────────────────────
def ensure_packages(required: dict[str, str]) -> None:
    """Install any missing packages programmatically."""
    print("=== Checking / Installing Required Packages ===")
    import importlib
    for import_name, pip_name in required.items():
        try:
            importlib.import_module(import_name)
            print(f"  ✓  {pip_name}")
        except ImportError:
            print(f"  ⬇  Installing {pip_name} …")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✓  {pip_name} installed")
            else:
                print(f"  ✗  Failed to install {pip_name}: {result.stderr[:200]}")


# ── 2. Download sample dataset from Hugging Face Hub (raw URL) ────────────────
def download_iris_from_hf() -> Path:
    """
    Download the Iris dataset CSV from Hugging Face Hub datasets.
    Uses raw HTTP — no huggingface_hub library needed.
    """
    import urllib.request

    url = (
        "https://huggingface.co/datasets/scikit-learn/iris/resolve/main/Iris.csv"
    )
    dest = DATA_DIR / "iris.csv"
    if dest.exists():
        print(f"\n=== Dataset already cached: {dest} ===")
        return dest

    print(f"\n=== Downloading Iris dataset from Hugging Face Hub ===")
    print(f"  URL : {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓  Saved to {dest}  ({dest.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        # Fallback: generate synthetic Iris-like data
        print(f"  ✗  Download failed ({e}). Generating synthetic fallback …")
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 150
        species = ["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50
        data = pd.DataFrame({
            "SepalLengthCm": rng.normal(5.8, 0.8, n).round(1),
            "SepalWidthCm":  rng.normal(3.0, 0.4, n).round(1),
            "PetalLengthCm": rng.normal(3.7, 1.8, n).round(1),
            "PetalWidthCm":  rng.normal(1.2, 0.8, n).round(1),
            "Species":       species,
        })
        data.to_csv(dest, index=False)
        print(f"  ✓  Synthetic dataset saved to {dest}")
    return dest


# ── 3. Explore the dataset ────────────────────────────────────────────────────
def explore_dataset(csv_path: Path) -> None:
    import pandas as pd
    import numpy as np

    print("\n=== Dataset Exploration ===")
    df = pd.read_csv(csv_path)
    print(f"  Shape     : {df.shape}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Dtypes    :")
    for col, dt in df.dtypes.items():
        print(f"    {col:<18} {dt}")
    print(f"\n  Class distribution:")
    species_col = "Species" if "Species" in df.columns else df.columns[-1]
    for cls, cnt in df[species_col].value_counts().items():
        print(f"    {cls:<20} {cnt}")
    print(f"\n  Descriptive stats (numeric columns):")
    print(df.select_dtypes("number").describe().round(2).to_string(index=True))


# ── 4. Train a simple model and log results ───────────────────────────────────
def train_and_evaluate(csv_path: Path) -> None:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    print("\n=== Training Models on Iris Dataset ===")
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if df[c].dtype != object and c != "Id"]
    species_col  = "Species" if "Species" in df.columns else df.columns[-1]

    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df[species_col])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, clf in [
        ("LogisticRegression",  LogisticRegression(max_iter=500)),
        ("RandomForest",        RandomForestClassifier(n_estimators=100, random_state=42)),
    ]:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        results[name] = scores
        print(f"  {name:<22} acc={scores.mean():.4f} ± {scores.std():.4f}")

    return results


# ── 5. Log environment + results to JSON artifact ─────────────────────────────
def save_run_artifact(model_results: dict) -> None:
    import importlib

    print("\n=== Saving Run Artifact ===")
    pkgs = {}
    for pkg in ["numpy", "pandas", "sklearn", "torch", "tensorflow"]:
        try:
            m = importlib.import_module(pkg)
            pkgs[pkg] = getattr(m, "__version__", "?")
        except ImportError:
            pkgs[pkg] = "not installed"

    artifact = {
        "python_version": sys.version,
        "platform":       platform.platform(),
        "packages":       pkgs,
        "model_results":  {
            name: {"mean": float(scores.mean()), "std": float(scores.std())}
            for name, scores in model_results.items()
        },
    }
    out = LOGS_DIR / "run_artifact.json"
    out.write_text(json.dumps(artifact, indent=2))
    print(f"  ✓  Artifact saved to {out}")
    print(f"  Env hash: {hashlib.md5(json.dumps(artifact).encode()).hexdigest()[:8]}")


# ── 6. Dependency conflict check ──────────────────────────────────────────────
def check_dependency_conflicts() -> None:
    print("\n=== Dependency Conflict Check ===")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, timeout=20
    )
    output = result.stdout.strip() or result.stderr.strip()
    if "No broken requirements" in output or result.returncode == 0:
        print("  ✓  No dependency conflicts detected")
    else:
        print("  ⚠  Issues found:")
        for line in output.splitlines()[:10]:
            print(f"    {line}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_packages(REQUIRED_PACKAGES)
    csv_path     = download_iris_from_hf()
    explore_dataset(csv_path)
    results      = train_and_evaluate(csv_path)
    save_run_artifact(results)
    check_dependency_conflicts()
