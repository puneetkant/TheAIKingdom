"""
Working Example 2: Image Classification — CNN proxy with sklearn on digits
===========================================================================
PCA feature extraction -> SVM, comparing CNN-like pipeline with raw pixels.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
except ImportError:
    raise SystemExit("pip install scikit-learn numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== Image Classification Pipeline ===")
    digits = load_digits()
    X = digits.data / 16.0  # normalise
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    # Raw pixels + SVM
    svm_raw = SVC(kernel="rbf", C=10, gamma="scale").fit(X_tr, y_tr)
    results["SVM (raw 64px)"] = accuracy_score(y_te, svm_raw.predict(X_te))

    # PCA + SVM (simulates CNN feature extraction)
    for n in [16, 32, 48]:
        pca = PCA(n_components=n, random_state=42)
        X_tr_p = pca.fit_transform(X_tr); X_te_p = pca.transform(X_te)
        svm = SVC(kernel="rbf", C=10, gamma="scale").fit(X_tr_p, y_tr)
        results[f"PCA({n})+SVM"] = accuracy_score(y_te, svm.predict(X_te_p))

    print("\n  Results:")
    for name, acc in results.items():
        print(f"    {name:20s}: {acc:.4f}")

    # Confusion matrix for best model
    best_pred = svm_raw.predict(X_te)
    cm = confusion_matrix(y_te, best_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax); ax.set_title("Confusion Matrix (SVM on digits)")
    plt.tight_layout(); plt.savefig(OUTPUT / "image_classification_cm.png"); plt.close()
    print("  Saved image_classification_cm.png")

if __name__ == "__main__":
    demo()
