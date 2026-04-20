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

def demo_class_activation():
    """Identify discriminative pixels using SVM linear weight vector."""
    print("\n=== Class Activation Analysis ===")
    digits = load_digits()
    X = digits.data / 16.0; y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.svm import SVC
    svm = SVC(kernel="linear", C=1.0).fit(X_tr, y_tr)
    class_idx = 5
    coef = svm.coef_[class_idx]
    top_pixels = np.argsort(np.abs(coef))[-8:]
    print(f"  Top 8 discriminative pixels for digit '{class_idx}': {top_pixels.tolist()}")
    print(f"  Max weight: {coef.max():.4f}  Min weight: {coef.min():.4f}")


def demo_error_analysis():
    """Examine misclassified samples to find most confused digit pairs."""
    print("\n=== Error Analysis ===")
    from collections import Counter
    digits = load_digits()
    X = digits.data / 16.0; y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVC(kernel="rbf", C=10, gamma="scale").fit(X_tr, y_tr)
    preds = svm.predict(X_te)
    errors = np.where(preds != y_te)[0]
    print(f"  Errors: {len(errors)} / {len(y_te)} ({100*len(errors)/len(y_te):.1f}%)")
    pairs = Counter((int(y_te[i]), int(preds[i])) for i in errors)
    print("  Top 5 confused (true -> pred):")
    for (t, p), cnt in pairs.most_common(5):
        print(f"    {t} -> {p}: {cnt} times")


if __name__ == "__main__":
    demo()
    demo_class_activation()
    demo_error_analysis()
