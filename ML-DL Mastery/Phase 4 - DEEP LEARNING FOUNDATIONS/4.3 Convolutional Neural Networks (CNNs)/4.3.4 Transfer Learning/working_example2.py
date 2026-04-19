"""
Working Example 2: Transfer Learning — feature extraction vs fine-tuning (sklearn proxy)
=========================================================================================
Uses pre-computed digit embeddings to simulate feature extraction vs training from scratch.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== Transfer Learning Proxy (sklearn digits) ===")
    digits = load_digits()
    X = digits.data / 16.0; y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # "Pretrained feature extractor" = PCA on full data (represents frozen backbone)
    pca = PCA(n_components=20, random_state=42).fit(X)   # fit on "imagenet" = all data
    X_tr_feat = pca.transform(X_tr); X_te_feat = pca.transform(X_te)

    # Feature extraction (frozen backbone, new head only)
    lr_feat = LogisticRegression(max_iter=500, random_state=42)
    lr_feat.fit(X_tr_feat, y_tr)
    acc_feat = accuracy_score(y_te, lr_feat.predict(X_te_feat))

    # "Fine-tuning" = raw pixels + logistic reg (all layers trainable)
    lr_raw = LogisticRegression(max_iter=500, random_state=42)
    lr_raw.fit(X_tr, y_tr)
    acc_raw = accuracy_score(y_te, lr_raw.predict(X_te))

    print(f"  Feature extraction (PCA backbone): acc={acc_feat:.4f}")
    print(f"  Fine-tuning (raw pixels):           acc={acc_raw:.4f}")

    # Few-shot: train only on 10% of data
    X_few, _, y_few, _ = train_test_split(X_tr, y_tr, train_size=0.1, random_state=42)
    X_few_feat = pca.transform(X_few)
    lr_few_feat = LogisticRegression(max_iter=500, random_state=42).fit(X_few_feat, y_few)
    lr_few_raw  = LogisticRegression(max_iter=500, random_state=42).fit(X_few, y_few)
    print(f"\n  Few-shot (10% data):")
    print(f"    Feature extraction: acc={accuracy_score(y_te, lr_few_feat.predict(X_te_feat)):.4f}")
    print(f"    From scratch:       acc={accuracy_score(y_te, lr_few_raw.predict(X_te)):.4f}")
    print("  -> Transfer learning advantage larger with less data!")

def demo_feature_extraction():
    """Simulate transfer learning: PCA features = 'pretrained', train classifier on top."""
    print("=== Feature Extraction (Transfer Learning Simulation) ===")
    from sklearn.datasets import fetch_california_housing
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
    # Baseline: raw features
    raw_acc = LogisticRegression(max_iter=500).fit(X_tr_s, y_tr).score(X_te_s, y_te)
    # "Pretrained" features via PCA (simulate encoder)
    pca = PCA(n_components=6).fit(X_tr_s)
    X_tr_pca = pca.transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)
    pca_acc = LogisticRegression(max_iter=500).fit(X_tr_pca, y_tr).score(X_te_pca, y_te)
    print(f"  Raw features accuracy:         {raw_acc:.4f}")
    print(f"  PCA 'pretrained' features acc: {pca_acc:.4f}")
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")


def demo_few_shot_transfer():
    """Few-shot: pretrained features enable good accuracy with few labels."""
    print("\n=== Few-Shot Learning with Transferred Features ===")
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    X, y = load_iris(return_X_y=True)
    sc = StandardScaler().fit(X)
    X_s = sc.transform(X)
    pca = PCA(n_components=3).fit(X_s)  # "pretrained" on all
    X_pca = pca.transform(X_s)
    for n_shots in [5, 10, 20, 50]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_shots, random_state=42)
        tr, te = next(sss.split(X_pca, y))
        acc_raw = LogisticRegression(max_iter=300).fit(X_s[tr], y[tr]).score(X_s[te], y[te])
        acc_tl  = LogisticRegression(max_iter=300).fit(X_pca[tr], y[tr]).score(X_pca[te], y[te])
        print(f"  n_shots={n_shots:>3}: raw={acc_raw:.3f}  transferred={acc_tl:.3f}")


def demo_domain_adaptation():
    """Domain shift: show accuracy drop when target distribution differs from source."""
    print("\n=== Domain Adaptation / Distribution Shift ===")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    rng = np.random.default_rng(42)
    # Source domain
    X_src, y_src = make_classification(n_samples=500, n_features=10, random_state=42)
    # Target domain: shifted means and scaled features
    X_tgt, y_tgt = make_classification(n_samples=200, n_features=10, random_state=99)
    X_tgt = X_tgt * 2 + 1.5  # distribution shift
    sc = StandardScaler().fit(X_src)
    # No adaptation
    clf = LogisticRegression(max_iter=300).fit(sc.transform(X_src), y_src)
    no_adapt = clf.score(sc.transform(X_tgt), y_tgt)
    # With target standardization (CORAL-like)
    sc_tgt = StandardScaler().fit(X_tgt)
    with_adapt = clf.score(sc_tgt.transform(X_tgt), y_tgt)
    print(f"  No adaptation:        {no_adapt:.4f}")
    print(f"  Target normalization: {with_adapt:.4f}")
    print(f"  Improvement:          {with_adapt - no_adapt:+.4f}")


if __name__ == "__main__":
    demo()
    demo_feature_extraction()
    demo_few_shot_transfer()
    demo_domain_adaptation()
