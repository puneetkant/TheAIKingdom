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
    print("  → Transfer learning advantage larger with less data!")

if __name__ == "__main__":
    demo()
