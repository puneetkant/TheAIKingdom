"""
Working Example: Information Theory
Covers entropy, joint/conditional entropy, mutual information,
KL divergence, cross-entropy, and their ML/DL applications.
"""
import numpy as np
from scipy import stats


# -- Helper: safe log ----------------------------------------------------------
def _xlogy(x, y):
    """x * log(y), treating 0*log(0) as 0."""
    return np.where(x == 0, 0.0, x * np.log2(y + 1e-300))


# -- 1. Shannon entropy H(X) ---------------------------------------------------
def shannon_entropy():
    print("=== Shannon Entropy H(X) = -Sigma p(x) log2 p(x) ===")

    def entropy(p):
        p = np.array(p, dtype=float)
        p = p / p.sum()
        return -np.sum(_xlogy(p, p))

    cases = {
        "Deterministic [1,0,0,0]":   [1, 0, 0, 0],
        "2-symbol {0.9,0.1}":        [0.9, 0.1],
        "2-symbol {0.5,0.5}":        [0.5, 0.5],
        "4-symbol uniform":           [0.25]*4,
        "8-symbol uniform":           [1/8]*8,
        "Non-uniform 4":              [0.5, 0.25, 0.125, 0.125],
    }
    print(f"  {'Distribution':<35} {'H (bits)':<12} {'H_max'}")
    for name, p in cases.items():
        H     = entropy(p)
        H_max = np.log2(len(p))
        print(f"  {name:<35} {H:<12.4f} {H_max:.4f}  ({H/H_max*100:.1f}% of max)")

    print("\n  Properties:")
    print(f"    H >= 0 always (equality iff deterministic)")
    print(f"    H <= log2|X| (equality iff uniform)")
    print(f"    More states -> higher maximum entropy")


# -- 2. Joint and conditional entropy -----------------------------------------
def joint_conditional_entropy():
    print("\n=== Joint and Conditional Entropy ===")
    # Joint distribution P(X,Y)
    Pxy = np.array([[0.1, 0.2, 0.0],
                    [0.2, 0.3, 0.1],
                    [0.0, 0.05, 0.05]])
    Pxy /= Pxy.sum()   # normalise

    Px  = Pxy.sum(axis=1)   # marginal P(X)
    Py  = Pxy.sum(axis=0)   # marginal P(Y)

    H_X  = -np.sum(_xlogy(Px, Px))
    H_Y  = -np.sum(_xlogy(Py, Py))
    H_XY = -np.sum(_xlogy(Pxy, Pxy))

    # Conditional H(Y|X) = H(X,Y) - H(X)
    H_Y_gX = H_XY - H_X
    H_X_gY = H_XY - H_Y

    print(f"  H(X)   = {H_X:.4f} bits")
    print(f"  H(Y)   = {H_Y:.4f} bits")
    print(f"  H(X,Y) = {H_XY:.4f} bits")
    print(f"  H(Y|X) = H(X,Y)-H(X) = {H_Y_gX:.4f} bits  (conditioning reduces entropy)")
    print(f"  H(X|Y) = H(X,Y)-H(Y) = {H_X_gY:.4f} bits")
    print(f"  Check: H(Y|X) <= H(Y): {H_Y_gX <= H_Y + 1e-10}")

    return H_X, H_Y, H_XY, H_Y_gX, H_X_gY, Px, Py, Pxy


# -- 3. Mutual Information -----------------------------------------------------
def mutual_information():
    print("\n=== Mutual Information I(X;Y) ===")
    H_X, H_Y, H_XY, H_Y_gX, H_X_gY, Px, Py, Pxy = joint_conditional_entropy()

    # I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    I_XY = H_X + H_Y - H_XY
    print(f"  I(X;Y) = H(X)+H(Y)-H(X,Y) = {I_XY:.4f} bits")
    print(f"         = H(Y)-H(Y|X)       = {H_Y - H_Y_gX:.4f}")
    print(f"         = H(X)-H(X|Y)       = {H_X - H_X_gY:.4f}")
    print(f"  I >= 0: {I_XY >= -1e-10}")
    print(f"  I = 0 iff X,Y independent")

    # Compare to independent case
    Pxy_indep = np.outer(Px, Py)
    H_XY_indep = -np.sum(_xlogy(Pxy_indep, Pxy_indep))
    I_indep = H_X + H_Y - H_XY_indep
    print(f"\n  If independent: I(X;Y) = {I_indep:.4f}  (~= 0)")


# -- 4. KL Divergence ---------------------------------------------------------
def kl_divergence():
    print("\n=== KL Divergence D_KL(P||Q) = Sigma p log(p/q) ===")

    def kl(p, q):
        p, q = np.array(p, float), np.array(q, float)
        p /= p.sum(); q /= q.sum()
        return np.sum(np.where(p == 0, 0, p * np.log2(p / (q + 1e-300))))

    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.25, 0.25, 0.25, 0.25])   # uniform

    print(f"  P = {p}  Q = {q} (uniform)")
    print(f"  D_KL(P||Q) = {kl(p, q):.4f} bits")
    print(f"  D_KL(Q||P) = {kl(q, p):.4f} bits  (asymmetric!)")
    print(f"  D_KL >= 0:  D_KL(P||Q)={kl(p,q)>=0}")
    print(f"  D_KL = 0 iff P = Q: {np.isclose(kl(p,p), 0)}")

    # KL between two Gaussians (analytical)
    mu1,sig1 = 0, 1
    mu2,sig2 = 1, 1.5
    kl_gauss = np.log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/(2*sig2**2) - 0.5
    print(f"\n  KL(N({mu1},{sig1}²) || N({mu2},{sig2}²)) = {kl_gauss:.4f} nats")
    kl_scipy  = stats.norm(mu1,sig1).entropy() - \
                np.sum(stats.norm(mu2,sig2).logpdf(np.linspace(-10,10,10000)) *
                       stats.norm(mu1,sig1).pdf(np.linspace(-10,10,10000))) * 20/10000
    print(f"  (numerical approximation: {kl_scipy:.4f} nats)")


# -- 5. Cross-entropy (ML/DL loss function) -----------------------------------
def cross_entropy():
    print("\n=== Cross-Entropy H(P,Q) = -Sigma p log q ===")
    print("  Used as loss in classification: H(y_true, y_pred)")
    print("  Relation: H(P,Q) = H(P) + D_KL(P||Q)")

    def cross_ent(p, q):
        p = np.array(p, float) / np.sum(p)
        q = np.array(q, float) / np.sum(q)
        return -np.sum(np.where(p==0, 0, p * np.log(q + 1e-300)))

    # Binary classification
    y_true  = np.array([1, 0, 0, 1, 1])   # one-hot for 5 samples
    y_pred1 = np.array([0.9, 0.1, 0.2, 0.8, 0.7])   # good preds
    y_pred2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])   # random preds
    y_pred3 = np.array([0.1, 0.9, 0.8, 0.2, 0.3])   # bad preds

    def binary_ce(y_true, y_pred):
        return -np.mean(y_true*np.log(y_pred+1e-9) + (1-y_true)*np.log(1-y_pred+1e-9))

    print(f"\n  Binary Cross-Entropy (5 samples):")
    for label, yp in [("good predictions", y_pred1),
                       ("random (0.5)",     y_pred2),
                       ("bad predictions",  y_pred3)]:
        print(f"    {label:<25}: BCE = {binary_ce(y_true, yp):.4f}")

    # Multinomial cross-entropy (softmax output)
    print(f"\n  Categorical Cross-Entropy (3-class):")
    y_true_cat  = np.array([0, 0, 1])   # class 2
    predictions = [
        ("confident correct",  np.array([0.02, 0.03, 0.95])),
        ("uniform",            np.array([1/3, 1/3, 1/3])),
        ("confident wrong",    np.array([0.95, 0.03, 0.02])),
    ]
    for label, pred in predictions:
        ce = -np.sum(y_true_cat * np.log(pred + 1e-9))
        print(f"    {label:<25}: CE = {ce:.4f}")


# -- 6. Information-theoretic quantities summary -------------------------------
def summary_table():
    print("\n=== Information Theory Summary ===")
    rows = [
        ("H(X)",       "-Sigma p log p",             "Self-information, uncertainty"),
        ("H(X,Y)",     "-SigmaSigma p(x,y) log p(x,y)", "Joint uncertainty"),
        ("H(Y|X)",     "H(X,Y) - H(X)",          "Remaining uncertainty in Y given X"),
        ("I(X;Y)",     "H(X)+H(Y)-H(X,Y)",       "Shared information (MI)"),
        ("D_KL(P||Q)", "Sigma p log(p/q)",            "How different Q is from P"),
        ("H(P,Q)",     "-Sigma p log q",              "CE loss in classification"),
    ]
    print(f"  {'Quantity':<12} {'Formula':<30} {'Interpretation'}")
    print("  " + "-"*75)
    for q, f, interp in rows:
        print(f"  {q:<12} {f:<30} {interp}")


if __name__ == "__main__":
    shannon_entropy()
    joint_conditional_entropy()
    mutual_information()
    kl_divergence()
    cross_entropy()
    summary_table()
