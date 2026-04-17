"""
Working Example: Core ML Concepts
Covers the ML taxonomy, bias-variance tradeoff, overfitting/underfitting,
the no-free-lunch theorem, PAC learning, and learning curves.
"""
import numpy as np
from scipy import stats
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_concepts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. ML Taxonomy ────────────────────────────────────────────────────────────
def ml_taxonomy():
    print("=== Machine Learning Taxonomy ===")
    taxonomy = {
        "Supervised": {
            "Classification": ["Logistic Regression","SVM","Decision Tree","KNN","Neural Net"],
            "Regression":     ["Linear Regression","Ridge","SVR","Random Forest"],
        },
        "Unsupervised": {
            "Clustering":     ["K-Means","DBSCAN","Hierarchical"],
            "Dim Reduction":  ["PCA","t-SNE","UMAP","Autoencoders"],
            "Density Est.":   ["GMM","KDE"],
        },
        "Semi-supervised": {"Mixed":   ["Self-training","Label Propagation"]},
        "Reinforcement":   {"Agents":  ["Q-Learning","Policy Gradient","PPO"]},
        "Self-supervised": {"Repr.":   ["Contrastive","BERT pretraining"]},
    }
    for cat, subcats in taxonomy.items():
        print(f"\n  {cat}:")
        for sub, algos in subcats.items():
            print(f"    {sub:<18}: {', '.join(algos)}")


# ── 2. Bias-variance tradeoff ─────────────────────────────────────────────────
def bias_variance_tradeoff():
    print("\n=== Bias-Variance Tradeoff ===")
    print("  MSE(θ̂) = Bias²(θ̂) + Var(θ̂) + σ²_noise")

    rng = np.random.default_rng(0)
    # True function: f(x) = sin(2πx)
    f    = lambda x: np.sin(2*np.pi*x)
    n    = 15
    M    = 200    # repeated experiments
    noise_std = 0.3

    xs_test = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(1, 3, figsize=(14,4))

    for ax, (degree, label) in zip(axes, [(1,"Underfit (d=1)"), (4,"Good fit (d=4)"), (15,"Overfit (d=15)")]):
        predictions = []
        for _ in range(M):
            x_train = rng.uniform(0, 1, n)
            y_train = f(x_train) + rng.normal(0, noise_std, n)
            coeffs  = np.polyfit(x_train, y_train, degree)
            y_pred  = np.polyval(coeffs, xs_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)
        mean_pred   = predictions.mean(axis=0)
        var_pred    = predictions.var(axis=0)
        bias2       = (mean_pred - f(xs_test))**2

        mse   = np.mean(bias2 + var_pred + noise_std**2)
        bias2m = np.mean(bias2)
        varm  = np.mean(var_pred)
        print(f"  {label}: MSE={mse:.4f}  Bias²={bias2m:.4f}  Var={varm:.4f}  Noise={noise_std**2:.4f}")

        ax.plot(xs_test, f(xs_test), 'k-', lw=2, label='True f')
        ax.plot(xs_test, mean_pred, 'b-', lw=2, label='Mean pred')
        ax.fill_between(xs_test,
                        mean_pred - 2*np.sqrt(var_pred),
                        mean_pred + 2*np.sqrt(var_pred),
                        alpha=0.2, color='blue', label='±2 std')
        ax.set(title=f"{label}\nMSE={mse:.3f}", ylim=(-3,3))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_variance.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Saved: {path}")


# ── 3. Overfitting and regularisation ────────────────────────────────────────
def overfitting_demo():
    print("\n=== Overfitting and Regularisation ===")
    rng = np.random.default_rng(1)
    n   = 20
    x   = rng.uniform(0, 1, n)
    y   = np.sin(2*np.pi*x) + rng.normal(0, 0.3, n)
    x_t = np.linspace(0, 1, 200)

    degrees = [1, 3, 9, 15]
    print(f"  {'Degree':<8} {'Train RMSE':<14} {'Test RMSE':<14} {'Status'}")
    for d in degrees:
        coeffs   = np.polyfit(x, y, d)
        train_rm = np.sqrt(np.mean((np.polyval(coeffs, x) - y)**2))
        test_rm  = np.sqrt(np.mean((np.polyval(coeffs, x_t) - np.sin(2*np.pi*x_t))**2))
        status   = "underfit" if d < 3 else ("good" if d <= 5 else "overfit")
        print(f"  {d:<8} {train_rm:<14.4f} {test_rm:<14.4f} {status}")


# ── 4. No Free Lunch theorem illustration ─────────────────────────────────────
def no_free_lunch():
    print("\n=== No Free Lunch Theorem ===")
    print("  NFL: averaged over ALL possible problems, no algorithm is better than random.")
    print("  → Domain knowledge and inductive bias are essential.")
    print("  → Choose model family matching your problem structure.")
    print()

    # Illustrate: KNN vs Linear for linear vs circular data
    rng = np.random.default_rng(2)
    n   = 100

    # Linear separable
    X_lin = rng.uniform(-1, 1, (n, 2))
    y_lin = (X_lin[:, 0] + X_lin[:, 1] > 0).astype(int)

    # Circular separable
    angles = rng.uniform(0, 2*np.pi, n)
    radii  = rng.uniform(0, 1, n)
    X_cir  = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
    y_cir  = (radii > 0.5).astype(int)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    for name, X, y in [("Linear data", X_lin, y_lin), ("Circular data", X_cir, y_cir)]:
        print(f"  {name}:")
        for clf_name, clf in [("Logistic Reg", LogisticRegression()),
                               ("KNN(k=5)",    KNeighborsClassifier(n_neighbors=5))]:
            acc = cross_val_score(clf, X, y, cv=5).mean()
            print(f"    {clf_name:<14}: acc={acc:.4f}")


# ── 5. PAC learning and VC dimension ─────────────────────────────────────────
def pac_learning():
    print("\n=== PAC Learning & VC Dimension ===")
    print("  PAC: Probably Approximately Correct learning")
    print("  With n samples, ε error, δ failure prob:")
    print("  n ≥ (1/ε)[log(|H|/δ)]  (finite H)")
    print("  n ≥ (1/ε)[d_VC·log(2/δ)]  (infinite H, d_VC = VC dimension)")
    print()

    # Sample complexity bounds
    eps, delta = 0.05, 0.05
    vc_dims = {"Halfplanes in R²": 3, "Decision stumps": 1,
               "SVM (linear kernel)": None, "1-layer NN (k nodes)": "k+1"}

    print(f"  ε={eps}, δ={delta}")
    for name, d_vc in vc_dims.items():
        if isinstance(d_vc, int):
            n_req = int(np.ceil((1/eps) * (d_vc * np.log(2/delta))))
            print(f"  {name:<28}: d_VC={d_vc}  n_req≈{n_req}")
        else:
            print(f"  {name:<28}: d_VC={d_vc} (depends on parameters)")


# ── 6. Learning curves ────────────────────────────────────────────────────────
def learning_curves():
    print("\n=== Learning Curves ===")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    rng  = np.random.default_rng(3)
    x_all = rng.uniform(0, 1, 300)
    y_all = np.sin(2*np.pi*x_all) + rng.normal(0, 0.3, 300)
    X_all = x_all.reshape(-1, 1)

    x_test = np.linspace(0, 1, 100)
    y_test = np.sin(2*np.pi*x_test)
    X_test = x_test.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(4), Ridge(alpha=0.01))

    print(f"  {'n_train':<10} {'Train RMSE':<14} {'Test RMSE'}")
    for n in [5, 10, 20, 50, 100, 200, 300]:
        X_tr, y_tr = X_all[:n], y_all[:n]
        model.fit(X_tr, y_tr)
        tr_rmse = np.sqrt(np.mean((model.predict(X_tr) - y_tr)**2))
        te_rmse = np.sqrt(np.mean((model.predict(X_test) - y_test)**2))
        print(f"  {n:<10} {tr_rmse:<14.4f} {te_rmse:.4f}")


if __name__ == "__main__":
    ml_taxonomy()
    bias_variance_tradeoff()
    overfitting_demo()
    no_free_lunch()
    pac_learning()
    learning_curves()
