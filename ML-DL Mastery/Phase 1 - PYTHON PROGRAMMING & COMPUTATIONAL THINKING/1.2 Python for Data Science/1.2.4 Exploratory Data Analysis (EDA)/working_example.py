"""
Working Example: Exploratory Data Analysis (EDA)
Covers data profiling, distributions, correlations,
outlier detection, feature relationships, and EDA checklist.
"""
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Synthetic dataset ─────────────────────────────────────────────────────────
def make_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    age    = rng.integers(22, 65, n).astype(float)
    exp    = age - 22 + rng.normal(0, 2, n)
    exp    = np.clip(exp, 0, 40)
    salary = 30_000 + 2_500 * exp + rng.normal(0, 8_000, n)
    dept   = rng.choice(["Eng", "Marketing", "Sales", "HR"], n, p=[0.4,0.2,0.25,0.15])
    score  = rng.normal(75, 15, n).clip(0, 100)

    # inject missing & outliers
    idx_miss = rng.choice(n, 10, replace=False)
    score[idx_miss] = np.nan
    salary[:3] = [500_000, 1_200, -100]   # outliers

    return pd.DataFrame({
        "age": age, "experience": exp,
        "salary": salary, "dept": dept, "score": score
    })


# ── 1. Basic profiling ────────────────────────────────────────────────────────
def basic_profile(df):
    print("=== 1. Basic Profile ===")
    print(f"  Shape     : {df.shape}")
    print(f"  dtypes    :\n{df.dtypes}")
    print(f"\n  head(3):\n{df.head(3).to_string()}")
    print(f"\n  describe:\n{df.describe().round(2)}")
    print(f"\n  nulls:\n{df.isnull().sum()}")
    print(f"\n  duplicated rows: {df.duplicated().sum()}")


# ── 2. Univariate analysis ────────────────────────────────────────────────────
def univariate(df):
    print("\n=== 2. Univariate Analysis ===")
    # Numeric
    for col in ["age", "salary", "score"]:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        print(f"  {col:<12}: mean={s.mean():.2f}  median={s.median():.2f}  "
              f"std={s.std():.2f}  skew={s.skew():.2f}  "
              f"IQR={iqr:.2f}  missing={df[col].isnull().sum()}")

    # Categorical
    print(f"\n  dept value_counts:\n{df['dept'].value_counts()}")
    print(f"  dept %:\n{df['dept'].value_counts(normalize=True).mul(100).round(1)}")


# ── 3. Outlier detection ──────────────────────────────────────────────────────
def outlier_detection(df):
    print("\n=== 3. Outlier Detection ===")
    s = df["salary"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(s < lo) | (s > hi)]
    print(f"  salary IQR bounds: [{lo:.0f}, {hi:.0f}]")
    print(f"  outliers found   : {len(outliers)}")
    print(f"  outlier salaries : {outliers['salary'].values}")

    # Z-score method
    from scipy.stats import zscore
    z_scores = zscore(df["salary"].fillna(df["salary"].median()))
    z_outliers = (np.abs(z_scores) > 3).sum()
    print(f"  z-score |z|>3 outliers: {z_outliers}")


# ── 4. Bivariate / correlation analysis ──────────────────────────────────────
def bivariate(df):
    print("\n=== 4. Bivariate Analysis ===")
    numeric = df[["age", "experience", "salary", "score"]].dropna()
    corr    = numeric.corr()
    print(f"  correlation matrix:\n{corr.round(3)}")

    # By group
    group = df.groupby("dept")[["salary","score"]].agg(["mean","std"]).round(2)
    print(f"\n  by department:\n{group}")


# ── 5. Plot distributions and relationships ───────────────────────────────────
def eda_plots(df):
    print("\n=== 5. EDA Plots (saved to output_eda/) ===")
    numeric = df[["age", "experience", "salary", "score"]]

    # Histograms for all numeric columns
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Numeric Distributions")
    for ax, col in zip(axes.flat, numeric.columns):
        data = numeric[col].dropna()
        ax.hist(data, bins=25, edgecolor="white", color="steelblue", alpha=0.8)
        ax.axvline(data.mean(),   color="red",   linestyle="--", label=f"mean={data.mean():.0f}")
        ax.axvline(data.median(), color="green", linestyle=":",  label=f"med={data.median():.0f}")
        ax.set_title(col)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_distributions.png"), dpi=100)
    plt.close()
    print("  Saved: eda_distributions.png")

    # Salary by dept (boxplot)
    fig, ax = plt.subplots(figsize=(8, 4))
    groups  = [df[df["dept"] == d]["salary"].dropna() for d in df["dept"].unique()]
    labels  = list(df["dept"].unique())
    ax.boxplot(groups, labels=labels, patch_artist=True)
    ax.set_title("Salary by Department")
    ax.set_ylabel("Salary")
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_salary_by_dept.png"), dpi=100)
    plt.close()
    print("  Saved: eda_salary_by_dept.png")

    # Scatter: experience vs salary
    fig, ax = plt.subplots(figsize=(6, 5))
    for dept, grp in df.groupby("dept"):
        ax.scatter(grp["experience"], grp["salary"], label=dept, alpha=0.5, s=25)
    ax.set_xlabel("Experience (years)")
    ax.set_ylabel("Salary ($)")
    ax.set_title("Salary vs Experience")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_salary_vs_exp.png"), dpi=100)
    plt.close()
    print("  Saved: eda_salary_vs_exp.png")


# ── 6. EDA checklist ─────────────────────────────────────────────────────────
def eda_checklist():
    print("\n=== 6. EDA Checklist ===")
    checklist = [
        "[ ] Shape, dtypes, memory usage",
        "[ ] Missing values — count, %, pattern",
        "[ ] Duplicate rows",
        "[ ] Descriptive statistics (mean, std, quartiles, skew, kurtosis)",
        "[ ] Univariate distributions (histograms, KDE, value_counts for cats)",
        "[ ] Outlier detection (IQR, z-score, domain knowledge)",
        "[ ] Bivariate analysis — scatter, groupby, correlation matrix",
        "[ ] Categorical vs numeric — boxplots, violin plots",
        "[ ] Time patterns if date column present",
        "[ ] Feature engineering insights",
        "[ ] Notes for preprocessing (encoding, scaling, imputation)",
    ]
    for item in checklist:
        print(f"  {item}")


if __name__ == "__main__":
    df = make_dataset()
    basic_profile(df)
    univariate(df)
    outlier_detection(df)
    bivariate(df)
    eda_plots(df)
    eda_checklist()
