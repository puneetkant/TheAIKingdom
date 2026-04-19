"""
Working Example: Data Visualization
Covers Matplotlib figures/axes, common plot types, styling,
subplots, saving figures, and Seaborn statistical plots.
All output is saved to files (no GUI required).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# -- 1. Line plot --------------------------------------------------------------
def line_plot():
    print("=== Line Plot ===")
    x = np.linspace(0, 4 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, np.sin(x),  label="sin(x)", color="royalblue", linewidth=2)
    ax.plot(x, np.cos(x),  label="cos(x)", color="crimson",   linestyle="--")
    ax.plot(x, np.sin(2*x),label="sin(2x)",color="seagreen",  linestyle=":")
    ax.set_title("Trigonometric Functions", fontsize=14)
    ax.set_xlabel("x (radians)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    save("01_line_plot")


# -- 2. Scatter plot -----------------------------------------------------------
def scatter_plot():
    print("=== Scatter Plot ===")
    rng = np.random.default_rng(0)
    n   = 200
    x   = rng.normal(0, 1, n)
    y   = 2 * x + rng.normal(0, 0.5, n)
    col = rng.uniform(0, 1, n)   # colour channel

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, c=col, cmap="viridis", alpha=0.7, edgecolors="none", s=40)
    plt.colorbar(sc, ax=ax, label="random hue")
    # Trend line
    m, b = np.polyfit(x, y, 1)
    ax.plot(np.sort(x), m * np.sort(x) + b, "r--", label=f"fit: y={m:.2f}x+{b:.2f}")
    ax.set_title("Scatter with Trend Line")
    ax.legend()
    save("02_scatter_plot")


# -- 3. Bar chart --------------------------------------------------------------
def bar_chart():
    print("=== Bar Chart ===")
    categories = ["Python", "NumPy", "Pandas", "Matplotlib", "scikit-learn"]
    scores     = [95, 88, 82, 75, 90]
    colors     = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(categories, scores, color=colors, edgecolor="white", width=0.6)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(score), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Library Proficiency Scores")
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y", alpha=0.3)
    save("03_bar_chart")


# -- 4. Histogram --------------------------------------------------------------
def histogram():
    print("=== Histogram ===")
    rng  = np.random.default_rng(1)
    data = rng.normal(loc=170, scale=10, size=1000)

    fig, ax = plt.subplots(figsize=(7, 4))
    n, bins, patches = ax.hist(data, bins=30, edgecolor="white", color="steelblue", alpha=0.8)
    # Overlay theoretical normal curve
    from scipy.stats import norm
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma) * len(data) * (bins[1]-bins[0]),
            "r-", linewidth=2, label=f"N({mu:.1f}, {sigma:.1f})")
    ax.axvline(mu, color="gold", linestyle="--", label=f"mean={mu:.1f}")
    ax.set_title("Height Distribution (simulated)")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Count")
    ax.legend()
    save("04_histogram")


# -- 5. Subplots grid ----------------------------------------------------------
def subplots_grid():
    print("=== Subplots Grid ===")
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("2×2 Subplot Grid", fontsize=15)

    # (0,0) — line
    x = np.linspace(0, 6, 100)
    axes[0, 0].plot(x, np.exp(-0.3*x)*np.sin(3*x), color="purple")
    axes[0, 0].set_title("Damped Oscillation")

    # (0,1) — scatter
    x2 = rng.uniform(-3, 3, 150)
    axes[0, 1].scatter(x2, x2**2 + rng.normal(0, 1, 150),
                       alpha=0.5, color="teal", s=20)
    axes[0, 1].set_title("Scatter: y ~= x²")

    # (1,0) — bar
    cats = ["A","B","C","D","E"]
    vals = rng.integers(20, 80, len(cats))
    axes[1, 0].barh(cats, vals, color="coral")
    axes[1, 0].set_title("Horizontal Bar")

    # (1,1) — pie
    sizes  = [30, 20, 25, 15, 10]
    labels = ["Python","Java","C++","JS","Other"]
    axes[1, 1].pie(sizes, labels=labels, autopct="%1.1f%%",
                   startangle=140, colors=plt.cm.Pastel1.colors[:5])
    axes[1, 1].set_title("Language Share")

    plt.tight_layout()
    save("05_subplots_grid")


# -- 6. Seaborn-style plots using Matplotlib ----------------------------------
def seaborn_demo():
    """Use seaborn if available, else skip gracefully."""
    print("=== Seaborn (statistical plots) ===")
    try:
        import seaborn as sns
        import pandas as pd

        tips = sns.load_dataset("tips")   # built-in sample dataset

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Seaborn Statistical Plots", fontsize=13)

        # boxplot
        sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0],
                    palette="Set2", order=["Thur","Fri","Sat","Sun"])
        axes[0].set_title("Total Bill by Day")

        # violin
        sns.violinplot(data=tips, x="sex", y="tip", ax=axes[1],
                       palette="muted", inner="quartile")
        axes[1].set_title("Tip by Gender")

        # scatter with regression
        sns.regplot(data=tips, x="total_bill", y="tip", ax=axes[2],
                    scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
        axes[2].set_title("Tip vs Bill")

        plt.tight_layout()
        save("06_seaborn_plots")
    except ImportError:
        print("  seaborn not installed — skipping. (pip install seaborn)")
    except Exception as e:
        print(f"  seaborn demo skipped: {e}")


# -- 7. Heatmap (manually with matplotlib) ------------------------------------
def heatmap():
    print("=== Heatmap (Correlation Matrix) ===")
    rng = np.random.default_rng(7)
    # Create a synthetic correlation-like matrix
    raw = rng.standard_normal((50, 5))
    cols = np.array(["Feature A","Feature B","Feature C","Feature D","Feature E"])
    corr = np.corrcoef(raw.T)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation Heatmap")
    save("07_heatmap")


if __name__ == "__main__":
    print(f"Saving plots to: {OUTPUT_DIR}\n")
    line_plot()
    scatter_plot()
    bar_chart()
    histogram()
    subplots_grid()
    seaborn_demo()
    heatmap()
    print("\nAll plots saved.")
