"""
Working Example 2: Multimodal Embeddings and Retrieval
Cross-modal retrieval using numpy-based embeddings,
recall@k metric, and embedding space visualisation.
Run: python working_example2.py
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


def unit_embed(seed, dim=64):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-10)


def recall_at_k(sim_matrix, k):
    """Recall@k for text->image retrieval. Diagonal is ground truth."""
    n = sim_matrix.shape[0]
    hits = 0
    for i in range(n):
        top_k = np.argsort(sim_matrix[i])[::-1][:k]
        if i in top_k:
            hits += 1
    return hits / n


def demo():
    print("=== Multimodal Embeddings and Retrieval ===")
    n_items = 20
    dim = 128

    # Matched image-text pairs (same base seed for alignment)
    img_emb = np.array([unit_embed(i * 2, dim) for i in range(n_items)])
    txt_emb = np.array([unit_embed(i * 2, dim) for i in range(n_items)])  # perfect match

    # Partially misaligned retrieval
    txt_emb_noisy = np.array([
        unit_embed(i * 2 + (0 if i % 3 != 0 else 999), dim) for i in range(n_items)
    ])  # every 3rd item misaligned

    # Cosine similarity matrices
    sim_perfect = img_emb @ txt_emb.T
    sim_noisy = img_emb @ txt_emb_noisy.T

    ks = [1, 2, 3, 5, 10]
    r_perfect = [recall_at_k(sim_perfect, k) for k in ks]
    r_noisy = [recall_at_k(sim_noisy, k) for k in ks]

    for k, rp, rn in zip(ks, r_perfect, r_noisy):
        print(f"  Recall@{k}: perfect={rp:.2f}, noisy={rn:.2f}")

    # Mean reciprocal rank
    def mrr(sim):
        n = sim.shape[0]
        rrs = []
        for i in range(n):
            rank = np.where(np.argsort(sim[i])[::-1] == i)[0][0] + 1
            rrs.append(1 / rank)
        return np.mean(rrs)

    mrr_p = mrr(sim_perfect)
    mrr_n = mrr(sim_noisy)
    print(f"\n  MRR: perfect={mrr_p:.3f}, noisy={mrr_n:.3f}")

    # 2D PCA proxy visualisation
    def pca2d(emb):
        centered = emb - emb.mean(0)
        cov = centered.T @ centered / len(centered)
        vals, vecs = np.linalg.eigh(cov)
        return centered @ vecs[:, -2:]

    img_2d = pca2d(img_emb)
    txt_2d = pca2d(txt_emb)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Recall@k
    axes[0][0].plot(ks, r_perfect, "o-", color="steelblue", lw=2, label="Perfect Align")
    axes[0][0].plot(ks, r_noisy, "s--", color="tomato", lw=2, label="Noisy Align")
    axes[0][0].set(xlabel="k", ylabel="Recall@k", title="Cross-Modal Retrieval Recall@k")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # Similarity heatmap (top-left 10x10)
    im = axes[0][1].imshow(sim_noisy[:10, :10], cmap="RdYlGn", vmin=-0.5, vmax=1)
    axes[0][1].set(title="Cosine Similarity (noisy, first 10 pairs)",
                   xlabel="Text Index", ylabel="Image Index")
    plt.colorbar(im, ax=axes[0][1])

    # Embedding scatter
    axes[1][0].scatter(img_2d[:, 0], img_2d[:, 1], marker="o", color="steelblue",
                        alpha=0.7, label="Images", s=50)
    axes[1][0].scatter(txt_2d[:, 0], txt_2d[:, 1], marker="^", color="tomato",
                        alpha=0.7, label="Texts", s=50)
    for i in range(n_items):
        axes[1][0].plot([img_2d[i, 0], txt_2d[i, 0]], [img_2d[i, 1], txt_2d[i, 1]],
                         "k-", alpha=0.1, lw=0.8)
    axes[1][0].set(title="Embedding Space (PCA 2D)", xlabel="PC1", ylabel="PC2")
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    # MRR bar chart
    axes[1][1].bar(["Perfect Align", "Noisy Align"], [mrr_p, mrr_n],
                    color=["steelblue", "tomato"])
    axes[1][1].set(ylabel="Mean Reciprocal Rank", title="MRR Comparison")
    axes[1][1].set_ylim(0, 1)
    axes[1][1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([mrr_p, mrr_n]):
        axes[1][1].text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT / "multimodal_retrieval.png", dpi=100)
    plt.close()
    print("  Saved multimodal_retrieval.png")


if __name__ == "__main__":
    demo()
