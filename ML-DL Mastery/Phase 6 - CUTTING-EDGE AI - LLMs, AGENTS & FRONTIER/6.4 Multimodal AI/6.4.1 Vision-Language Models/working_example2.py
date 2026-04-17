"""
Working Example 2: Vision-Language Models
CLIP-style cosine similarity between image and text embeddings
(random numpy vectors as proxies) and contrastive loss.
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


def clip_embed(seed, dim=128):
    """Proxy embedding: deterministic from seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-10)


def contrastive_loss(logits, temperature=0.07):
    """InfoNCE / contrastive loss for diagonal (matched) pairs."""
    n = logits.shape[0]
    labels = np.arange(n)
    # Row-wise softmax cross-entropy
    logits_scaled = logits / temperature
    log_sum_exp = np.log(np.sum(np.exp(logits_scaled - logits_scaled.max(axis=1, keepdims=True)),
                                axis=1)) + logits_scaled.max(axis=1)
    loss_i = log_sum_exp - logits_scaled[np.arange(n), labels]
    # Column-wise
    log_sum_exp_j = np.log(np.sum(np.exp(logits_scaled - logits_scaled.max(axis=0, keepdims=True)),
                                   axis=0)) + logits_scaled.max(axis=0)
    loss_j = log_sum_exp_j - logits_scaled[labels, np.arange(n)]
    return (loss_i.mean() + loss_j.mean()) / 2


def demo():
    print("=== Vision-Language Models: CLIP-style Similarity ===")
    dim = 128
    # Image embeddings: 6 synthetic images
    image_names = ["dog", "cat", "car", "airplane", "dog2", "cat2"]
    text_queries = ["a photo of a dog", "a photo of a cat", "a car on the road",
                    "an airplane in the sky", "puppy playing", "kitten sleeping"]

    # Related images have similar seeds
    img_seeds = [10, 20, 30, 40, 11, 21]  # dog=10,11, cat=20,21, car=30, plane=40
    txt_seeds = [10, 20, 30, 40, 11, 21]  # matched pairs

    img_emb = np.array([clip_embed(s, dim) for s in img_seeds])
    txt_emb = np.array([clip_embed(s, dim) for s in txt_seeds])

    # Cosine similarity matrix
    sim_matrix = img_emb @ txt_emb.T
    loss = contrastive_loss(sim_matrix)
    print(f"  Embedding dim: {dim}")
    print(f"  Contrastive loss: {loss:.4f}")
    print(f"  Diagonal (matched pairs) mean sim: {np.diag(sim_matrix).mean():.3f}")
    print(f"  Off-diagonal mean sim: {(sim_matrix.sum() - np.trace(sim_matrix)) / (36-6):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Similarity matrix heatmap
    im = axes[0].imshow(sim_matrix, cmap="RdYlGn", vmin=-0.3, vmax=1)
    axes[0].set(xticks=range(6), yticks=range(6),
                xticklabels=[t[:15] for t in text_queries],
                yticklabels=image_names,
                title="CLIP Cosine Similarity Matrix")
    axes[0].tick_params(axis="x", rotation=30)
    plt.colorbar(im, ax=axes[0])

    # Contrastive loss vs temperature
    temps = np.linspace(0.01, 0.5, 100)
    losses = [contrastive_loss(sim_matrix, t) for t in temps]
    axes[1].plot(temps, losses, color="steelblue", lw=2)
    axes[1].axvline(0.07, color="red", linestyle="--", label="τ=0.07")
    axes[1].set(xlabel="Temperature τ", ylabel="InfoNCE Loss",
                title="Contrastive Loss vs Temperature")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Top-1 image retrieval from text query
    top1 = np.argmax(sim_matrix, axis=0)
    correct = sum(1 for i, t in enumerate(top1) if t == i)
    axes[2].bar(["Correct", "Wrong"], [correct, 6 - correct],
                color=["mediumseagreen", "tomato"])
    axes[2].set(ylabel="Count", title=f"Text→Image Retrieval Accuracy ({correct}/6)")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "vision_language.png", dpi=100)
    plt.close()
    print("  Saved vision_language.png")


if __name__ == "__main__":
    demo()
