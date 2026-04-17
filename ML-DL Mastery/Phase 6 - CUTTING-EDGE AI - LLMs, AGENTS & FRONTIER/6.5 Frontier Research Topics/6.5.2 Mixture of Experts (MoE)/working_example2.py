"""
Working Example 2: Mixture of Experts (MoE)
Top-k gating with load balancing loss.
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


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def top_k_gate(logits, k):
    """Top-k gating: keep top-k experts, zero-out the rest, then normalise."""
    n, E = logits.shape
    probs = softmax(logits)  # (n, E)
    top_k_idx = np.argsort(probs, axis=1)[:, -k:]
    gate = np.zeros_like(probs)
    for i in range(n):
        gate[i, top_k_idx[i]] = probs[i, top_k_idx[i]]
    # Normalise
    gate /= (gate.sum(axis=1, keepdims=True) + 1e-10)
    return gate, top_k_idx


def load_balance_loss(gate):
    """Auxiliary loss = E * mean(f_i * P_i) where f=fraction, P=prob."""
    E = gate.shape[1]
    f = (gate > 0).mean(axis=0)     # fraction of tokens assigned to each expert
    P = gate.mean(axis=0)           # mean gate weight per expert
    return E * np.dot(f, P)


def expert_forward(x, W):
    """Simple linear expert."""
    return np.tanh(x @ W)


def demo():
    print("=== Mixture of Experts (MoE) ===")
    rng = np.random.default_rng(42)
    n_tokens = 64
    d_model = 16
    n_experts = 8
    k = 2  # top-k experts

    # Tokens and gate parameters
    tokens = rng.standard_normal((n_tokens, d_model))
    W_gate = rng.standard_normal((d_model, n_experts)) * 0.5

    # Expert weights
    expert_weights = [rng.standard_normal((d_model, d_model)) * 0.1
                      for _ in range(n_experts)]

    # Gate
    logits = tokens @ W_gate  # (n_tokens, n_experts)
    gate, top_k_idx = top_k_gate(logits, k)
    lb_loss = load_balance_loss(gate)

    print(f"  Tokens: {n_tokens}, Experts: {n_experts}, k={k}")
    print(f"  Load balance loss: {lb_loss:.4f}")

    # Expert utilisation
    expert_counts = np.zeros(n_experts)
    for i in range(n_tokens):
        for idx in top_k_idx[i]:
            expert_counts[idx] += 1
    print(f"  Expert usage: {expert_counts.astype(int)}")
    print(f"  Load std: {expert_counts.std():.2f}")

    # MoE forward pass
    output = np.zeros_like(tokens)
    for i in range(n_tokens):
        for j, eidx in enumerate(top_k_idx[i]):
            output[i] += gate[i, eidx] * expert_forward(tokens[i], expert_weights[eidx])

    # Sweep k
    k_vals = range(1, n_experts + 1)
    lb_losses = []
    load_stds = []
    for kk in k_vals:
        g, tidx = top_k_gate(logits, kk)
        lb_losses.append(load_balance_loss(g))
        ec = np.zeros(n_experts)
        for i in range(n_tokens):
            ec[tidx[i]] += 1
        load_stds.append(ec.std())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Expert utilisation bar chart
    axes[0][0].bar(range(n_experts), expert_counts, color="steelblue", edgecolor="k")
    axes[0][0].axhline(n_tokens * k / n_experts, color="red", linestyle="--",
                        label=f"Perfect balance ({n_tokens*k//n_experts})")
    axes[0][0].set(xlabel="Expert ID", ylabel="# Tokens Assigned",
                   title=f"Expert Utilisation (k={k})")
    axes[0][0].legend()
    axes[0][0].grid(True, axis="y", alpha=0.3)

    # Gate weight heatmap
    im = axes[0][1].imshow(gate[:20].T, cmap="Blues", aspect="auto")
    axes[0][1].set(xlabel="Token", ylabel="Expert",
                   title="Gate Weights (first 20 tokens)")
    plt.colorbar(im, ax=axes[0][1])

    # Load balance loss vs k
    axes[1][0].plot(k_vals, lb_losses, "o-", color="tomato", lw=2)
    axes[1][0].set(xlabel="k (# active experts)", ylabel="Load Balance Loss",
                   title="Auxiliary Loss vs Top-k")
    axes[1][0].grid(True, alpha=0.3)

    # Load std vs k
    axes[1][1].plot(k_vals, load_stds, "s-", color="mediumseagreen", lw=2)
    axes[1][1].set(xlabel="k (# active experts)", ylabel="Load Std Dev",
                   title="Expert Load Imbalance vs Top-k")
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "moe_gating.png", dpi=100)
    plt.close()
    print("  Saved moe_gating.png")


if __name__ == "__main__":
    demo()
