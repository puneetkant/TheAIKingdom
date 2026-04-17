"""
Working Example: Training Practices
Covers mini-batch construction, gradient clipping, mixed-precision concepts,
gradient accumulation, data pipelines, checkpointing, and debugging strategies.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os, time


def relu(z):    return np.maximum(0, z)
def relu_d(z):  return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── 1. Mini-batch gradient descent ───────────────────────────────────────────
def mini_batch_gd():
    print("=== Mini-Batch Gradient Descent ===")
    print("  Trade-off: batch_size ↑ → noisier but faster per epoch")

    rng = np.random.default_rng(0)
    n, d = 1000, 10
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + X[:, 2] > 0).astype(int)

    def make_batches(X, y, batch_size, rng):
        idx  = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    print(f"\n  {'batch_size':<14} {'batches/epoch':<16} {'Notes'}")
    for bs in [1, 32, 128, 512, 1000]:
        n_batches = int(np.ceil(n / bs))
        mode = ("Full GD" if bs==n else "SGD" if bs==1 else "Mini-batch")
        print(f"  {bs:<14} {n_batches:<16} {mode}")

    print(f"\n  Recommendations:")
    print(f"  - GPU: powers of 2 (32, 64, 128, 256) for memory alignment")
    print(f"  - Large batch needs LR scaling: lr_new = lr × (bs_new / bs_base)")
    print(f"  - Always shuffle data each epoch")


# ── 2. Gradient clipping ─────────────────────────────────────────────────────
def gradient_clipping():
    print("\n=== Gradient Clipping ===")
    print("  Prevents exploding gradients (common in RNNs)")
    print("  Norm clipping: g_clipped = g · max_norm / ||g|| if ||g|| > max_norm")
    print("  Value clipping: g_clipped = clip(g, -c, c)  per element")

    rng = np.random.default_rng(1)
    for scale in [1.0, 5.0, 50.0, 500.0]:
        g     = rng.standard_normal(100) * scale
        g_norm = np.linalg.norm(g)
        max_norm = 1.0
        # Norm clipping
        g_clipped = g * (max_norm / g_norm) if g_norm > max_norm else g
        g_norm_clipped = np.linalg.norm(g_clipped)
        # Value clipping
        g_val_clipped = np.clip(g, -max_norm, max_norm)
        g_val_norm    = np.linalg.norm(g_val_clipped)
        print(f"  ||g||={g_norm:>8.2f}  norm_clip={g_norm_clipped:>6.4f}  val_clip_norm={g_val_norm:>8.4f}")


# ── 3. Data pipeline and shuffling ───────────────────────────────────────────
def data_pipeline():
    print("\n=== Data Pipeline Best Practices ===")
    print("  1. Shuffle:        randomise order each epoch (prevents cycle bias)")
    print("  2. Normalize:      fit on train only; transform train+val+test")
    print("  3. Augmentation:   apply on-the-fly during training only")
    print("  4. Prefetch:       load next batch while GPU processes current")
    print("  5. Pin memory:     CPU→GPU transfer without copy (PyTorch)")
    print()

    # Simulate data generator
    rng = np.random.default_rng(2)
    n, d = 500, 10
    X    = rng.standard_normal((n, d))
    y    = rng.integers(0, 3, n)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)

    def dataset_gen(X, y, batch_size=32, shuffle=True, seed=0):
        rng = np.random.default_rng(seed)
        epoch = 0
        while True:
            idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for i in range(0, len(X), batch_size):
                yield X[idx[i:i+batch_size]], y[idx[i:i+batch_size]]
            epoch += 1

    gen = dataset_gen(X_norm, y, batch_size=32)
    for i in range(3):
        Xb, yb = next(gen)
        print(f"  Batch {i}: X={Xb.shape}  y_dist={dict(zip(*np.unique(yb, return_counts=True)))}")


# ── 4. Gradient accumulation ─────────────────────────────────────────────────
def gradient_accumulation():
    print("\n=== Gradient Accumulation ===")
    print("  Simulate large batch without large GPU memory:")
    print("  Accumulate gradients over N micro-batches, then update weights")
    print("  Effective batch = micro_batch_size × N_accumulation_steps")
    print()

    rng        = np.random.default_rng(3)
    micro_bs   = 16
    accum_steps = 4
    eff_bs     = micro_bs * accum_steps
    print(f"  micro_batch={micro_bs}  accum_steps={accum_steps}  effective_batch={eff_bs}")
    print()

    # Simulate
    total_grads = np.zeros(5)
    for step in range(accum_steps):
        # Simulate per-sample gradient
        g = rng.standard_normal(5)  # gradient from micro-batch
        total_grads += g / accum_steps  # normalise!
        print(f"  Step {step+1}: accumulated gradient = {total_grads.round(4)}")
    print(f"  → Apply update after {accum_steps} steps")


# ── 5. Checkpointing ─────────────────────────────────────────────────────────
def checkpointing():
    print("\n=== Model Checkpointing ===")
    print("  Save model state (weights + optimizer state) periodically")
    print("  Strategies:")
    print("    Save best:   save when val_loss improves")
    print("    Save every:  save every N epochs (full recovery)")
    print("    Top-K:       keep the best K checkpoints only")
    print()

    # Simulate checkpoint logic
    val_losses = [1.0, 0.8, 0.7, 0.75, 0.65, 0.66, 0.60, 0.58, 0.61, 0.57]
    best_loss  = float("inf")
    print(f"  {'Epoch':<8} {'Val Loss':<12} {'Action'}")
    for ep, vl in enumerate(val_losses, 1):
        if vl < best_loss:
            best_loss = vl
            action    = f"✓ SAVED (best={vl:.4f})"
        else:
            action    = "— skipped"
        print(f"  {ep:<8} {vl:<12} {action}")


# ── 6. Mixed precision training ───────────────────────────────────────────────
def mixed_precision():
    print("\n=== Mixed Precision Training (FP16/BF16) ===")
    print("  FP32: 32-bit float (standard); FP16: 16-bit float (2× memory, faster)")
    print()
    print("  Strategy:")
    print("  1. Forward pass in FP16 → faster matrix ops on GPU")
    print("  2. Loss in FP32         → numerical stability")
    print("  3. Gradient in FP16     → scaled by loss_scale to avoid underflow")
    print("  4. Update weights in FP32 master copy")
    print()
    print("  Loss scaling: multiply loss by S before backward; divide gradients by S")
    print("  S dynamically adjusted (GradScaler in PyTorch)")
    print()
    print("  Benefits: ~2× memory reduction, ~2-3× throughput on modern GPUs")
    print("  Supported by: PyTorch autocast, TF/Keras mixed_float16 policy")


# ── 7. Debugging training ─────────────────────────────────────────────────────
def debugging_training():
    print("\n=== Debugging Deep Learning Training ===")
    print()
    checks = [
        ("Loss doesn't decrease",   "Check LR (too small?), gradients not flowing, dead ReLU"),
        ("Loss explodes",           "Reduce LR, add gradient clipping, check data normalisation"),
        ("Loss decreases on train", "Overfit: add regularisation, more data, reduce capacity"),
        (" but not on val",         ""),
        ("NaN loss",                "LR too large, log(0) in loss, exploding gradients"),
        ("Slow training",           "Increase batch size, check data pipeline, use GPU"),
        ("Poor final accuracy",     "Tune HPs, try different architecture, check class balance"),
        ("Metrics inconsistent",    "Data leakage, wrong normalisation on val/test"),
    ]
    for issue, remedy in checks:
        if remedy:
            print(f"  Issue: {issue}")
            print(f"    Fix: {remedy}")
        else:
            print(f"         {issue}")
    print()
    print("  Always start with:")
    print("  1. Overfit a single batch (verify forward/backward correctness)")
    print("  2. Gradually scale data and model")
    print("  3. Monitor: train loss, val loss, gradient norms, weight norms")


if __name__ == "__main__":
    mini_batch_gd()
    gradient_clipping()
    data_pipeline()
    gradient_accumulation()
    checkpointing()
    mixed_precision()
    debugging_training()
