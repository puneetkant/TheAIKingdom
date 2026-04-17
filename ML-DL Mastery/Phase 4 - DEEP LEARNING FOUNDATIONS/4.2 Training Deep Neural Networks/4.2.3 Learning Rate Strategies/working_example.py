"""
Working Example: Learning Rate Strategies
Covers constant LR, step decay, exponential decay, cosine annealing,
warm-up, cyclical LR, one-cycle policy, and learning rate finder.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_lr_strategies")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Learning rate schedulers ──────────────────────────────────────────────────
def constant_lr(epoch, base_lr=0.01, **_):
    return base_lr

def step_decay(epoch, base_lr=0.1, drop=0.5, step_size=10, **_):
    return base_lr * drop ** (epoch // step_size)

def exponential_decay(epoch, base_lr=0.1, decay=0.05, **_):
    return base_lr * np.exp(-decay * epoch)

def polynomial_decay(epoch, base_lr=0.1, end_lr=0.001, power=2, total=100, **_):
    epoch = min(epoch, total)
    return (base_lr - end_lr) * (1 - epoch/total)**power + end_lr

def cosine_annealing(epoch, base_lr=0.1, min_lr=1e-4, T_max=100, **_):
    return min_lr + 0.5*(base_lr - min_lr)*(1 + np.cos(np.pi * epoch / T_max))

def cosine_with_warmup(epoch, base_lr=0.1, min_lr=1e-5, warmup=10, T_max=100, **_):
    if epoch < warmup:
        return base_lr * epoch / warmup
    t = epoch - warmup
    T = T_max - warmup
    return min_lr + 0.5*(base_lr-min_lr)*(1 + np.cos(np.pi * t / T))

def cyclical_lr(epoch, base_lr=1e-4, max_lr=0.01, step_size=20, **_):
    cycle  = epoch // (2 * step_size)
    x      = abs(epoch / step_size - 2 * cycle - 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)

def one_cycle(epoch, base_lr=1e-5, max_lr=0.1, total=100, **_):
    """1-cycle: warm-up to max_lr, then cosine decay to base_lr/div."""
    pct_up = 0.3
    div    = max_lr / base_lr
    final_div = 1e4
    if epoch < int(total * pct_up):
        t = epoch / (total * pct_up)
        return base_lr + (max_lr - base_lr) * t
    else:
        t = (epoch - total*pct_up) / (total*(1-pct_up))
        return base_lr + (max_lr - base_lr) * 0.5*(1 + np.cos(np.pi*t))


# ── 1. Plot all schedules ─────────────────────────────────────────────────────
def plot_schedules():
    print("=== Learning Rate Schedules ===")
    epochs = np.arange(100)
    schedules = {
        "Constant (0.01)":     [constant_lr(e) for e in epochs],
        "Step Decay":          [step_decay(e) for e in epochs],
        "Exponential Decay":   [exponential_decay(e) for e in epochs],
        "Polynomial Decay":    [polynomial_decay(e) for e in epochs],
        "Cosine Annealing":    [cosine_annealing(e) for e in epochs],
        "Cosine + Warmup":     [cosine_with_warmup(e) for e in epochs],
        "Cyclical LR":         [cyclical_lr(e) for e in epochs],
        "One-Cycle Policy":    [one_cycle(e) for e in epochs],
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for ax, (name, lrs) in zip(axes, schedules.items()):
        ax.plot(epochs, lrs, lw=2)
        ax.set_title(name, fontsize=9); ax.set(xlabel="Epoch", ylabel="LR")
        ax.grid(True, alpha=0.3)
        print(f"  {name:<30}: min={min(lrs):.5f}  max={max(lrs):.5f}")

    plt.suptitle("Learning Rate Schedules", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lr_schedules.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Plot saved: {path}")


# ── 2. Why learning rate matters ──────────────────────────────────────────────
def lr_sensitivity():
    print("\n=== Learning Rate Sensitivity ===")
    print("  Too small: slow convergence, may get stuck")
    print("  Too large: diverges or oscillates")
    print("  Just right: fast convergence to good minimum")

    # Simulate loss curves for different LRs on a quadratic loss
    def train(lr, n_steps=200):
        w = 2.0   # start far from optimum w*=0
        losses = []
        for _ in range(n_steps):
            loss = w**2
            grad = 2*w
            w   -= lr * grad
            losses.append(loss)
        return losses

    print(f"\n  {'LR':<10} {'Final loss':<15} {'Behaviour'}")
    for lr, label in [(0.0001, "too small"), (0.01, "small"), (0.1, "good"),
                      (0.9, "large"), (1.1, "diverging")]:
        losses = train(lr)
        print(f"  {lr:<10} {losses[-1]:<15.6f} {label}")


# ── 3. Warm-up strategy ───────────────────────────────────────────────────────
def warmup_strategy():
    print("\n=== Learning Rate Warm-up ===")
    print("  Start with very small LR; linearly/cosine increase to target LR")
    print("  Prevents large gradient updates in early training (poor init)")
    print()
    print("  BERT, GPT-2, ViT all use linear warmup then cosine decay")
    print()
    warmup_epochs = 10
    base_lr = 0.1
    epochs  = np.arange(100)
    lrs = [cosine_with_warmup(e, base_lr=base_lr, warmup=warmup_epochs) for e in epochs]
    print(f"  Warmup ({warmup_epochs} epochs): LR goes 0 → {base_lr}")
    print(f"  Then cosine decay to min_lr=1e-5")
    for e in [0, 5, 10, 25, 50, 75, 99]:
        print(f"  Epoch {e:>3}: LR={lrs[e]:.6f}")


# ── 4. Cyclical LR (Smith 2017) ───────────────────────────────────────────────
def cyclical_lr_demo():
    print("\n=== Cyclical Learning Rates (CLR) ===")
    print("  Oscillates LR between base_lr and max_lr")
    print("  Benefit: escapes saddle points; no need to tune LR schedule")
    print()
    print("  Step size: half the cycle; rule of thumb = 2-8× steps/epoch")
    print("  3 modes: triangular, triangular2 (halve max each cycle), exp_range")
    print()
    lrs = [cyclical_lr(e, base_lr=1e-4, max_lr=0.01, step_size=20) for e in range(100)]
    for e in [0, 10, 20, 30, 40, 50]:
        print(f"  Epoch {e:>3}: LR={lrs[e]:.5f}")


# ── 5. LR Finder ──────────────────────────────────────────────────────────────
def lr_finder():
    print("\n=== Learning Rate Finder (Smith, fast.ai) ===")
    print("  Exponentially increase LR from min to max; plot loss vs LR")
    print("  Choose LR just before loss starts to diverge (steepest descent)")
    print()
    print("  Algorithm:")
    print("  1. Start with LR_min (e.g. 1e-7)")
    print("  2. Each mini-batch: LR *= factor (e.g. 1.05)")
    print("  3. Record loss; stop when loss > 4× best_loss")
    print("  4. Plot: choose LR at steepest negative slope")

    # Simulate LR finder curve
    lrs_tried = np.logspace(-7, 1, 200)
    losses = []
    for lr in lrs_tried:
        if lr < 1e-4:
            L = 2.0 - lr*1e4*0.5
        elif lr < 1e-2:
            L = 1.5 - np.log10(lr)*0.2
        elif lr < 0.1:
            L = 1.0 + (lr - 0.01)*5
        else:
            L = 3.0 + lr*10
        losses.append(L)

    losses = np.array(losses) + np.random.default_rng(0).normal(0, 0.05, len(lrs_tried))
    best_idx = np.argmin(losses)
    opt_lr   = lrs_tried[best_idx]
    print(f"\n  (Simulated) Optimal LR ≈ {opt_lr:.2e}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(lrs_tried, losses, lw=2)
    ax.axvline(opt_lr, color='r', lw=2, linestyle='--', label=f"Min loss LR={opt_lr:.2e}")
    ax.set(xlabel="Learning Rate (log scale)", ylabel="Loss", title="LR Finder Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lr_finder.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot saved: {path}")


# ── 6. One-Cycle Policy ───────────────────────────────────────────────────────
def one_cycle_policy():
    print("\n=== One-Cycle Policy (Smith 2018) ===")
    print("  Phase 1: increase LR from base to max (30% of steps)")
    print("  Phase 2: decrease LR back to base/div (70% of steps)")
    print("  Momentum: decrease during phase 1, increase during phase 2")
    print()
    print("  Benefits: achieves super-convergence (10× fewer epochs)")
    print("  Used by: fastai, PyTorch Lightning, transformers")
    for e in [0, 15, 30, 50, 75, 99]:
        lr = one_cycle(e, base_lr=1e-5, max_lr=0.1, total=100)
        print(f"  Epoch {e:>3}: LR={lr:.6f}")


if __name__ == "__main__":
    plot_schedules()
    lr_sensitivity()
    warmup_strategy()
    cyclical_lr_demo()
    lr_finder()
    one_cycle_policy()
