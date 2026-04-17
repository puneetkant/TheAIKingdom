"""
Working Example: Activation Functions
Covers sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU, Swish, softmax,
their properties, gradients, dying ReLU, and how to choose.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_activations")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Activation functions ──────────────────────────────────────────────────────
def sigmoid(z):       return 1 / (1 + np.exp(-z.clip(-500,500)))
def sigmoid_d(z):     s = sigmoid(z); return s * (1 - s)
def tanh_d(z):        return 1 - np.tanh(z)**2
def relu(z):          return np.maximum(0, z)
def relu_d(z):        return (z > 0).astype(float)
def leaky_relu(z, a=0.01): return np.where(z > 0, z, a*z)
def leaky_relu_d(z, a=0.01): return np.where(z > 0, 1, a)
def elu(z, a=1.0):    return np.where(z > 0, z, a*(np.exp(z.clip(-500,0))-1))
def elu_d(z, a=1.0):  return np.where(z > 0, 1, elu(z, a) + a)
def gelu(z):          return 0.5*z*(1 + np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*z**3)))
def swish(z):         return z * sigmoid(z)
def selu(z, l=1.0507, a=1.6733): return l * np.where(z > 0, z, a*(np.exp(z.clip(-500,0))-1))


# ── 1. Properties of activation functions ────────────────────────────────────
def activation_properties():
    print("=== Activation Function Properties ===")
    z = np.linspace(-5, 5, 1000)
    functions = {
        "Sigmoid":     (sigmoid(z),       sigmoid_d(z)),
        "Tanh":        (np.tanh(z),        tanh_d(z)),
        "ReLU":        (relu(z),           relu_d(z)),
        "Leaky ReLU":  (leaky_relu(z),     leaky_relu_d(z)),
        "ELU":         (elu(z),            elu_d(z)),
        "GELU":        (gelu(z),           None),
        "Swish":       (swish(z),          None),
        "SELU":        (selu(z),           None),
    }
    print(f"\n  {'Function':<15} {'Range':<20} {'Max grad':>10} {'Mean grad z∈[-1,1]':>20}")
    for name, (f, fd) in functions.items():
        fmin, fmax = f.min(), f.max()
        frange = f"[{fmin:.2f}, {fmax:.2f}]"
        if fd is not None:
            mask = (z >= -1) & (z <= 1)
            mg   = fd[mask].mean()
            mg_max = fd.max()
            print(f"  {name:<15} {frange:<20} {mg_max:>10.4f} {mg:>20.4f}")
        else:
            print(f"  {name:<15} {frange:<20} {'—':>10} {'—':>20}")


# ── 2. Visualise activations ─────────────────────────────────────────────────
def plot_activations():
    z = np.linspace(-4, 4, 300)
    funcs = [
        ("Sigmoid", sigmoid(z), sigmoid_d(z)),
        ("Tanh",    np.tanh(z), tanh_d(z)),
        ("ReLU",    relu(z),    relu_d(z)),
        ("Leaky ReLU", leaky_relu(z), leaky_relu_d(z)),
        ("ELU",     elu(z),     elu_d(z)),
        ("GELU",    gelu(z),    None),
        ("Swish",   swish(z),   None),
        ("SELU",    selu(z),    None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for ax, (name, f, fd) in zip(axes, funcs):
        ax.plot(z, f, lw=2, label="f(z)")
        if fd is not None:
            ax.plot(z, fd, lw=2, linestyle="--", label="f'(z)")
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.set_title(name); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4); ax.set_ylim(-2.5, 2.5)
    plt.suptitle("Activation Functions and Their Derivatives", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "activation_functions.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Activation functions plot saved: {path}")


# ── 3. Vanishing gradient ────────────────────────────────────────────────────
def vanishing_gradient():
    print("\n=== Vanishing Gradient Problem ===")
    print("  Deep sigmoid networks suffer from vanishing gradients")
    print("  σ'(z) ≤ 0.25  →  after L layers, gradient ≤ 0.25^L")

    for L in [1, 5, 10, 20, 50]:
        max_grad = 0.25**L
        print(f"  L={L:<4} layers: max gradient ≤ {max_grad:.2e}")

    print("\n  ReLU solution: gradient = 1 for positive z → no vanishing")
    print("  But: dead neurons when z ≤ 0 for all inputs (dying ReLU)")


# ── 4. Dying ReLU ────────────────────────────────────────────────────────────
def dying_relu():
    print("\n=== Dying ReLU Problem ===")
    print("  If neuron always gets z < 0, gradient = 0 → weights never update")
    print("  Causes: large learning rate, poor weight init, negative bias")
    rng = np.random.default_rng(0)

    # Simulate a neuron that "dies"
    n_neurons = 1000
    z_neg = rng.normal(-2, 0.5, n_neurons)   # all negative pre-activations
    z_mix = rng.normal(0, 1, n_neurons)

    dead_neg = (relu_d(z_neg) == 0).mean()
    dead_mix = (relu_d(z_mix) == 0).mean()
    print(f"\n  Neurons dead (z<0): {dead_neg*100:.1f}% when mean(z)=-2")
    print(f"  Neurons dead (z<0): {dead_mix*100:.1f}% when mean(z)=0")
    print(f"\n  Solutions:")
    print(f"    Leaky ReLU: f(z) = max(0.01z, z)  — never truly dead")
    print(f"    ELU:        smooth negative region  — self-normalising")
    print(f"    SELU:       self-normalising (λ=1.0507, α=1.6733)")
    print(f"    He init:    proper weight init prevents large negative z")


# ── 5. Softmax (multi-class output) ──────────────────────────────────────────
def softmax_activation():
    print("\n=== Softmax Activation (Multi-class Output) ===")
    print("  σ(z)_k = exp(z_k) / Σ exp(z_j)   (numerically stable: subtract max)")

    def softmax(z):
        e = np.exp(z - z.max())   # numerical stability
        return e / e.sum()

    z_examples = [
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
    ]
    for z in z_examples:
        s = softmax(z)
        print(f"  z={z} → softmax={s.round(4)} (sum={s.sum():.4f})")

    print(f"\n  Properties:")
    print(f"    Output sums to 1 (valid probability distribution)")
    print(f"    Amplifies differences between logits")
    print(f"    Temperature: σ(z/T) where T→0 = argmax, T→∞ = uniform")


# ── 6. Activation choice guide ───────────────────────────────────────────────
def activation_guide():
    print("\n=== Activation Function Selection Guide ===")
    print(f"  {'Layer/Task':<30} {'Recommended':<20} {'Why'}")
    rows = [
        ("Hidden layers (default)",     "ReLU",         "Fast, works well, He init"),
        ("Hidden layers (deep nets)",   "GELU/Swish",   "Smooth, better gradient flow"),
        ("Residual networks",           "ReLU",         "Gradient highways via skip"),
        ("Self-normalizing nets",       "SELU",         "Keeps activations ~N(0,1)"),
        ("Binary classification output","Sigmoid",      "Maps to [0,1] probability"),
        ("Multi-class output",          "Softmax",      "Probability distribution"),
        ("Regression output",           "Linear/None",  "Unbounded real-valued output"),
        ("Generative models (hidden)",  "ELU/Swish",    "Smooth gradients everywhere"),
    ]
    for row in rows:
        print(f"  {row[0]:<30} {row[1]:<20} {row[2]}")


if __name__ == "__main__":
    activation_properties()
    plot_activations()
    vanishing_gradient()
    dying_relu()
    softmax_activation()
    activation_guide()
