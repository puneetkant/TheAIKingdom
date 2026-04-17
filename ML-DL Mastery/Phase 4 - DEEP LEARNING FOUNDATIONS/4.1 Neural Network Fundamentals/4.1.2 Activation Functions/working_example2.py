"""
Working Example 2: Activation Functions — numpy implementations + gradients
============================================================================
Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, GELU, Softmax — values, gradients, dead neuron.

Run:  python working_example2.py
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

# ── Activations ──────────────────────────────────────────────────────────────
def sigmoid(x):   return 1 / (1 + np.exp(-x))
def sigmoid_d(x): s = sigmoid(x); return s * (1 - s)

def tanh_d(x):    return 1 - np.tanh(x)**2

def relu(x):      return np.maximum(0, x)
def relu_d(x):    return (x > 0).astype(float)

def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)

def elu(x, a=1.0): return np.where(x > 0, x, a * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def demo_values():
    print("=== Activation Values at z ∈ {-2, -1, 0, 1, 2} ===")
    z = np.array([-2., -1., 0., 1., 2.])
    print(f"  {'z':>6}  {'sigmoid':>10}  {'tanh':>10}  {'relu':>10}  {'gelu':>10}")
    for zi in z:
        print(f"  {zi:6.1f}  {sigmoid(zi):10.4f}  {np.tanh(zi):10.4f}  "
              f"{relu(zi):10.4f}  {gelu(zi):10.4f}")

def demo_gradients():
    print("\n=== Gradients at z ∈ {-2, 0, 2} ===")
    z = np.array([-2., 0., 2.])
    print(f"  {'z':>6}  {'σ\'':>10}  {'tanh\'':>10}  {'ReLU\'':>10}")
    for zi in z:
        print(f"  {zi:6.1f}  {sigmoid_d(zi):10.4f}  {tanh_d(zi):10.4f}  {relu_d(zi):10.4f}")

def demo_plot():
    x = np.linspace(-4, 4, 300)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    pairs = [("Sigmoid", sigmoid(x), "blue"), ("Tanh", np.tanh(x), "green"),
             ("ReLU", relu(x), "red"), ("Leaky ReLU", leaky_relu(x), "orange"),
             ("ELU", elu(x), "purple"), ("GELU", gelu(x), "brown")]
    for ax, (name, vals, col) in zip(axes.flat, pairs):
        ax.plot(x, vals, col, lw=2); ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
        ax.set_title(name); ax.set_ylim(-2, 3); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT / "activation_functions.png"); plt.close()
    print("\n  Saved activation_functions.png")

def demo_softmax():
    print("\n=== Softmax (3-class logits) ===")
    logits = np.array([2.0, 1.0, 0.5])
    probs  = softmax(logits)
    print(f"  Logits: {logits}  → Probs: {probs.round(4)}  (sum={probs.sum():.6f})")

if __name__ == "__main__":
    demo_values()
    demo_gradients()
    demo_plot()
    demo_softmax()
