"""
Working Example: The Neuron and Perceptron
Covers biological inspiration, McCulloch-Pitts neuron, threshold logic unit,
Rosenblatt perceptron, multi-input neuron computation, and linear separability.
"""
import numpy as np


# -- 1. Biological inspiration -------------------------------------------------
def biological_inspiration():
    print("=== Biological Inspiration ===")
    print("  Biological neuron:")
    print("    Dendrites    -> receive signals (inputs)")
    print("    Soma (body)  -> integrate signals")
    print("    Axon         -> transmit output to other neurons")
    print("    Synapse      -> connection strength (= weight)")
    print()
    print("  Mathematical abstraction:")
    print("    z = w1x1 + w2x2 + ... + wnxn + b  (weighted sum + bias)")
    print("    a = f(z)                              (activation function)")


# -- 2. McCulloch-Pitts neuron -------------------------------------------------
def mcculloch_pitts():
    print("\n=== McCulloch-Pitts Neuron (1943) ===")
    print("  Binary inputs, binary output, threshold activation")
    print("  y = 1 if Sigmawᵢxᵢ >= theta else 0")

    def mp_neuron(x, w, theta):
        return int(np.dot(x, w) >= theta)

    # AND gate
    print("\n  AND gate (w=[1,1], theta=2):")
    for x in [(0,0),(0,1),(1,0),(1,1)]:
        y = mp_neuron(x, [1,1], 2)
        print(f"    x={x}: y={y}")

    # OR gate
    print("\n  OR gate (w=[1,1], theta=1):")
    for x in [(0,0),(0,1),(1,0),(1,1)]:
        y = mp_neuron(x, [1,1], 1)
        print(f"    x={x}: y={y}")

    # NOT gate
    print("\n  NOT gate (w=[-1], theta=0):")
    for x in [0, 1]:
        y = mp_neuron([x], [-1], 0)
        print(f"    x={x}: y={y}")

    print("\n  XOR is NOT linearly separable -> single MP neuron fails!")


# -- 3. Rosenblatt Perceptron --------------------------------------------------
def rosenblatt_perceptron():
    print("\n=== Rosenblatt Perceptron (1958) ===")
    print("  First learnable neural model")
    print("  Update rule: w <- w + eta(y - ŷ)x   b <- b + eta(y - ŷ)")

    class Perceptron:
        def __init__(self, lr=0.1, max_iter=100):
            self.lr, self.max_iter = lr, max_iter

        def fit(self, X, y):
            self.w = np.zeros(X.shape[1])
            self.b = 0.0
            self.errors_ = []
            for _ in range(self.max_iter):
                errors = 0
                for xi, yi in zip(X, y):
                    pred = self.predict_one(xi)
                    update = self.lr * (yi - pred)
                    self.w += update * xi
                    self.b += update
                    errors += int(update != 0)
                self.errors_.append(errors)
                if errors == 0:
                    break
            return self

        def predict_one(self, x):
            return 1 if np.dot(self.w, x) + self.b >= 0 else 0

        def predict(self, X):
            return np.array([self.predict_one(x) for x in X])

    # AND
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    p = Perceptron(lr=0.1, max_iter=50)
    p.fit(X, y)
    preds = p.predict(X)
    print(f"\n  AND gate convergence in {len(p.errors_)} epochs")
    print(f"  Final weights: {p.w}  bias: {p.b:.2f}")
    print(f"  Predictions: {preds}  (target: {y})")
    acc = (preds == y).mean()
    print(f"  Accuracy: {acc:.4f}")

    # XOR (should fail)
    y_xor = np.array([0, 1, 1, 0])
    p2 = Perceptron(lr=0.1, max_iter=100)
    p2.fit(X, y_xor)
    preds2 = p2.predict(X)
    print(f"\n  XOR gate ({len(p2.errors_)} epochs, never 0 errors): preds={preds2}  target={y_xor}")
    print(f"  Not linearly separable -> perceptron cannot converge")


# -- 4. Multi-input neuron computation ----------------------------------------
def neuron_computation():
    print("\n=== Single Neuron Forward Pass ===")
    print("  z = Sigma wᵢxᵢ + b = w·x + b")
    print("  a = sigma(z) = 1/(1+e^{-z})  (sigmoid)")

    rng = np.random.default_rng(0)
    n   = 5   # 5 inputs
    x   = rng.standard_normal(n)
    w   = rng.standard_normal(n)
    b   = 0.1

    z    = np.dot(w, x) + b
    a    = 1 / (1 + np.exp(-z))       # sigmoid
    a_r  = np.maximum(0, z)           # ReLU
    a_t  = np.tanh(z)

    print(f"\n  Inputs x:  {x.round(4)}")
    print(f"  Weights w: {w.round(4)}")
    print(f"  Bias b:    {b}")
    print(f"  z = w·x+b = {z:.4f}")
    print(f"  sigma(z)  = {a:.4f}")
    print(f"  ReLU(z)= {a_r:.4f}")
    print(f"  tanh(z)= {a_t:.4f}")


# -- 5. Linear separability ---------------------------------------------------
def linear_separability():
    print("\n=== Linear Separability ===")
    print("  A dataset is linearly separable if a hyperplane can divide classes")
    print("  Single perceptron/neuron can only learn linearly separable problems")
    print()
    problems = [
        ("AND",  [(0,0,0),(0,1,0),(1,0,0),(1,1,1)], True),
        ("OR",   [(0,0,0),(0,1,1),(1,0,1),(1,1,1)], True),
        ("NAND", [(0,0,1),(0,1,1),(1,0,1),(1,1,0)], True),
        ("XOR",  [(0,0,0),(0,1,1),(1,0,1),(1,1,0)], False),
        ("XNOR", [(0,0,1),(0,1,0),(1,0,0),(1,1,1)], False),
    ]
    for name, truth_table, sep in problems:
        print(f"  {name:<6}: {'Linearly separable' if sep else 'NOT linearly separable'}")


# -- 6. From perceptron to MLP -------------------------------------------------
def to_mlp():
    print("\n=== From Perceptron to Multi-Layer Perceptron ===")
    print("  Solution to XOR and non-linear problems: add hidden layer(s)")
    print()
    print("  MLP structure:")
    print("    Input layer  ->  Hidden layer(s)  ->  Output layer")
    print("    Each hidden neuron: aₕ = f(Wₕ·x + bₕ)")
    print("    Output neuron:      ŷ  = f(Wₒ·aₕ + bₒ)")
    print()
    print("  Key insight: composition of non-linear functions = universal approximator")
    print("  (Universal Approximation Theorem, Cybenko 1989)")

    # Solve XOR with 2-layer MLP (hand-coded weights)
    def sigmoid(z): return 1 / (1 + np.exp(-z))

    # Hidden layer weights (learned values that solve XOR)
    W1 = np.array([[20, 20], [-20, -20]])
    b1 = np.array([-10, 30])
    W2 = np.array([[20, 20]])
    b2 = np.array([-30])

    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    print(f"\n  XOR solved with 2-layer MLP (sigmoid):")
    for xi, yi in zip(X, y):
        h1   = sigmoid(W1 @ xi + b1)
        out  = sigmoid(W2 @ h1 + b2)[0]
        pred = int(out >= 0.5)
        print(f"    x={xi.astype(int)}: h={h1.round(3)}  out={out:.4f}  pred={pred}  target={yi}")


if __name__ == "__main__":
    biological_inspiration()
    mcculloch_pitts()
    rosenblatt_perceptron()
    neuron_computation()
    linear_separability()
    to_mlp()
