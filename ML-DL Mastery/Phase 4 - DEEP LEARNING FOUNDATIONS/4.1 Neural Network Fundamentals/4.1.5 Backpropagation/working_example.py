"""
Working Example: Backpropagation
Covers chain rule derivation, manual gradient computation for a 2-layer MLP,
gradient checking, a complete training loop, and numerical gradient verification.
"""
import numpy as np


def sigmoid(z):  return 1 / (1 + np.exp(-z.clip(-500, 500)))
def relu(z):     return np.maximum(0, z)
def relu_d(z):   return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── 1. Chain rule primer ─────────────────────────────────────────────────────
def chain_rule():
    print("=== Chain Rule Primer ===")
    print("  If L = f(g(x)) then dL/dx = (dL/dg)(dg/dx)")
    print()
    print("  Neural net: L(ŷ(A2(Z2(A1(Z1(x,W1,b1),W2,b2))))")
    print()
    print("  ∂L/∂W1 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂A1 · ∂A1/∂Z1 · ∂Z1/∂W1")
    print()
    print("  Gradient flows BACKWARDS through the computation graph")
    print("  Each layer receives upstream gradient and passes downstream gradient")


# ── 2. Full 2-layer MLP with forward + backward ───────────────────────────────
class MLP2:
    """2-layer MLP: input → hidden(ReLU) → output(Sigmoid), BCE loss."""

    def __init__(self, d_in, d_h, d_out, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2 / d_in)
        scale2 = np.sqrt(2 / d_h)
        self.W1 = rng.standard_normal((d_in, d_h)) * scale1
        self.b1 = np.zeros(d_h)
        self.W2 = rng.standard_normal((d_h, d_out)) * scale2
        self.b2 = np.zeros(d_out)

    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def loss(self, y):
        """Binary cross-entropy."""
        eps = 1e-15
        return -np.mean(y * np.log(self.A2 + eps) + (1-y) * np.log(1-self.A2 + eps))

    def backward(self, y):
        n = len(y)
        # Output layer
        # dL/dZ2 = A2 - y  (combined sigmoid + BCE gradient)
        dZ2 = (self.A2 - y) / n                       # (n, d_out)
        dW2 = self.A1.T @ dZ2                         # (d_h, d_out)
        db2 = dZ2.sum(axis=0)                         # (d_out,)
        # Hidden layer
        dA1 = dZ2 @ self.W2.T                         # (n, d_h)
        dZ1 = dA1 * relu_d(self.Z1)                  # (n, d_h)  ← ReLU gradient
        dW1 = self.X.T @ dZ1                          # (d_in, d_h)
        db1 = dZ1.sum(axis=0)                         # (d_h,)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def apply_gradients(self, grads, lr):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]


# ── 3. Manual gradient derivation trace ──────────────────────────────────────
def gradient_trace():
    print("\n=== Backward Pass Gradient Derivation ===")
    print("  Loss: L = BCE = -[y log(σ(Z2)) + (1-y)log(1-σ(Z2))] / n")
    print()
    print("  OUTPUT LAYER:")
    print("  ∂L/∂Z2 = A2 - y      (elegant! sigmoid cancels BCE denominator)")
    print("  ∂L/∂W2 = A1ᵀ · δ2    (δ2 = ∂L/∂Z2)")
    print("  ∂L/∂b2 = Σ δ2")
    print()
    print("  HIDDEN LAYER:")
    print("  ∂L/∂A1 = δ2 · W2ᵀ    (propagate error back through W2)")
    print("  ∂L/∂Z1 = ∂L/∂A1 ⊙ relu'(Z1)   (element-wise, relu'=0 or 1)")
    print("  ∂L/∂W1 = Xᵀ · δ1    (δ1 = ∂L/∂Z1)")
    print("  ∂L/∂b1 = Σ δ1")


# ── 4. Gradient checking ─────────────────────────────────────────────────────
def gradient_check():
    print("\n=== Gradient Checking (Numerical Verification) ===")
    print("  Numerical gradient: ∂L/∂θ ≈ [L(θ+ε) - L(θ-ε)] / 2ε")
    print("  Compare with analytical gradient; rel error < 1e-5 → correct")

    rng = np.random.default_rng(5)
    X   = rng.standard_normal((8, 3))
    y   = rng.integers(0, 2, (8, 1)).astype(float)

    net   = MLP2(3, 4, 1, seed=0)
    net.forward(X)
    grads = net.backward(y)

    eps = 1e-5
    for param_name in ["W1", "b1", "W2", "b2"]:
        param = getattr(net, param_name)
        grad_analytical = grads[f"d{param_name}"].ravel()
        grad_numerical  = np.zeros_like(param.ravel())

        for j in range(len(grad_numerical)):
            param_flat = param.ravel()
            old = param_flat[j]

            param_flat[j] = old + eps
            param.ravel()[:] = param_flat
            net.forward(X); Lp = net.loss(y)

            param_flat[j] = old - eps
            param.ravel()[:] = param_flat
            net.forward(X); Lm = net.loss(y)

            grad_numerical[j] = (Lp - Lm) / (2 * eps)
            param_flat[j] = old
            param.ravel()[:] = param_flat

        diff = np.abs(grad_analytical - grad_numerical).max()
        print(f"  {param_name:<4}: max abs diff = {diff:.2e}  {'✓ OK' if diff < 1e-4 else '✗ ERROR'}")


# ── 5. Full training loop ─────────────────────────────────────────────────────
def training_loop():
    print("\n=== Full Training Loop ===")
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = y.reshape(-1, 1).astype(float)

    net    = MLP2(d_in=2, d_h=16, d_out=1, seed=1)
    lr     = 0.05
    epochs = 200

    for ep in range(epochs + 1):
        net.forward(X)
        L = net.loss(y)
        grads = net.backward(y)
        net.apply_gradients(grads, lr)
        if ep % 50 == 0:
            y_pred = (net.A2 >= 0.5).astype(int)
            acc = (y_pred == y.astype(int)).mean()
            print(f"  Epoch {ep:>4}: loss={L:.4f}  acc={acc:.4f}")


# ── 6. Backprop for softmax + CCE ────────────────────────────────────────────
def softmax_backprop():
    print("\n=== Softmax + Categorical Cross-Entropy Gradient ===")
    print("  ∂L/∂Z = ŷ - one_hot(y)    (same elegant form as sigmoid + BCE!)")
    print()
    rng  = np.random.default_rng(6)
    n, K = 4, 3
    Z    = rng.standard_normal((n, K))
    y    = np.array([0, 1, 2, 0])

    yhat  = softmax(Z)
    y_ohe = np.zeros((n, K)); y_ohe[np.arange(n), y] = 1
    dZ    = (yhat - y_ohe) / n

    eps   = 1e-15
    L     = -np.mean(np.sum(y_ohe * np.log(yhat + eps), axis=1))
    print(f"  Loss = {L:.4f}")
    print(f"  ∂L/∂Z (first sample):")
    print(f"    ŷ    = {yhat[0].round(4)}")
    print(f"    y    = {y_ohe[0]}")
    print(f"    ∂L/∂Z= {dZ[0].round(6)}")


if __name__ == "__main__":
    chain_rule()
    gradient_trace()
    gradient_check()
    training_loop()
    softmax_backprop()
