"""
Working Example: Matrix Calculus
Covers derivatives of scalars/vectors/matrices w.r.t. vectors and matrices,
chain rule in matrix form, and the connection to neural-network backpropagation.
"""
import numpy as np


# ── Notation reference ────────────────────────────────────────────────────────
def notation():
    print("=== Matrix Calculus Notation ===")
    ref = [
        ("∂y/∂x",   "y scalar, x scalar",  "scalar"),
        ("∂y/∂x",   "y scalar, x (n,) vec","row vector (1×n) — 'numerator layout'"),
        ("∂y/∂x",   "y scalar, x (n,) vec","column vector (n×1) — 'denominator layout'"),
        ("∂y/∂X",   "y scalar, X (m×n)",   "(m×n) matrix — gradient matrix"),
        ("∂y/∂X",   "y (p,) vec, x (n,)",  "(n×p) Jacobian"),
    ]
    for lhs, dims, result in ref:
        print(f"  {lhs:<10} | {dims:<30} | result = {result}")
    print("\n  (We follow denominator/column layout: ∂y/∂x is a column vector)")


# ── 1. Scalar-by-vector: ∂(aᵀx)/∂x = a ──────────────────────────────────────
def scalar_by_vector():
    print("\n=== ∂(aᵀx)/∂x = a   and   ∂(xᵀAx)/∂x = (A+Aᵀ)x ===")
    n = 4
    rng = np.random.default_rng(0)
    a   = rng.standard_normal(n)
    A   = rng.standard_normal((n, n))
    x0  = rng.standard_normal(n)

    # f = aᵀx → ∂f/∂x = a
    f1       = lambda x: a @ x
    grad1_ex = a
    grad1_nu = numerical_grad(f1, x0)
    print(f"  f = aᵀx:  exact={np.round(grad1_ex,4)}  numeric={np.round(grad1_nu,4)}")
    print(f"  match: {np.allclose(grad1_ex, grad1_nu)}")

    # f = xᵀAx → ∂f/∂x = (A+Aᵀ)x
    f2       = lambda x: x @ A @ x
    grad2_ex = (A + A.T) @ x0
    grad2_nu = numerical_grad(f2, x0)
    print(f"\n  f = xᵀAx:  max|exact-numeric| = {np.max(np.abs(grad2_ex-grad2_nu)):.2e}")

    # If A symmetric: ∂(xᵀAx)/∂x = 2Ax
    A_sym    = A + A.T
    grad_sym = 2 * A_sym @ x0
    f_sym    = lambda x: x @ A_sym @ x
    print(f"  f = xᵀ(A+Aᵀ)x symmetric: grad=2(A+Aᵀ)x,  match={np.allclose(grad_sym, numerical_grad(f_sym, x0))}")


# ── 2. ∂(Ax)/∂x = A and ∂(xᵀA)/∂x = Aᵀ ─────────────────────────────────────
def vector_by_vector():
    print("\n=== Jacobians: ∂(Ax)/∂x = A ===")
    m, n = 3, 4
    rng  = np.random.default_rng(1)
    A    = rng.standard_normal((m, n))
    x0   = rng.standard_normal(n)

    # J[i,j] = ∂(Ax)_i / ∂x_j = A[i,j]
    J_exact   = A
    J_numeric = numerical_jacobian(lambda x: A @ x, x0)
    print(f"  J = A ({m}×{n}):  max|J_exact - J_numeric| = {np.max(np.abs(J_exact - J_numeric)):.2e}")


# ── 3. Derivative of loss w.r.t. weight matrix (linear layer) ────────────────
def linear_layer_gradient():
    print("\n=== Gradient Through Linear Layer y = Wx + b ===")
    n_in, n_out = 4, 3
    rng  = np.random.default_rng(2)
    W    = rng.standard_normal((n_out, n_in))
    b    = rng.standard_normal(n_out)
    x    = rng.standard_normal(n_in)
    y_t  = rng.standard_normal(n_out)   # target

    # Forward: y = Wx + b,  L = ||y - y_t||²
    y    = W @ x + b
    loss = 0.5 * np.sum((y - y_t)**2)
    dL_dy = y - y_t                      # ∂L/∂y = (y - y_t)

    # Backprop:
    #   ∂L/∂W = ∂L/∂y · ∂y/∂W = dL_dy ⊗ x (outer product)
    #   ∂L/∂b = ∂L/∂y · ∂y/∂b = dL_dy (identity Jacobian)
    #   ∂L/∂x = Wᵀ dL_dy
    dL_dW = np.outer(dL_dy, x)
    dL_db = dL_dy
    dL_dx = W.T @ dL_dy

    # Verify numerically
    h = 1e-5
    dL_dW_num = np.zeros_like(W)
    for i in range(n_out):
        for j in range(n_in):
            W1 = W.copy(); W1[i,j] += h
            W2 = W.copy(); W2[i,j] -= h
            y1 = W1 @ x + b;  L1 = 0.5*np.sum((y1-y_t)**2)
            y2 = W2 @ x + b;  L2 = 0.5*np.sum((y2-y_t)**2)
            dL_dW_num[i,j] = (L1 - L2) / (2*h)

    print(f"  Loss = {loss:.4f}")
    print(f"  max|∂L/∂W  analytic - numeric| = {np.max(np.abs(dL_dW - dL_dW_num)):.2e}")
    print(f"  ∂L/∂b = {np.round(dL_db, 4)}")
    print(f"  ∂L/∂x shape: {dL_dx.shape}  (flows to previous layer)")


# ── 4. Chain rule in matrix form (composition of layers) ─────────────────────
def chain_rule_matrix():
    print("\n=== Chain Rule: L = loss(g(f(x))) ===")
    # f: R³→R², g: R²→R², loss: R²→R
    # f(x) = W1 x  (linear)
    # g(z) = tanh(W2 z)  (activation)
    # L(a) = 0.5 ||a||²
    rng = np.random.default_rng(3)
    x   = rng.standard_normal(3)
    W1  = rng.standard_normal((2, 3)) * 0.5
    W2  = rng.standard_normal((2, 2)) * 0.5

    # Forward
    z    = W1 @ x
    a    = np.tanh(W2 @ z)
    loss = 0.5 * np.sum(a**2)

    # Backward
    dL_da  = a
    dtanh  = 1 - np.tanh(W2 @ z)**2       # elementwise
    dL_dWz = dL_da * dtanh                 # dL/d(W2 z) via elementwise chain
    dL_dW2 = np.outer(dL_dWz, z)          # ∂L/∂W2
    dL_dz  = W2.T @ dL_dWz               # propagate through W2
    dL_dW1 = np.outer(dL_dz, x)           # ∂L/∂W1
    dL_dx  = W1.T @ dL_dz                # gradient w.r.t. input

    # Numeric verification
    def forward(x_):
        z_ = W1 @ x_
        a_ = np.tanh(W2 @ z_)
        return 0.5 * np.sum(a_**2)

    dL_dx_num = numerical_grad(forward, x)
    print(f"  Loss = {loss:.6f}")
    print(f"  ∂L/∂x  analytic: {np.round(dL_dx, 6)}")
    print(f"  ∂L/∂x  numeric : {np.round(dL_dx_num, 6)}")
    print(f"  max diff: {np.max(np.abs(dL_dx - dL_dx_num)):.2e}")


# ── 5. Common matrix derivative identities ────────────────────────────────────
def identities():
    print("\n=== Common Matrix Derivative Identities ===")
    idents = [
        ("∂(aᵀx)/∂x",           "a"),
        ("∂(xᵀa)/∂x",           "a"),
        ("∂(xᵀx)/∂x",           "2x"),
        ("∂(xᵀAx)/∂x",          "(A+Aᵀ)x  →  2Ax if A symmetric"),
        ("∂(aᵀXb)/∂X",          "abᵀ"),
        ("∂tr(AX)/∂X",          "Aᵀ"),
        ("∂tr(XᵀA)/∂X",         "A"),
        ("∂tr(AXBXᵀ)/∂X",       "AᵀXBᵀ + AXB"),
        ("∂(Ax)/∂x",             "A  (Jacobian)"),
        ("∂ln det(X)/∂X",        "X⁻ᵀ"),
    ]
    for lhs, rhs in idents:
        print(f"  {lhs:<30} = {rhs}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def numerical_grad(f, x0, h=1e-6):
    grad = np.zeros_like(x0)
    for i in range(len(x0)):
        e = np.zeros_like(x0); e[i] = h
        grad[i] = (f(x0+e) - f(x0-e)) / (2*h)
    return grad


def numerical_jacobian(f, x0, h=1e-6):
    f0 = f(x0)
    J  = np.zeros((len(f0), len(x0)))
    for j in range(len(x0)):
        e = np.zeros_like(x0); e[j] = h
        J[:, j] = (f(x0+e) - f(x0-e)) / (2*h)
    return J


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    notation()
    scalar_by_vector()
    vector_by_vector()
    linear_layer_gradient()
    chain_rule_matrix()
    identities()
