"""
Working Example 2: State Space Models (SSMs)
Linear SSM with A, B, C, D matrices on a 1D input sequence.
Compares SSM output with a simple RNN proxy.
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


def ssm_step(x, h_prev, A, B, C, D):
    """Single SSM step: h_t = A h_{t-1} + B x_t, y_t = C h_t + D x_t."""
    h = A @ h_prev + B * x
    y = C @ h + D * x
    return h, y


def run_ssm(u, A, B, C, D):
    """Run SSM over input sequence u (T,)."""
    N = A.shape[0]
    h = np.zeros(N)
    ys = []
    for x in u:
        h, y = ssm_step(x, h, A, B, C, D)
        ys.append(float(y))
    return np.array(ys)


def discretise_ssm(A_c, B_c, delta=0.1):
    """Bilinear (Tustin) discretisation: A_d = (I + Δ/2 A)(I - Δ/2 A)^{-1}."""
    N = A_c.shape[0]
    I = np.eye(N)
    A_d = np.linalg.solve(I - (delta / 2) * A_c, I + (delta / 2) * A_c)
    B_d = np.linalg.solve(I - (delta / 2) * A_c, delta * B_c)
    return A_d, B_d


def demo():
    print("=== State Space Models (SSMs) ===")
    T = 200
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, T)

    # Input: noisy sine wave
    u = np.sin(t) + 0.3 * rng.standard_normal(T)

    # Continuous SSM: diagonal state matrix (stable if eigenvalues < 0)
    N = 4  # state dimension
    # Stable diagonal A_c (negative eigenvalues)
    A_c = -np.diag(np.array([0.5, 1.0, 2.0, 4.0]))
    B_c = rng.standard_normal((N, 1))[:, 0]
    C = rng.standard_normal((1, N))[0]
    D = 0.1

    # Discretise
    A_d, B_d = discretise_ssm(A_c, B_c, delta=4 * np.pi / T)

    print(f"  State dim N={N}, T={T}")
    print(f"  A_d eigenvalues: {np.linalg.eigvals(A_d).real.round(4)}")

    y_ssm = run_ssm(u, A_d, B_d, C, D)

    # Compare with different delta values (time-step sensitivity)
    deltas = [0.05, 0.1, 0.2, 0.5]
    y_deltas = []
    for delta in deltas:
        Ad, Bd = discretise_ssm(A_c, B_c, delta)
        y_deltas.append(run_ssm(u, Ad, Bd, C, D))

    # Impulse response
    impulse = np.zeros(T)
    impulse[0] = 1.0
    y_impulse = run_ssm(impulse, A_d, B_d, C, D)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Input vs SSM output
    axes[0][0].plot(t, u, color="lightgray", lw=1, label="Input u(t)", alpha=0.8)
    axes[0][0].plot(t, y_ssm, color="steelblue", lw=2, label="SSM output y(t)")
    axes[0][0].set(xlabel="t", ylabel="Signal", title="SSM: Input vs Output")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # Δ sensitivity
    for delta, yd in zip(deltas, y_deltas):
        axes[0][1].plot(t, yd, lw=1.5, label=f"Δ={delta}")
    axes[0][1].set(xlabel="t", ylabel="y(t)", title="SSM Output vs Step Size Δ")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(True, alpha=0.3)

    # Impulse response
    axes[1][0].plot(t[:80], y_impulse[:80], color="tomato", lw=2)
    axes[1][0].set(xlabel="t", ylabel="y(t)", title="Impulse Response")
    axes[1][0].grid(True, alpha=0.3)

    # Eigenvalue plot
    eigs = np.linalg.eigvals(A_d)
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[1][1].plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, lw=1, label="Unit circle")
    axes[1][1].scatter(eigs.real, eigs.imag, s=80, zorder=5, color="steelblue", label="A_d eigenvalues")
    axes[1][1].set(xlabel="Re", ylabel="Im", title="Eigenvalues of A_d")
    axes[1][1].axhline(0, color="k", lw=0.5)
    axes[1][1].axvline(0, color="k", lw=0.5)
    axes[1][1].legend()
    axes[1][1].set_aspect("equal")
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "state_space.png", dpi=100)
    plt.close()
    print("  Saved state_space.png")


if __name__ == "__main__":
    demo()
