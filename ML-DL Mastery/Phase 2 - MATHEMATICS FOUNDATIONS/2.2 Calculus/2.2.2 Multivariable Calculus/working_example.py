"""
Working Example: Multivariable Calculus
Covers partial derivatives, gradients, Jacobians, Hessians,
directional derivatives, critical points, and double integrals.
"""
import numpy as np
from scipy import optimize, integrate


# -- 1. Partial derivatives ----------------------------------------------------
def partial_derivatives():
    print("=== Partial Derivatives ===")
    # f(x,y) = x²y + sin(xy)
    f    = lambda x, y: x**2 * y + np.sin(x*y)
    df_x = lambda x, y: 2*x*y + y*np.cos(x*y)   # df/dx
    df_y = lambda x, y: x**2   + x*np.cos(x*y)   # df/dy

    x0, y0 = 1.0, 2.0
    h = 1e-6
    print(f"  f(x,y) = x²y + sin(xy) at ({x0},{y0})")
    print(f"  df/dx  exact   = {df_x(x0,y0):.6f}")
    print(f"  df/dx  numeric = {(f(x0+h,y0) - f(x0-h,y0))/(2*h):.6f}")
    print(f"  df/dy  exact   = {df_y(x0,y0):.6f}")
    print(f"  df/dy  numeric = {(f(x0,y0+h) - f(x0,y0-h))/(2*h):.6f}")


# -- 2. Gradient ---------------------------------------------------------------
def gradients():
    print("\n=== Gradient ∇f (direction of steepest ascent) ===")
    # f(x,y,z) = x² + 2y² + 3z²
    f = lambda v: v[0]**2 + 2*v[1]**2 + 3*v[2]**2
    grad_f = lambda v: np.array([2*v[0], 4*v[1], 6*v[2]])

    v0 = np.array([1., 2., 3.])
    h  = 1e-6
    g_exact   = grad_f(v0)
    g_numeric = np.array([(f(v0 + h*e) - f(v0 - h*e)) / (2*h)
                           for e in np.eye(3)])
    print(f"  f(v) = x² + 2y² + 3z²  at v={v0}")
    print(f"  ∇f exact  : {g_exact}")
    print(f"  ∇f numeric: {np.round(g_numeric, 8)}")
    print(f"  gradient points toward: {g_exact / np.linalg.norm(g_exact)}")
    print(f"  ||∇f|| = {np.linalg.norm(g_exact):.4f}")


# -- 3. Directional derivative -------------------------------------------------
def directional_derivative():
    print("\n=== Directional Derivative D_u f = ∇f · û ===")
    f      = lambda v: v[0]**2 + v[1]**2
    grad_f = lambda v: np.array([2*v[0], 2*v[1]])

    v0 = np.array([1., 1.])
    g  = grad_f(v0)

    directions = {
        "gradient dir (max)": g / np.linalg.norm(g),
        "neg gradient (min)": -g / np.linalg.norm(g),
        "orthogonal":         np.array([-g[1], g[0]]) / np.linalg.norm(g),
    }
    print(f"  f(x,y) = x² + y²  at {v0}   ∇f = {g}")
    for name, u in directions.items():
        Duf = np.dot(g, u)
        print(f"  D_{name}: {Duf:.4f}")


# -- 4. Jacobian ---------------------------------------------------------------
def jacobian():
    print("\n=== Jacobian Matrix J_f (mxn for f: Rⁿ->Rᵐ) ===")
    # f: R² -> R², f(x,y) = [x²+y, x·sin(y)]
    def f(v):
        x, y = v
        return np.array([x**2 + y, x * np.sin(y)])

    def J_exact(v):
        x, y = v
        return np.array([[2*x,        1      ],
                         [np.sin(y),  x*np.cos(y)]])

    v0 = np.array([1., np.pi/4])
    h  = 1e-6
    J_e = J_exact(v0)
    # Finite difference Jacobian
    J_n = np.zeros((2, 2))
    f0  = f(v0)
    for j in range(2):
        e = np.zeros(2); e[j] = h
        J_n[:, j] = (f(v0 + e) - f(v0 - e)) / (2*h)

    print(f"  f(x,y) = [x²+y, x·sin(y)]   at v={np.round(v0,4)}")
    print(f"  J exact:\n{np.round(J_e, 6)}")
    print(f"  J numeric:\n{np.round(J_n, 6)}")
    print(f"  max diff: {np.max(np.abs(J_e - J_n)):.2e}")


# -- 5. Hessian ----------------------------------------------------------------
def hessian():
    print("\n=== Hessian Matrix H_f (second-order partial derivatives) ===")
    # f(x,y) = x³ + x·y² - 3y
    f = lambda v: v[0]**3 + v[0]*v[1]**2 - 3*v[1]

    def H_exact(v):
        x, y = v
        return np.array([[6*x,   2*y],
                         [2*y,   2*x]])

    v0 = np.array([1., 2.])
    H_e = H_exact(v0)

    # Finite difference Hessian
    h = 1e-4
    n = len(v0)
    H_n = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H_n[i,j] = (f(v0+ei+ej) - f(v0+ei-ej) - f(v0-ei+ej) + f(v0-ei-ej)) / (4*h**2)

    print(f"  f(x,y) = x³ + xy² - 3y  at {v0}")
    print(f"  H exact:\n{H_e}")
    print(f"  H numeric:\n{np.round(H_n, 4)}")

    eigvals = np.linalg.eigvalsh(H_e)
    print(f"  eigenvalues: {np.round(eigvals, 4)}")
    if np.all(eigvals > 0):  print("  -> positive definite -> local minimum")
    elif np.all(eigvals < 0): print("  -> negative definite -> local maximum")
    else:                     print("  -> indefinite -> saddle point")


# -- 6. Critical points classification -----------------------------------------
def critical_points():
    print("\n=== Critical Points via Gradient=0 ===")
    # f(x,y) = x⁴ - 4x² + y² - 2y
    # df/dx = 4x³ - 8x = 4x(x²-2) = 0  -> x=0,+/-sqrt2
    # df/dy = 2y - 2 = 0               -> y=1
    f = lambda v: v[0]**4 - 4*v[0]**2 + v[1]**2 - 2*v[1]

    critical = [np.array([0., 1.]),
                np.array([np.sqrt(2), 1.]),
                np.array([-np.sqrt(2), 1.])]

    def H(v):
        x = v[0]
        return np.array([[12*x**2 - 8, 0],
                         [0,            2]])

    print(f"  f(x,y) = x⁴ - 4x² + y² - 2y")
    for cp in critical:
        Hi     = H(cp)
        eigs   = np.linalg.eigvalsh(Hi)
        nature = ("minimum" if np.all(eigs>0) else
                  "maximum" if np.all(eigs<0) else "saddle")
        print(f"  ({np.round(cp,4)}) -> f={f(cp):.4f}  {nature}  (H eigs={np.round(eigs,4)})")


# -- 7. Double integral (numeric) ----------------------------------------------
def double_integral():
    print("\n=== Double Integral integralintegral f(x,y) dA ===")
    # integral01 integral01 (x²y + y²x) dx dy = 1/3 · 1/2 + 1/2 · 1/3 = 1/3
    f = lambda y, x: x**2 * y + y**2 * x
    result, err = integrate.dblquad(f, 0, 1, 0, 1)
    exact = 1.0 / 3.0
    print(f"  integral01integral01 (x²y + xy²) dx dy")
    print(f"  exact   = {exact:.8f}")
    print(f"  numeric = {result:.8f}  (err est={err:.2e})")

    # Area of an ellipse: integralintegral_{x²/a²+y²/b²<=1} dA = piab
    a, b = 3, 2
    f_ellipse = lambda y, x: 1.0
    area, _ = integrate.dblquad(f_ellipse,
                                 -a, a,
                                 lambda x: -b * np.sqrt(np.maximum(0, 1 - (x/a)**2)),
                                 lambda x:  b * np.sqrt(np.maximum(0, 1 - (x/a)**2)))
    print(f"\n  Area of ellipse a={a},b={b}: numeric={area:.4f}  piab={np.pi*a*b:.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    partial_derivatives()
    gradients()
    directional_derivative()
    jacobian()
    hessian()
    critical_points()
    double_integral()
