"""
Working Example: NumPy
Covers array creation, dtypes, indexing, broadcasting,
universal functions (ufuncs), linear algebra, and random.
"""
import numpy as np


def array_creation():
    print("=== Array Creation ===")
    # From Python sequences
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"  1D array   : {a}  dtype={a.dtype}")
    print(f"  2D float32 :\n{b}")

    # Constructors
    print(f"  zeros(3,4):\n{np.zeros((3, 4))}")
    print(f"  ones(2,3) :\n{np.ones((2, 3), dtype=int)}")
    print(f"  eye(3)    :\n{np.eye(3)}")
    print(f"  arange    : {np.arange(0, 20, 3)}")
    print(f"  linspace  : {np.linspace(0, 1, 6)}")
    print(f"  full(2,3,7): {np.full((2, 3), 7)}")


def array_attributes():
    print("\n=== Array Attributes ===")
    a = np.arange(24).reshape(2, 3, 4)
    print(f"  shape  : {a.shape}")
    print(f"  ndim   : {a.ndim}")
    print(f"  size   : {a.size}")
    print(f"  dtype  : {a.dtype}")
    print(f"  itemsize: {a.itemsize} bytes")
    print(f"  nbytes  : {a.nbytes} bytes")


def indexing_and_slicing():
    print("\n=== Indexing & Slicing ===")
    a = np.arange(1, 13).reshape(3, 4)
    print(f"  array:\n{a}")
    print(f"  a[1, 2]     = {a[1, 2]}")
    print(f"  a[0]        = {a[0]}")
    print(f"  a[:, 2]     = {a[:, 2]}")
    print(f"  a[1:, 1:3]  =\n{a[1:, 1:3]}")
    print(f"  a[::2, ::2] =\n{a[::2, ::2]}")

    # Boolean indexing
    mask = a > 6
    print(f"  a > 6 mask:\n{mask}")
    print(f"  a[a > 6]    = {a[mask]}")

    # Fancy indexing
    rows = np.array([0, 2])
    print(f"  a[[0,2], :] =\n{a[rows, :]}")


def broadcasting():
    print("\n=== Broadcasting ===")
    a = np.array([[1], [2], [3]])          # shape (3, 1)
    b = np.array([10, 20, 30])             # shape (3,) → (1, 3)
    result = a + b
    print(f"  a (3×1):\n{a}")
    print(f"  b (3,) : {b}")
    print(f"  a + b  :\n{result}")

    # Normalise each row
    mat   = np.random.randint(1, 10, (3, 4)).astype(float)
    norms = mat / mat.sum(axis=1, keepdims=True)
    print(f"\n  row-normalised:\n{np.round(norms, 3)}")
    print(f"  row sums: {norms.sum(axis=1)}")


def ufuncs_and_aggregation():
    print("\n=== ufuncs & Aggregation ===")
    a = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    print(f"  sqrt   : {np.sqrt(a)}")
    print(f"  exp    : {np.exp([0, 1, 2])}")
    print(f"  log    : {np.log([1, np.e, np.e**2])}")

    data = np.random.normal(loc=0, scale=1, size=(100,))
    print(f"\n  100 N(0,1) samples:")
    print(f"    mean  = {data.mean():.4f}")
    print(f"    std   = {data.std():.4f}")
    print(f"    min   = {data.min():.4f}")
    print(f"    max   = {data.max():.4f}")
    print(f"    median= {np.median(data):.4f}")

    # axis-wise
    mat = np.arange(1, 13).reshape(3, 4)
    print(f"\n  matrix:\n{mat}")
    print(f"  sum(axis=0) = {mat.sum(axis=0)}")
    print(f"  sum(axis=1) = {mat.sum(axis=1)}")
    print(f"  cumsum      = {mat.cumsum()}")


def linear_algebra():
    print("\n=== Linear Algebra ===")
    A = np.array([[2, 1], [5, 3]])
    B = np.array([[1, 2], [3, 4]])

    print(f"  A:\n{A}")
    print(f"  B:\n{B}")
    print(f"  A @ B (matmul):\n{A @ B}")
    print(f"  A.T (transpose):\n{A.T}")
    print(f"  det(A) = {np.linalg.det(A):.2f}")
    print(f"  inv(A):\n{np.linalg.inv(A)}")

    # Solve Ax = b
    b_vec = np.array([4, 13])
    x = np.linalg.solve(A, b_vec)
    print(f"  solve Ax=b: x={x}  check Ax={A @ x}")

    # Eigenvalues
    vals, vecs = np.linalg.eig(A)
    print(f"  eigenvalues : {vals}")
    print(f"  eigenvectors:\n{np.round(vecs, 4)}")


def numpy_random():
    print("\n=== numpy.random ===")
    rng = np.random.default_rng(seed=42)   # recommended modern API

    uniform   = rng.uniform(0, 10, 5)
    normal    = rng.normal(0, 1, 5)
    integers  = rng.integers(1, 101, 5)
    shuffled  = np.arange(1, 11)
    rng.shuffle(shuffled)

    print(f"  uniform(0,10)   : {np.round(uniform, 2)}")
    print(f"  normal(0,1)     : {np.round(normal, 4)}")
    print(f"  integers(1,100) : {integers}")
    print(f"  shuffled 1-10   : {shuffled}")
    print(f"  choice 3        : {rng.choice(['a','b','c','d'], 3)}")


def performance_comparison():
    print("\n=== NumPy vs Pure Python Performance ===")
    import time
    n = 1_000_000
    py_list = list(range(n))
    np_arr  = np.arange(n)

    start = time.perf_counter()
    py_sum = sum(x**2 for x in py_list)
    t_py  = time.perf_counter() - start

    start = time.perf_counter()
    np_sum = (np_arr**2).sum()
    t_np  = time.perf_counter() - start

    print(f"  Python sum of squares : {py_sum} in {t_py:.4f}s")
    print(f"  NumPy  sum of squares : {np_sum} in {t_np:.4f}s")
    print(f"  Speedup               : {t_py/t_np:.1f}×")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    array_creation()
    array_attributes()
    indexing_and_slicing()
    broadcasting()
    ufuncs_and_aggregation()
    linear_algebra()
    numpy_random()
    performance_comparison()
