"""
Working Example: Combinatorics
Covers permutations, combinations, multinomials, the pigeonhole principle,
inclusion-exclusion, generating functions, and derangements.
"""
import math
from itertools import permutations, combinations, combinations_with_replacement, product
from functools import reduce
from collections import Counter


# -- 1. Factorials, permutations, combinations ---------------------------------
def basic_counting():
    print("=== Basic Counting ===")
    n, r = 6, 3

    nPr = math.perm(n, r)   # n! / (n-r)!
    nCr = math.comb(n, r)   # n! / (r!(n-r)!)

    print(f"  n={n}, r={r}")
    print(f"  P(n,r) = n!/(n-r)! = {nPr}")
    print(f"  C(n,r) = n!/(r!(n-r)!) = {nCr}")
    print(f"  Multinomial (2,2,2)! / (2!·2!·2!) = {math.factorial(6)//(2**3)}")

    # Stars and bars: number of ways to put k identical balls in n bins
    k, n_bins = 5, 3
    sb = math.comb(k + n_bins - 1, n_bins - 1)
    print(f"\n  Stars & bars: {k} identical balls into {n_bins} bins = C({k+n_bins-1},{n_bins-1}) = {sb}")


# -- 2. Explicit enumeration via itertools -------------------------------------
def itertools_demo():
    print("\n=== itertools Enumeration ===")
    items = ['A', 'B', 'C', 'D']

    perms = list(permutations(items, 2))
    combs = list(combinations(items, 2))
    combs_r = list(combinations_with_replacement(items, 2))
    cart = list(product([0,1], repeat=3))   # 2³ = 8

    print(f"  P(4,2) permutations: {len(perms)}  e.g. {perms[:4]}")
    print(f"  C(4,2) combinations : {len(combs)}  e.g. {combs}")
    print(f"  C(4+2-1,2) comb w/ rep: {len(combs_r)}  e.g. {combs_r}")
    print(f"  2³ Cartesian product: {len(cart)}  binary strings of length 3")


# -- 3. Pigeonhole principle ---------------------------------------------------
def pigeonhole():
    print("\n=== Pigeonhole Principle ===")
    print("  If n+1 items fit into n pigeonholes, at least one holds >=2 items.")
    print()
    # Birthday paradox probability
    def birthday_prob(k, n=365):
        """P(at least 2 people share a birthday among k people)"""
        prob_all_diff = 1.0
        for i in range(k):
            prob_all_diff *= (n - i) / n
        return 1 - prob_all_diff

    print(f"  {'People':<8} {'P(shared birthday)'}")
    for k in [10, 20, 23, 30, 50, 70]:
        p = birthday_prob(k)
        mark = " <- >50%" if p > 0.5 else ""
        print(f"  {k:<8} {p:.4f}{mark}")


# -- 4. Inclusion-exclusion principle -----------------------------------------
def inclusion_exclusion():
    print("\n=== Inclusion-Exclusion ===")
    # |AuBuC| = |A|+|B|+|C| - |AnB| - |AnC| - |BnC| + |AnBnC|
    A = set(range(1, 11))      # multiples of 1 in 1..30 (proxy)
    B = set(range(2, 31, 2))   # even numbers 1..30
    C = set(range(3, 31, 3))   # multiples of 3

    union_IE  = len(B) + len(C) - len(B & C)
    union_set = len(B | C)
    print(f"  B = even numbers in [1,30]:   {len(B)} elements")
    print(f"  C = mult of 3 in [1,30]:      {len(C)} elements")
    print(f"  |BnC| = mult of 6:            {len(B&C)} elements")
    print(f"  |BuC| via I-E: {union_IE}   via set: {union_set}  match={union_IE==union_set}")

    # Derangements via I-E: D_n = n! * Sigma_{k=0}^{n} (-1)^k / k!
    print("\n  Derangements D_n (permutations with no fixed point):")
    for n in range(1, 9):
        Dn_exact = round(math.factorial(n) * sum((-1)**k / math.factorial(k) for k in range(n+1)))
        Dn_approx = round(math.factorial(n) / math.e)
        print(f"    D_{n} = {Dn_exact}  (~= n!/e = {Dn_approx})")


# -- 5. Pascal's triangle and binomial theorem ---------------------------------
def pascals_triangle():
    print("\n=== Pascal's Triangle ===")
    rows = 8
    triangle = [[math.comb(n, k) for k in range(n+1)] for n in range(rows)]
    for row in triangle:
        print("  " + "  ".join(f"{x:3d}" for x in row))

    # Binomial theorem: (a+b)^n = Sigma C(n,k) a^k b^(n-k)
    a, b, n = 2, 3, 4
    exact   = (a + b)**n
    binom   = sum(math.comb(n,k) * a**k * b**(n-k) for k in range(n+1))
    print(f"\n  Binomial: ({a}+{b})^{n} = {exact}  via Sigma C({n},k)·{a}^k·{b}^(n-k) = {binom}")


# -- 6. Counting lattice paths -------------------------------------------------
def lattice_paths():
    print("\n=== Lattice Paths from (0,0) to (m,n) ===")
    # Number of paths using only right/up moves = C(m+n, m)
    for m, n in [(2,2),(3,3),(4,2),(5,5)]:
        paths = math.comb(m+n, m)
        print(f"  ({m},{n}): C({m+n},{m}) = {paths}")
    # Catalan numbers: paths that don't cross the diagonal
    print("\n  Catalan numbers C_n = C(2n,n)/(n+1):")
    for n in range(8):
        Cn = math.comb(2*n, n) // (n+1)
        print(f"    C_{n} = {Cn}")


if __name__ == "__main__":
    basic_counting()
    itertools_demo()
    pigeonhole()
    inclusion_exclusion()
    pascals_triangle()
    lattice_paths()
