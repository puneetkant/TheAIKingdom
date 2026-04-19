"""
Working Example 2: Combinatorics — Counting, Permutations, Combinations
========================================================================
Factorials, nCr, nPr, stars-and-bars, birthday problem Monte Carlo,
multinomial coefficients, password entropy estimation.

Run:  python working_example2.py
"""
import math, random, itertools
from pathlib import Path
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_counting():
    print("=== Permutations & Combinations ===")
    n, k = 10, 3
    perm = math.factorial(n) // math.factorial(n - k)   # nPk
    comb = math.comb(n, k)                               # nCk
    print(f"  n={n}, k={k}")
    print(f"  P(n,k) = {perm}   (ordered selections)")
    print(f"  C(n,k) = {comb}   (unordered selections)")
    print(f"  Verify: math.perm(10,3) = {math.perm(10,3)}")

def demo_birthday_problem():
    print("\n=== Birthday Problem (Monte Carlo) ===")
    random.seed(42)
    def prob_shared(n, trials=50_000):
        hits = 0
        for _ in range(trials):
            birthdays = [random.randint(0, 364) for _ in range(n)]
            if len(set(birthdays)) < n:
                hits += 1
        return hits / trials

    # Analytic: P(no shared) = 365!/((365-n)! * 365^n)
    def prob_analytic(n):
        p = 1.0
        for i in range(1, n):
            p *= (365 - i) / 365
        return 1 - p

    print(f"  {'n':>4}  {'MC prob':>9}  {'Analytic':>9}")
    for n in [10, 20, 23, 30, 50]:
        mc  = prob_shared(n)
        ana = prob_analytic(n)
        print(f"  {n:>4}  {mc:>9.4f}  {ana:>9.4f}")

def demo_multinomial():
    print("\n=== Multinomial Coefficients ===")
    # How many ways to arrange letters in "MISSISSIPPI"?
    # n=11, M:1, I:4, S:4, P:2  -> 11!/(1!4!4!2!)
    word = "MISSISSIPPI"
    from collections import Counter
    counts = Counter(word)
    num = math.factorial(len(word))
    den = math.prod(math.factorial(v) for v in counts.values())
    print(f"  Arrangements of '{word}': {num // den:,}")

def demo_entropy():
    print("\n=== Password Entropy ===")
    # Entropy = log2(alphabet_size^length) = length * log2(size)
    configs = [
        ("lowercase", 26, 8),
        ("lower+digits", 36, 8),
        ("lower+upper+digits+sym", 95, 8),
        ("lower+upper+digits+sym", 95, 12),
        ("lower+upper+digits+sym", 95, 16),
    ]
    for name, size, length in configs:
        entropy = length * math.log2(size)
        guesses = size ** length
        print(f"  {name} len={length}: {entropy:.1f} bits  ({guesses:.2e} guesses)")

def demo_stars_and_bars():
    print("\n=== Stars and Bars (Multiset Coefficients) ===")
    # Ways to distribute k identical items into n distinct bins = C(n+k-1, k)
    print(f"  C(n+k-1, k) = ways to put k identical items into n bins")
    print(f"  {'n':>4}  {'k':>4}  {'C(n+k-1,k)':>12}")
    for n, k in [(3, 5), (4, 3), (2, 10), (5, 0), (1, 7)]:
        result = math.comb(n + k - 1, k)
        verify = math.comb(n + k - 1, n - 1)
        print(f"  {n:>4}  {k:>4}  {result:>12,}  (alt formula C(n+k-1,n-1)={verify}, equal={result==verify})")
    # Example: 4 ice-cream flavors, choose 6 scoops with repetition
    n_f, sc = 4, 6
    print(f"  Ice cream {n_f} flavors, {sc} scoops (with repeats): {math.comb(n_f+sc-1, sc):,} combos")


def demo_pigeonhole():
    print("\n=== Pigeonhole Principle ===")
    # If n+1 items go into n bins, at least one bin has >= 2 items.
    # Birthday: 365 days (bins). Guaranteed collision at n=366.
    print("  Pigeonhole guarantees collision when n > 365 (i.e., 366+ people).")
    # Expected collisions E = C(n,2)/365; compare to probability
    print(f"\n  {'n':>5}  {'E[collisions]':>15}  {'P(>=1 collision)':>18}")
    for n in [5, 10, 20, 23, 30, 50, 100, 366]:
        expected = math.comb(n, 2) / 365
        if n > 365:
            p_coll = 1.0
        else:
            log_p_none = sum(math.log((365 - i) / 365) for i in range(n))
            p_coll = 1.0 - math.exp(log_p_none)
        print(f"  {n:>5}  {expected:>15.4f}  {p_coll:>18.6f}")


def demo_inclusion_exclusion():
    print("\n=== Inclusion-Exclusion Principle ===")
    # |A u B u C| = |A|+|B|+|C| - |AnB| - |AnC| - |BnC| + |AnBnC|
    N = 100
    A = set(range(2, N + 1, 2))   # divisible by 2
    B = set(range(3, N + 1, 3))   # divisible by 3
    C = set(range(5, N + 1, 5))   # divisible by 5
    ie = (len(A) + len(B) + len(C)
          - len(A & B) - len(A & C) - len(B & C)
          + len(A & B & C))
    direct = len(A | B | C)
    print(f"  Divisible by 2, 3, or 5 in [1,{N}]:")
    print(f"  |A|={len(A)}, |B|={len(B)}, |C|={len(C)}, "
          f"|AnB|={len(A&B)}, |AnC|={len(A&C)}, |BnC|={len(B&C)}, |AnBnC|={len(A&B&C)}")
    print(f"  IE formula = {ie},  direct count = {direct},  match={ie==direct}")
    # Derangements: D(n) = n! * sum_{k=0}^{n} (-1)^k / k!
    print(f"\n  Derangements D(n) via inclusion-exclusion (no fixed points):")
    for n in range(1, 9):
        D = round(math.factorial(n) * sum((-1)**k / math.factorial(k) for k in range(n + 1)))
        print(f"    D({n}) = {D}")


def demo_generating_function():
    print("\n=== Generating Functions (Coin Change Counting) ===")
    # OGF: (1/(1-x)) * (1/(1-x^2)) * (1/(1-x^5))
    # Count ways to make n cents with coins {1, 2, 5} via DP (= poly multiplication)
    max_val = 20
    dp = [0] * (max_val + 1)
    dp[0] = 1
    for coin in [1, 2, 5]:
        for i in range(coin, max_val + 1):
            dp[i] += dp[i - coin]
    print(f"  Ways to make n cents with coins {{1, 2, 5}} (n=0..{max_val}):")
    for n in range(max_val + 1):
        bar = "#" * dp[n]
        print(f"    n={n:>2}: {dp[n]:>3} ways  {bar}")
    print(f"  Spot-check n=10: {dp[10]} ways (expected 10)")
    print(f"  Spot-check n=20: {dp[20]} ways (expected 29)")


def demo_catalan_numbers():
    print("\n=== Catalan Numbers ===")
    # C(n) = C(2n,n)/(n+1) = number of valid bracket sequences, BST shapes, etc.
    print(f"  C(n) = C(2n,n)/(n+1)  -- valid bracket seqs, triangulations, BSTs")
    print(f"  {'n':>4}  {'Catalan C(n)':>14}  {'DP recurrence':>15}")
    # DP: C(0)=1, C(n) = sum_{i=0}^{n-1} C(i)*C(n-1-i)
    dp = [0] * 12
    dp[0] = 1
    for n in range(1, 12):
        dp[n] = sum(dp[i] * dp[n - 1 - i] for i in range(n))
    for n in range(10):
        formula = math.comb(2 * n, n) // (n + 1)
        print(f"  {n:>4}  {formula:>14,}  {dp[n]:>15,}  match={formula==dp[n]}")
    # Application: count valid bracket sequences for n pairs
    print(f"\n  Valid bracket sequences for n pairs: C(3)={dp[3]} (e.g. ()()(), (())(), ...)")

    # Pascal's triangle coefficients (binomial row sums)
    print(f"\n=== Pascal's Triangle (rows 0-6) ===")
    for row in range(7):
        coeffs = [math.comb(row, k) for k in range(row + 1)]
        s = "  " + " ".join(f"{c:>3}" for c in coeffs)
        print(s)
    print(f"  Row sums: {[sum(math.comb(r,k) for k in range(r+1)) for r in range(7)]}")
    print(f"  (Each row sum = 2^n)")


def demo_stirling_approximation():
    print("\n=== Stirling's Approximation: n! ~= sqrt(2*pi*n) * (n/e)^n ===")
    print(f"  {'n':>5}  {'n! exact':>15}  {'Stirling':>18}  {'rel err %':>10}")
    for n in [1, 2, 5, 10, 20, 50, 100]:
        exact    = math.factorial(n)
        stirling = math.sqrt(2 * math.pi * n) * (n / math.e) ** n
        rel_err  = abs(exact - stirling) / exact * 100
        print(f"  {n:>5}  {exact:>15,}  {stirling:>18.2f}  {rel_err:>10.4f}")
    # Log-Stirling is used in entropy and information theory
    print(f"\n  log(n!) via Stirling vs exact sum:")
    for n in [10, 100, 1_000, 10_000]:
        log_exact    = sum(math.log(k) for k in range(1, n + 1))
        log_stirling = n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
        diff = abs(log_exact - log_stirling)
        print(f"    n={n:>6}: exact={log_exact:.4f}, Stirling={log_stirling:.4f}, |diff|={diff:.6f}")
    print(f"  (Used in entropy H = -sum p*log(p) computations at scale)")


if __name__ == "__main__":
    demo_counting()
    demo_birthday_problem()
    demo_multinomial()
    demo_entropy()
    demo_stars_and_bars()
    demo_pigeonhole()
    demo_inclusion_exclusion()
    demo_generating_function()
    demo_catalan_numbers()
    demo_stirling_approximation()
