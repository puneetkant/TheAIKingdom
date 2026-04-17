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

if __name__ == "__main__":
    demo_counting()
    demo_birthday_problem()
    demo_multinomial()
    demo_entropy()
