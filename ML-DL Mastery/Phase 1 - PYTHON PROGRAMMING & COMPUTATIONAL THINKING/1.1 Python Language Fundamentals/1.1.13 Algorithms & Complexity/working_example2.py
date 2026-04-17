"""
Working Example 2: Algorithms & Complexity — ML-Relevant Algorithm Analysis
===========================================================================
Implements and benchmarks algorithms used in ML engineering:
  - Sorting algorithms with runtime comparisons
  - Binary search for hyperparameter ranges
  - Heap-based top-K selection (O(n log k) vs O(n log n) sort)
  - Dynamic programming: edit distance (used in NLP)
  - Complexity timing harness

Run:  python working_example2.py
"""
import csv
import heapq
import math
import random
import time
import urllib.request
from pathlib import Path
from functools import lru_cache

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# ── Utility ────────────────────────────────────────────────────────────────────
def timed(label: str, fn, *args) -> tuple:
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {label:<40} {elapsed:>8.3f} ms")
    return result, elapsed


# ── 1. Sorting comparison ──────────────────────────────────────────────────────
def merge_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(a: list, b: list) -> list:
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def insertion_sort(arr: list) -> list:
    arr = arr[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def demo_sorting():
    print("=== Sorting Algorithm Comparison ===")
    random.seed(0)
    sizes = [100, 1_000, 5_000]
    for n in sizes:
        data = [random.random() for _ in range(n)]
        print(f"\n  n={n:>5,}")
        timed("  python sorted() [Timsort O(n log n)]",   sorted, data)
        timed("  merge_sort()   [O(n log n)]",            merge_sort, data)
        if n <= 1000:
            timed("  insertion_sort() [O(n²)]",           insertion_sort, data)
        else:
            print(f"  {'insertion_sort() [O(n²)]':<40} {'skipped (too slow)':>12}")


# ── 2. Binary search for hyperparameter tuning ────────────────────────────────
def binary_search_range(target: float, lo: float, hi: float,
                         fn, tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Find x in [lo, hi] such that fn(x) ≈ target using binary search.
    fn must be monotonically increasing.
    """
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = fn(mid)
        if abs(val - target) < tol:
            return mid
        if val < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def demo_binary_search():
    print("\n=== Binary Search: Find Learning Rate for Target Loss ===")
    # Simulated: loss decreases as sqrt(lr) (toy model)
    def simulated_loss(lr: float) -> float:
        return 1.0 - math.sqrt(lr) * 0.9

    target_loss = 0.5
    # loss = 1 - 0.9*sqrt(lr) → sqrt(lr) = (1-loss)/0.9 → lr = ((1-0.5)/0.9)^2 ≈ 0.309
    found_lr = binary_search_range(
        target=target_loss,
        lo=0.0001, hi=1.0,
        fn=simulated_loss,
        tol=1e-7
    )
    print(f"  Target loss     : {target_loss}")
    print(f"  Found lr        : {found_lr:.8f}")
    print(f"  Achieved loss   : {simulated_loss(found_lr):.8f}")


# ── 3. Top-K selection (heap vs sort) ─────────────────────────────────────────
def top_k_heap(values: list, k: int) -> list:
    """O(n log k) — efficient for small k, large n."""
    return heapq.nlargest(k, values)


def top_k_sort(values: list, k: int) -> list:
    """O(n log n) — simpler but slower for small k."""
    return sorted(values, reverse=True)[:k]


def demo_top_k():
    print("\n=== Top-K Selection: Heap vs Sort ===")
    random.seed(1)
    n = 100_000; k = 10
    values = [random.random() for _ in range(n)]
    print(f"  n={n:,}, k={k}")
    heap_result, t_heap = timed("  heapq.nlargest() [O(n log k)]", top_k_heap, values, k)
    sort_result, t_sort = timed("  sorted()[:k]     [O(n log n)]", top_k_sort, values, k)
    print(f"  Results match : {heap_result == sort_result}")
    print(f"  Speedup       : {t_sort/t_heap:.2f}x")


# ── 4. Dynamic programming: Edit distance ─────────────────────────────────────
def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance — O(m*n) DP."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def demo_edit_distance():
    print("\n=== Edit Distance (Levenshtein — used in NLP/spell correction) ===")
    pairs = [
        ("kitten",  "sitting"),
        ("saturday","sunday"),
        ("transformer", "transducer"),
        ("",         "abc"),
    ]
    for s1, s2 in pairs:
        d = edit_distance(s1, s2)
        print(f"  edit_distance({s1!r:12}, {s2!r:12}) = {d}")


# ── 5. Complexity benchmarks on real data ─────────────────────────────────────
def download_ratings() -> Path:
    dest = DATA / "ratings.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/Shahrukh0/MovieLens-Small/resolve/main/ratings.csv",
                dest
            )
        except Exception:
            # synthetic fallback
            lines = ["userId,movieId,rating,timestamp"]
            for i in range(10_000):
                lines.append(f"{i%100},{i%1000},{random.choice([1.0,2.0,3.0,4.0,5.0])},{1000000+i}")
            dest.write_text("\n".join(lines))
    return dest


def demo_real_data_complexity():
    print("\n=== Complexity on Real Data (MovieLens Ratings) ===")
    dest = download_ratings()
    with open(dest, newline="", encoding="utf-8") as f:
        ratings = [float(r["rating"]) for r in csv.DictReader(f)]

    n = len(ratings)
    print(f"  Ratings loaded: {n:,}")

    # O(n) mean
    timed("  Linear scan mean   [O(n)]",    lambda r: sum(r)/len(r), ratings)
    # O(n log n) sort
    timed("  Sort all ratings   [O(n logn)]", sorted, ratings)
    # O(n log k) top-10
    timed("  Top-10 via heap    [O(n log k)]", heapq.nlargest, 10, ratings)


if __name__ == "__main__":
    demo_sorting()
    demo_binary_search()
    demo_top_k()
    demo_edit_distance()
    demo_real_data_complexity()
