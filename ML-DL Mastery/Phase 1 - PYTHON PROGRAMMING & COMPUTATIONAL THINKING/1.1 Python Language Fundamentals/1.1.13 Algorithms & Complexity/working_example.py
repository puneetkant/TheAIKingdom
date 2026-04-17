"""
Working Example: Algorithms & Complexity
Covers Big-O analysis, sorting algorithms, searching,
recursion vs iteration, and time/space measurement.
"""
import time
import random
import sys
from functools import lru_cache


# ── Big-O complexity reference ────────────────────────────────────────────────
def big_o_demo():
    print("=== Big-O Complexity Reference ===")
    examples = [
        ("O(1)",      "dict[key], list[i], set membership"),
        ("O(log n)",  "binary search, balanced BST operations"),
        ("O(n)",      "linear scan, list.count(), sum()"),
        ("O(n log n)","merge sort, heap sort, sorted()"),
        ("O(n²)",     "bubble/insertion/selection sort (worst), nested loops"),
        ("O(2ⁿ)",     "naive Fibonacci, power set generation"),
        ("O(n!)",     "permutation generation, brute-force TSP"),
    ]
    for complexity, example in examples:
        print(f"  {complexity:<12} {example}")


# ── Sorting algorithms ────────────────────────────────────────────────────────
def bubble_sort(arr):
    """O(n²) — simple, rarely used in practice."""
    a = arr[:]
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    return a


def insertion_sort(arr):
    """O(n²) worst, O(n) best — good for nearly-sorted data."""
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key
    return a


def merge_sort(arr):
    """O(n log n) — divide and conquer, stable."""
    if len(arr) <= 1:
        return arr[:]
    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """O(n log n) avg, O(n²) worst — in-place variant."""
    a = arr[:]
    _quick_sort(a, 0, len(a) - 1)
    return a


def _quick_sort(a, lo, hi):
    if lo < hi:
        p = _partition(a, lo, hi)
        _quick_sort(a, lo, p - 1)
        _quick_sort(a, p + 1, hi)


def _partition(a, lo, hi):
    pivot = a[hi]
    i = lo - 1
    for j in range(lo, hi):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i+1], a[hi] = a[hi], a[i+1]
    return i + 1


def sorting_demo():
    print("\n=== Sorting Algorithms ===")
    random.seed(0)
    data = random.sample(range(1, 101), 10)
    print(f"  input  : {data}")

    algorithms = [
        ("bubble_sort",    bubble_sort),
        ("insertion_sort", insertion_sort),
        ("merge_sort",     merge_sort),
        ("quick_sort",     quick_sort),
        ("sorted() builtin", lambda x: sorted(x)),
    ]
    for name, fn in algorithms:
        result = fn(data)
        print(f"  {name:<22}: {result}")


def sorting_benchmark():
    print("\n=== Sorting Benchmark (n=3000) ===")
    random.seed(42)
    data = [random.random() for _ in range(3000)]

    def timed(fn, arr):
        start = time.perf_counter()
        fn(arr)
        return time.perf_counter() - start

    print(f"  bubble_sort   : {timed(bubble_sort,    data):.4f}s")
    print(f"  insertion_sort: {timed(insertion_sort, data):.4f}s")
    print(f"  merge_sort    : {timed(merge_sort,     data):.4f}s")
    print(f"  quick_sort    : {timed(quick_sort,     data):.4f}s")
    print(f"  sorted()      : {timed(sorted,         data):.4f}s")


# ── Searching ─────────────────────────────────────────────────────────────────
def linear_search(arr, target):
    """O(n)"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


def binary_search(arr, target):
    """O(log n) — requires sorted array."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def searching_demo():
    print("\n=== Searching ===")
    arr    = sorted(random.sample(range(1, 200), 20))
    target = arr[7]   # guaranteed hit
    miss   = 999

    print(f"  array   : {arr}")
    print(f"  target  : {target}")
    print(f"  linear_search({target}) = index {linear_search(arr, target)}")
    print(f"  binary_search({target}) = index {binary_search(arr, target)}")
    print(f"  linear_search({miss})   = {linear_search(arr, miss)}")
    print(f"  binary_search({miss})   = {binary_search(arr, miss)}")


# ── Recursion vs iteration ────────────────────────────────────────────────────
def fib_naive(n):
    if n < 2: return n
    return fib_naive(n-1) + fib_naive(n-2)   # O(2ⁿ)


@lru_cache(maxsize=None)
def fib_memo(n):
    if n < 2: return n
    return fib_memo(n-1) + fib_memo(n-2)     # O(n) with cache


def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a                                   # O(n) time, O(1) space


def recursion_vs_iteration():
    print("\n=== Recursion vs Iteration (Fibonacci) ===")
    test_ns = [10, 20, 30]
    for n in test_ns:
        start = time.perf_counter()
        r1 = fib_naive(n)
        t1 = time.perf_counter() - start

        start = time.perf_counter()
        r2 = fib_memo(n)
        t2 = time.perf_counter() - start

        start = time.perf_counter()
        r3 = fib_iter(n)
        t3 = time.perf_counter() - start

        print(f"  fib({n:2d}): naive={r1} ({t1:.5f}s)  "
              f"memo={r2} ({t2:.5f}s)  "
              f"iter={r3} ({t3:.5f}s)")


# ── Space complexity ──────────────────────────────────────────────────────────
def space_demo():
    print("\n=== Space Complexity ===")
    sizes = [1_000, 10_000, 100_000]
    for n in sizes:
        lst  = list(range(n))
        gen  = (x for x in range(n))
        lst_bytes = sys.getsizeof(lst)
        gen_bytes = sys.getsizeof(gen)
        print(f"  n={n:>7}: list={lst_bytes:>8} bytes  generator={gen_bytes} bytes")


if __name__ == "__main__":
    big_o_demo()
    sorting_demo()
    sorting_benchmark()
    searching_demo()
    recursion_vs_iteration()
    space_demo()
