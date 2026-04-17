"""
Working Example 2: Functions — Real-World ML Utility Functions
==============================================================
Builds a mini ML utility library with:
  - Higher-order functions and decorators for timing/caching
  - Generators for memory-efficient data streaming
  - Closures for stateful metric tracking
  - Partial functions and function composition

Run:  python working_example2.py
"""
import urllib.request
import csv
import time
import functools
import math
from pathlib import Path
from typing import Callable, Generator, TypeVar, Any

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
T = TypeVar("T")


# ── 1. Decorators for ML utilities ────────────────────────────────────────────
def timer(func: Callable) -> Callable:
    """Decorator: measure and print function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱  {func.__name__} took {elapsed*1000:.2f} ms")
        return result
    return wrapper


def lru_cache_demo(maxsize: int = 128) -> Callable:
    """Decorator factory: memoisation with hit/miss tracking."""
    def decorator(func: Callable) -> Callable:
        cache: dict = {}
        hits = misses = 0
        @functools.wraps(func)
        def wrapper(*args):
            nonlocal hits, misses
            if args in cache:
                hits += 1
                return cache[args]
            misses += 1
            result = func(*args)
            if len(cache) < maxsize:
                cache[args] = result
            return result
        wrapper.cache_info = lambda: f"hits={hits}, misses={misses}, size={len(cache)}"
        return wrapper
    return decorator


@lru_cache_demo(maxsize=50)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def demo_decorators():
    print("=== Decorators ===")

    @timer
    def compute_stats(data: list[float]) -> dict:
        mean = sum(data) / len(data)
        var  = sum((x - mean) ** 2 for x in data) / len(data)
        return {"mean": mean, "std": math.sqrt(var), "n": len(data)}

    import random
    random.seed(0)
    data = [random.gauss(0, 1) for _ in range(100_000)]
    stats = compute_stats(data)
    print(f"  Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Memoised fibonacci
    [fibonacci(n) for n in range(35)]
    print(f"  fibonacci(34) = {fibonacci(34)}")
    print(f"  Cache info: {fibonacci.cache_info()}")


# ── 2. Generators for memory-efficient data streaming ─────────────────────────
def csv_stream(path: Path, chunk_size: int = 100) -> Generator[list[dict], None, None]:
    """Yield chunks of rows from a CSV without loading all into RAM."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def pipeline(*funcs: Callable) -> Callable:
    """Compose multiple functions left-to-right: pipeline(f, g, h)(x) = h(g(f(x)))."""
    def composed(x):
        for f in funcs:
            x = f(x)
        return x
    return composed


def demo_generators():
    dest = DATA_DIR / "titanic.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv",
                dest
            )
        except Exception:
            dest.write_text(
                "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
                + "\n".join(f"{i},1,1,Person {i},male,{20+i},0,0,T{i},{50+i}.,C{i},S" for i in range(1, 50))
            )
    print("\n=== Generators (memory-efficient streaming) ===")
    total_fare = 0.0
    row_count  = 0
    for chunk in csv_stream(dest, chunk_size=50):
        for row in chunk:
            fare = float(row.get("Fare") or 0)
            total_fare += fare
            row_count  += 1
    print(f"  Streamed {row_count} rows, total fare: £{total_fare:.2f}")

    # Generator expression (lazy evaluation)
    gen = (float(row.get("Fare") or 0)
           for chunk in csv_stream(dest)
           for row in chunk
           if float(row.get("Fare") or 0) > 100)
    high_fares = list(gen)
    print(f"  Fares > £100: {len(high_fares)} — avg: £{sum(high_fares)/max(len(high_fares),1):.2f}")


# ── 3. Closures — stateful metric trackers ────────────────────────────────────
def make_metric_tracker(name: str) -> Callable:
    """Closure: returns a function that tracks running mean."""
    values: list[float] = []

    def track(x: float) -> dict:
        values.append(x)
        mean = sum(values) / len(values)
        best = min(values)
        return {"name": name, "mean": mean, "best": best, "n": len(values)}

    return track


def demo_closures():
    print("\n=== Closures — Stateful Metric Trackers ===")
    track_val_loss = make_metric_tracker("val_loss")
    track_accuracy = make_metric_tracker("accuracy")

    import random
    random.seed(1)
    for epoch in range(1, 6):
        vl  = random.gauss(0.5 - epoch * 0.08, 0.02)
        acc = random.gauss(0.6 + epoch * 0.05, 0.01)
        vl_state  = track_val_loss(vl)
        acc_state = track_accuracy(acc)
        print(f"  Epoch {epoch}: val_loss={vl:.4f} (best={vl_state['best']:.4f}) "
              f"acc={acc:.4f} (best={acc_state['best']:.4f})")


# ── 4. Partial functions and function composition ─────────────────────────────
def demo_partial_and_composition():
    print("\n=== Partial Functions & Composition ===")

    # functools.partial
    def normalize(x: float, min_val: float, max_val: float) -> float:
        return (x - min_val) / (max_val - min_val + 1e-8)

    normalize_0_1 = functools.partial(normalize, min_val=0.0, max_val=1.0)
    normalize_fare = functools.partial(normalize, min_val=0.0, max_val=512.0)

    fares = [7.25, 71.28, 53.1, 0.0, 512.32]
    print("  Fares normalized [0, 512]:")
    for f in fares:
        print(f"    {f:>8.2f} → {normalize_fare(f):.4f}")

    # Function composition
    strip_lower   = lambda s: s.strip().lower()
    replace_spaces = lambda s: s.replace(" ", "_")
    clean_name    = pipeline(strip_lower, replace_spaces)

    names = ["  Ada Lovelace  ", "Alan Turing", " Grace Hopper "]
    print("\n  Cleaned names (pipeline):")
    for n in names:
        print(f"    {n!r:25} → {clean_name(n)!r}")


if __name__ == "__main__":
    demo_decorators()
    demo_generators()
    demo_closures()
    demo_partial_and_composition()
