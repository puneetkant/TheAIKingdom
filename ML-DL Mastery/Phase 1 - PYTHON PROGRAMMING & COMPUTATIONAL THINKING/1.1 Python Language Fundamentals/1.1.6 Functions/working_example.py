"""
Working Example: Functions
Covers positional/keyword args, *args/**kwargs, default values,
closures, decorators, recursion, lambda, and functools.
"""
import functools
import time


# ── Basic function ─────────────────────────────────────────────────────────────
def greet(name, greeting="Hello"):
    """Returns a greeting string."""
    return f"{greeting}, {name}!"


def positional_and_keyword_args():
    print("=== Positional & Keyword Args ===")
    print(f"  {greet('Alice')}")
    print(f"  {greet('Bob', greeting='Hi')}")
    print(f"  {greet(greeting='Hey', name='Carol')}")


# ── *args and **kwargs ──────────────────────────────────────────────────────────
def variadic(*args, **kwargs):
    print(f"  args   = {args}")
    print(f"  kwargs = {kwargs}")


def args_kwargs():
    print("\n=== *args / **kwargs ===")
    variadic(1, 2, 3, name="Alice", score=95)

    # Unpacking with * and **
    nums  = [4, 5, 6]
    attrs = {"color": "red", "size": "large"}
    variadic(*nums, **attrs)


# ── Return multiple values ─────────────────────────────────────────────────────
def min_max(numbers):
    return min(numbers), max(numbers)


def returns():
    print("\n=== Multiple Return Values ===")
    lo, hi = min_max([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"  min={lo}, max={hi}")


# ── Closures ──────────────────────────────────────────────────────────────────
def make_multiplier(n):
    def multiplier(x):
        return x * n   # n is captured from enclosing scope
    return multiplier


def closures():
    print("\n=== Closures ===")
    double = make_multiplier(2)
    triple = make_multiplier(3)
    print(f"  double(7) = {double(7)}")
    print(f"  triple(7) = {triple(7)}")


# ── Decorators ────────────────────────────────────────────────────────────────
def timer(func):
    """Measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [{func.__name__}] took {elapsed:.6f}s")
        return result
    return wrapper


def repeat(times):
    """Decorator factory: run function N times."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


@timer
def slow_sum(n):
    return sum(range(n))


@repeat(3)
def say_hi():
    print("  Hi!")


def decorators():
    print("\n=== Decorators ===")
    total = slow_sum(1_000_000)
    print(f"  slow_sum(1_000_000) = {total}")
    say_hi()


# ── Recursion ─────────────────────────────────────────────────────────────────
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci_recursive(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_recursive(n-1, memo) + fibonacci_recursive(n-2, memo)
    return memo[n]


def recursion():
    print("\n=== Recursion ===")
    for n in [0, 1, 5, 10]:
        print(f"  factorial({n}) = {factorial(n)}")
    fibs = [fibonacci_recursive(i) for i in range(15)]
    print(f"  fibonacci(0-14) = {fibs}")


# ── Lambda ────────────────────────────────────────────────────────────────────
def lambdas():
    print("\n=== Lambda Functions ===")
    square   = lambda x: x ** 2
    add      = lambda x, y: x + y
    classify = lambda x: "even" if x % 2 == 0 else "odd"

    print(f"  square(9)      = {square(9)}")
    print(f"  add(3, 4)      = {add(3, 4)}")
    print(f"  classify(7)    = {classify(7)}")

    # With built-ins
    people = [("Alice", 30), ("Bob", 25), ("Carol", 35)]
    by_age = sorted(people, key=lambda p: p[1])
    print(f"  sorted by age  = {by_age}")

    nums = [1, 2, 3, 4, 5, 6]
    evens = list(filter(lambda x: x % 2 == 0, nums))
    cubed = list(map(lambda x: x**3, nums))
    print(f"  filter evens   = {evens}")
    print(f"  map cubed      = {cubed}")


# ── functools ────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=128)
def fib_cached(n):
    if n < 2:
        return n
    return fib_cached(n-1) + fib_cached(n-2)


def functools_demo():
    print("\n=== functools ===")
    # lru_cache
    print(f"  fib_cached(40) = {fib_cached(40)}")
    print(f"  cache info     = {fib_cached.cache_info()}")

    # partial
    power_of_2 = functools.partial(pow, 2)
    print(f"  2**10 via partial = {power_of_2(10)}")

    # reduce
    from functools import reduce
    product = reduce(lambda a, b: a * b, range(1, 8))
    print(f"  reduce product 1..7 = {product}")


if __name__ == "__main__":
    positional_and_keyword_args()
    args_kwargs()
    returns()
    closures()
    decorators()
    recursion()
    lambdas()
    functools_demo()
