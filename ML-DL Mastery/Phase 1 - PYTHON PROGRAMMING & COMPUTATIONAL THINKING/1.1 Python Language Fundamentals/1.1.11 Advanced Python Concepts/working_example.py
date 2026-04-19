"""
Working Example: Advanced Python Concepts
Covers generators, iterators, context managers (contextlib),
decorators (advanced), metaclasses, descriptors, and slots.
"""
import contextlib
import functools
import time


# -- Generators ----------------------------------------------------------------
def count_up(start=0, step=1):
    """Infinite generator — yields values lazily."""
    n = start
    while True:
        yield n
        n += step


def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def take(n, iterable):
    from itertools import islice
    return list(islice(iterable, n))


def generator_pipeline(data):
    """Compose generators as a lazy pipeline."""
    cleaned    = (x.strip()   for x in data)
    non_empty  = (x           for x in cleaned   if x)
    uppercased = (x.upper()   for x in non_empty)
    return uppercased


def generators_demo():
    print("=== Generators ===")
    print(f"  count_up first 8  : {take(8, count_up(1, 2))}")
    print(f"  fibonacci first 12: {take(12, fibonacci())}")

    # Generator expression vs list comprehension
    gen_expr = (x**2 for x in range(1_000_000))   # no memory spike
    lst_comp = [x**2 for x in range(5)]
    print(f"  gen_expr type  : {type(gen_expr).__name__}")
    print(f"  list_comp      : {lst_comp}")

    # Send values into a generator
    def accumulator():
        total = 0
        while True:
            value = yield total
            if value is None:
                break
            total += value

    acc = accumulator()
    next(acc)               # prime the generator
    acc.send(10)
    acc.send(20)
    result = acc.send(5)
    print(f"  accumulator after 10+20+5 = {result}")

    # Pipeline
    raw = ["  hello ", "", "world  ", "  python ", " "]
    processed = list(generator_pipeline(raw))
    print(f"  pipeline result: {processed}")


# -- Custom Iterator -----------------------------------------------------------
class Countdown:
    """Implements __iter__ / __next__ directly."""
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value


def iterator_demo():
    print("\n=== Custom Iterator ===")
    for n in Countdown(5):
        print(n, end=" ")
    print()


# -- Context Managers ---------------------------------------------------------
class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        return False   # don't suppress exceptions


@contextlib.contextmanager
def managed_resource(name):
    """Generator-based context manager."""
    print(f"  -> acquiring {name}")
    try:
        yield name.upper()
    finally:
        print(f"  -> releasing {name}")


def context_managers_demo():
    print("\n=== Context Managers ===")
    with Timer() as t:
        _ = sum(range(1_000_000))
    print(f"  sum(0..999999) in {t.elapsed:.4f}s")

    with managed_resource("database connection") as resource:
        print(f"  using: {resource}")

    # contextlib.suppress
    with contextlib.suppress(ZeroDivisionError):
        _ = 1 / 0
    print("  ZeroDivisionError suppressed silently")


# -- Advanced Decorators -------------------------------------------------------
def retry(times=3, exceptions=(Exception,), delay=0):
    """Decorator factory: retry on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"  attempt {attempt} failed: {e}")
                    if attempt == times:
                        raise
                    if delay:
                        time.sleep(delay)
        return wrapper
    return decorator


def memoize(func):
    """Simple memoization decorator (functools.lru_cache is better in practice)."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper


_attempt_counter = 0

@retry(times=3, exceptions=(ValueError,))
def flaky_function():
    global _attempt_counter
    _attempt_counter += 1
    if _attempt_counter < 3:
        raise ValueError(f"not ready (attempt {_attempt_counter})")
    return "success"


@memoize
def slow_fib(n):
    if n < 2: return n
    return slow_fib(n-1) + slow_fib(n-2)


def decorators_demo():
    print("\n=== Advanced Decorators ===")
    result = flaky_function()
    print(f"  flaky_function returned: {result!r}")

    print(f"  slow_fib(35) = {slow_fib(35)}")
    print(f"  cache size   = {len(slow_fib.cache)}")


# -- Descriptors ---------------------------------------------------------------
class Validated:
    """Descriptor: enforces a min/max range on a numeric attribute."""
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f"{self.name} must be in [{self.min_val}, {self.max_val}], got {value}"
            )
        setattr(obj, self.storage_name, value)


class Thermostat:
    temperature = Validated(-40, 100)

    def __init__(self, temp):
        self.temperature = temp


def descriptors_demo():
    print("\n=== Descriptors ===")
    t = Thermostat(22)
    print(f"  temp = {t.temperature}")
    t.temperature = 37
    print(f"  updated = {t.temperature}")
    try:
        t.temperature = 150
    except ValueError as e:
        print(f"  ValueError: {e}")


# -- __slots__ ----------------------------------------------------------------
class PointSlots:
    __slots__ = ("x", "y")   # prevents arbitrary attribute creation

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"PointSlots({self.x}, {self.y})"


def slots_demo():
    print("\n=== __slots__ ===")
    p = PointSlots(3, 4)
    print(f"  {p}")
    try:
        p.z = 5
    except AttributeError as e:
        print(f"  AttributeError: {e}")
    print("  __slots__ saves memory vs __dict__")


if __name__ == "__main__":
    generators_demo()
    iterator_demo()
    context_managers_demo()
    decorators_demo()
    descriptors_demo()
    slots_demo()
