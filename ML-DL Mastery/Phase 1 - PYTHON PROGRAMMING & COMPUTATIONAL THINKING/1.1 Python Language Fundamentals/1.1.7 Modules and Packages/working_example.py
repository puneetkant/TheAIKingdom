"""
Working Example: Modules and Packages
Covers import styles, __name__, standard library highlights,
creating your own module (inline demo), and __all__.
"""

# ── import styles ─────────────────────────────────────────────────────────────
import os
import sys
import math
import random
import datetime
from pathlib import Path
from collections import Counter
from itertools import islice, chain, combinations
from functools import reduce


def import_styles():
    print("=== Import Styles ===")
    # import module
    print(f"  math.pi         = {math.pi:.6f}")
    print(f"  math.sqrt(144)  = {math.sqrt(144)}")

    # from module import name
    from math import log2, factorial
    print(f"  log2(1024)      = {log2(1024)}")
    print(f"  factorial(10)   = {factorial(10)}")

    # import with alias
    import json as _json
    data = _json.dumps({"key": "value", "n": 42}, indent=2)
    back = _json.loads(data)
    print(f"  json roundtrip  = {back}")


def name_guard():
    print("\n=== __name__ guard ===")
    print(f"  __name__ in this script = {__name__!r}")
    print("  Code inside 'if __name__ == \"__main__\":' only runs when executed directly,")
    print("  not when imported as a module.")


def os_and_path():
    print("\n=== os & pathlib ===")
    cwd = Path.cwd()
    print(f"  cwd            = {cwd}")
    print(f"  os.sep         = {os.sep!r}")
    print(f"  home dir       = {Path.home()}")

    # Path operations
    p = Path("/some/example/file.txt")
    print(f"  stem           = {p.stem}")
    print(f"  suffix         = {p.suffix}")
    print(f"  parent         = {p.parent}")
    print(f"  parts          = {p.parts}")

    # os.environ sampling
    env_keys = list(os.environ.keys())[:5]
    print(f"  env vars (first 5 keys): {env_keys}")


def sys_info():
    print("\n=== sys module ===")
    print(f"  sys.version    = {sys.version.split()[0]}")
    print(f"  sys.platform   = {sys.platform}")
    print(f"  sys.argv       = {sys.argv}")
    print(f"  sys.path[:3]   = {sys.path[:3]}")
    print(f"  sys.maxsize    = {sys.maxsize}")


def datetime_demo():
    print("\n=== datetime module ===")
    now = datetime.datetime.now()
    today = datetime.date.today()
    print(f"  now            = {now:%Y-%m-%d %H:%M:%S}")
    print(f"  today          = {today}")
    delta = datetime.timedelta(days=30)
    print(f"  30 days later  = {today + delta}")

    # Parsing a date string
    parsed = datetime.datetime.strptime("2024-07-04", "%Y-%m-%d")
    print(f"  parsed         = {parsed.date()}")


def random_demo():
    print("\n=== random module ===")
    random.seed(42)
    print(f"  random()       = {random.random():.4f}")
    print(f"  randint(1,100) = {random.randint(1, 100)}")
    print(f"  choice         = {random.choice(['a', 'b', 'c', 'd'])}")
    nums = list(range(1, 11))
    random.shuffle(nums)
    print(f"  shuffled       = {nums}")
    print(f"  sample 3       = {random.sample(nums, 3)}")


def itertools_demo():
    print("\n=== itertools module ===")
    # islice — lazy slice of any iterable
    naturals = (n for n in range(1, 1000))
    print(f"  islice(1..∞, 7)  = {list(islice(naturals, 7))}")

    # chain — flatten iterables
    combined = list(chain([1, 2], [3, 4], [5]))
    print(f"  chain            = {combined}")

    # combinations
    cards = ["A", "K", "Q"]
    hands = list(combinations(cards, 2))
    print(f"  combinations 2   = {hands}")

    # reduce (from functools)
    print(f"  reduce product   = {reduce(lambda a, b: a*b, range(1, 8))}")


def inline_module_demo():
    """Simulate what a module looks like when imported."""
    print("\n=== Simulated Module ===")
    module_code = '''\
# mymodule.py
__all__ = ["add", "multiply"]

_PRIVATE = "internal"   # not exported by default

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def _helper():
    return "private helper"

if __name__ == "__main__":
    print("Running as script")
'''
    print("  Content of a typical mymodule.py:")
    for line in module_code.strip().splitlines():
        print(f"    {line}")

    print("\n  Usage from another file:")
    print("    from mymodule import add, multiply  # only __all__ items if 'import *'")
    print("    add(2, 3)       → 5")
    print("    multiply(4, 5)  → 20")


if __name__ == "__main__":
    import_styles()
    name_guard()
    os_and_path()
    sys_info()
    datetime_demo()
    random_demo()
    itertools_demo()
    inline_module_demo()
