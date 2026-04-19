"""
Working Example 2: Modules and Packages — Real-World Package Patterns
======================================================================
Demonstrates how a production ML project organises code into packages:
  - A mini `mlutils` package structure
  - Dynamic imports and plugin patterns
  - __all__ export control
  - importlib for runtime module loading
  - Entry-point style CLI via __main__.py pattern

Run:  python working_example2.py
"""
import sys
import os
import importlib
import importlib.util
import inspect
from pathlib import Path

BASE = Path(__file__).parent


# -- 1. Simulate a mini ml package ---------------------------------------------
def create_mini_package() -> None:
    """Write a tiny `mlutils` package to disk to demo package structure."""
    pkg = BASE / "mlutils"
    pkg.mkdir(exist_ok=True)

    (pkg / "__init__.py").write_text(
        '"""mlutils - a demo ML utilities package."""\n'
        "from .metrics import accuracy, f1\n"
        "from .preprocessing import normalize, standardize\n"
        "__all__ = ['accuracy', 'f1', 'normalize', 'standardize']\n"
        "__version__ = '0.1.0'\n",
        encoding='utf-8',
    )

    (pkg / "metrics.py").write_text(
        '"""Classification and regression metrics."""\n'
        "def accuracy(y_true, y_pred):\n"
        "    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)\n\n"
        "def f1(y_true, y_pred):\n"
        "    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 == b)\n"
        "    fp = sum(1 for a, b in zip(y_true, y_pred) if b == 1 != a)\n"
        "    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 != b)\n"
        "    prec = tp / (tp + fp + 1e-9)\n"
        "    rec  = tp / (tp + fn + 1e-9)\n"
        "    return 2 * prec * rec / (prec + rec + 1e-9)\n"
    )

    (pkg / "preprocessing.py").write_text(
        '"""Data preprocessing utilities."""\n'
        "def normalize(data):\n"
        "    lo, hi = min(data), max(data)\n"
        "    return [(x - lo) / (hi - lo + 1e-8) for x in data]\n\n"
        "def standardize(data):\n"
        "    mean = sum(data) / len(data)\n"
        "    std  = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5\n"
        "    return [(x - mean) / (std + 1e-8) for x in data]\n"
    )
    print("=== Mini Package Structure (mlutils/) ===")
    for f in sorted(pkg.rglob("*")):
        rel = f.relative_to(BASE)
        indent = "  " * (len(rel.parts) - 1)
        print(f"  {indent}{f.name}")


# -- 2. Import the mini package -------------------------------------------------
def demo_package_import() -> None:
    print("\n=== Importing mlutils Package ===")
    # Add BASE to sys.path so `mlutils` is findable
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))

    import mlutils
    print(f"  mlutils.__version__  : {mlutils.__version__}")
    print(f"  mlutils.__all__      : {mlutils.__all__}")
    print(f"  mlutils.__file__     : {mlutils.__file__}")

    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 0, 1, 1, 1]
    print(f"  accuracy             : {mlutils.accuracy(y_true, y_pred):.4f}")
    print(f"  f1 score             : {mlutils.f1(y_true, y_pred):.4f}")

    data = [1, 2, 3, 4, 5]
    print(f"  normalize([1..5])    : {[round(x, 3) for x in mlutils.normalize(data)]}")
    print(f"  standardize([1..5])  : {[round(x, 3) for x in mlutils.standardize(data)]}")


# -- 3. Introspection -----------------------------------------------------------
def demo_introspection() -> None:
    print("\n=== Module Introspection ===")
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))

    import mlutils.metrics as m

    print(f"  Module file : {m.__file__}")
    print(f"  Functions   :")
    for name, obj in inspect.getmembers(m, inspect.isfunction):
        sig = inspect.signature(obj)
        print(f"    {name}{sig}")

    # dir() of the module
    public = [x for x in dir(m) if not x.startswith("_")]
    print(f"  Public names: {public}")


# -- 4. Dynamic import with importlib ------------------------------------------
def demo_dynamic_import() -> None:
    print("\n=== Dynamic Import (importlib) ===")
    module_name = "mlutils.preprocessing"
    mod = importlib.import_module(module_name)
    print(f"  Loaded     : {mod.__name__}")

    # Dynamically call a function by name
    fn_name = "normalize"
    fn = getattr(mod, fn_name)
    result = fn([10, 20, 30, 40, 50])
    print(f"  {fn_name}([10..50]) = {[round(x,3) for x in result]}")

    # Reload a module (useful in notebooks / dev)
    importlib.reload(mod)
    print(f"  Reloaded {mod.__name__}")


# -- 5. Standard library highlights --------------------------------------------
def demo_stdlib_highlights() -> None:
    print("\n=== Useful Standard Library Modules ===")
    modules = {
        "os / pathlib":  "File system operations (prefer pathlib in modern code)",
        "json":          "JSON serialisation/deserialisation",
        "csv":           "Read/write CSV files",
        "re":            "Regular expressions",
        "datetime":      "Date/time handling",
        "logging":       "Production logging (not print!)",
        "argparse":      "CLI argument parsing",
        "collections":   "Counter, defaultdict, deque, namedtuple",
        "itertools":     "chain, product, groupby, islice, combinations",
        "functools":     "lru_cache, partial, reduce, wraps",
        "typing":        "Type annotations (List, Dict, Optional, Union)",
        "dataclasses":   "Lightweight classes from annotations",
        "contextlib":    "Context managers (with statement)",
        "concurrent.futures": "Thread / process pool executors",
        "hashlib":       "SHA-256 checksums for data integrity",
    }
    for name, desc in modules.items():
        print(f"  {name:<28} {desc}")


if __name__ == "__main__":
    create_mini_package()
    demo_package_import()
    demo_introspection()
    demo_dynamic_import()
    demo_stdlib_highlights()
