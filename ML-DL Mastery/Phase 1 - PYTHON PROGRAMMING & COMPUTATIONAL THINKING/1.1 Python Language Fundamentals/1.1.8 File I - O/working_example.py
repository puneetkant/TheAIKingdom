"""
Working Example: File I/O
Covers text files, binary files, CSV, JSON, pathlib,
context managers, and temporary files.
"""
import csv
import json
import os
import tempfile
from pathlib import Path


# ── Text files ────────────────────────────────────────────────────────────────
def text_file_demo():
    print("=== Text File Read/Write ===")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8") as f:
        path = Path(f.name)
        f.write("Line 1: Hello, File I/O!\n")
        f.write("Line 2: Python makes it easy.\n")
        f.write("Line 3: Always use context managers.\n")

    # Read entire file
    content = path.read_text(encoding="utf-8")
    print(f"  Full content:\n{content.rstrip()}")

    # Read line by line
    print("  Line by line:")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            print(f"    [{i}] {line.rstrip()}")

    # Append
    with path.open("a", encoding="utf-8") as f:
        f.write("Line 4: Appended later.\n")

    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"  After append, total lines: {len(lines)}")
    path.unlink()


# ── Binary files ──────────────────────────────────────────────────────────────
def binary_file_demo():
    print("\n=== Binary File ===")
    data = bytes(range(16))          # 16 bytes: 0x00 – 0x0F
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        path = Path(f.name)
        f.write(data)

    raw = path.read_bytes()
    print(f"  bytes  : {raw.hex(' ')}")
    print(f"  length : {len(raw)} bytes")
    path.unlink()


# ── CSV ───────────────────────────────────────────────────────────────────────
def csv_demo():
    print("\n=== CSV File ===")
    rows = [
        {"name": "Alice",   "score": 92, "grade": "A"},
        {"name": "Bob",     "score": 74, "grade": "B"},
        {"name": "Charlie", "score": 55, "grade": "C"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                    delete=False, newline="",
                                    encoding="utf-8") as f:
        path = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=["name", "score", "grade"])
        writer.writeheader()
        writer.writerows(rows)

    print("  Written CSV:")
    with path.open(encoding="utf-8") as f:
        for line in f:
            print(f"    {line.rstrip()}")

    print("  Read back as dicts:")
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(f"    {dict(row)}")

    path.unlink()


# ── JSON ──────────────────────────────────────────────────────────────────────
def json_demo():
    print("\n=== JSON File ===")
    config = {
        "model": "random_forest",
        "hyperparams": {"n_estimators": 100, "max_depth": 5},
        "features": ["age", "income", "score"],
        "active": True,
        "threshold": None,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, encoding="utf-8") as f:
        path = Path(f.name)
        json.dump(config, f, indent=2)

    text = path.read_text(encoding="utf-8")
    print("  Written JSON:")
    print("\n".join("    " + l for l in text.splitlines()))

    loaded = json.loads(text)
    print(f"\n  model     = {loaded['model']}")
    print(f"  threshold = {loaded['threshold']!r}")  # None survives
    path.unlink()

    # json.dumps / loads (no file)
    s = json.dumps({"a": 1, "b": [2, 3]}, separators=(",", ":"))
    print(f"\n  compact dumps = {s}")
    print(f"  loads back    = {json.loads(s)}")


# ── pathlib ───────────────────────────────────────────────────────────────────
def pathlib_demo():
    print("\n=== pathlib ===")
    p = Path(".")
    print(f"  cwd            = {p.resolve()}")

    tmp = Path(tempfile.mkdtemp())
    (tmp / "subdir").mkdir()
    (tmp / "subdir" / "hello.txt").write_text("hi", encoding="utf-8")
    (tmp / "data.csv").write_text("a,b\n1,2", encoding="utf-8")

    print(f"  all files under tmp:")
    for child in sorted(tmp.rglob("*")):
        print(f"    {child.relative_to(tmp)}  (dir={child.is_dir()})")

    # glob
    txts = list(tmp.rglob("*.txt"))
    print(f"  *.txt files: {[str(f.name) for f in txts]}")

    # cleanup
    import shutil
    shutil.rmtree(tmp)


# ── Context manager (custom) ──────────────────────────────────────────────────
class ManagedFile:
    def __init__(self, path, mode="r"):
        self.path = path
        self.mode = mode
        self._file = None

    def __enter__(self):
        self._file = open(self.path, self.mode, encoding="utf-8")
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False   # don't suppress exceptions


def context_manager_demo():
    print("\n=== Custom Context Manager ===")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        path = f.name
        f.write("Context managers ensure cleanup.\n")

    with ManagedFile(path) as f:
        print(f"  read via custom ctx: {f.read().rstrip()!r}")

    os.unlink(path)
    print("  file closed and removed automatically")


if __name__ == "__main__":
    text_file_demo()
    binary_file_demo()
    csv_demo()
    json_demo()
    pathlib_demo()
    context_manager_demo()
