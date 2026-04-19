"""
Working Example 2: Shell Basics — Cross-Platform Shell Scripting via Python
===========================================================================
Demonstrates writing Python scripts that replicate common shell operations:
  - Directory navigation (pushd/popd equivalent)
  - File search (glob, recursive walk)
  - Text processing (grep/awk equivalent)
  - Process management (ps-like)
  - Pipes in Python (chaining iterables)
  - Writing a Makefile-style task runner

Run:  python working_example2.py
"""
import csv
import fnmatch
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# -- 1. Directory operations (pushd/popd) --------------------------------------
def demo_directory_ops() -> None:
    print("=== Directory Operations ===")
    original = Path.cwd()

    # pushd equivalent
    os.chdir(DATA)
    print(f"  cd data/    ->  {Path.cwd().name}/")

    # Create some test files
    (Path("tmp_a.txt")).write_text("alpha 1\nbeta 2\ngamma 3\n")
    (Path("tmp_b.csv")).write_text("name,value\nalpha,1\nbeta,2\n")

    # popd equivalent
    os.chdir(original)
    print(f"  cd back     ->  {Path.cwd().name}/")


# -- 2. File search (find equivalent) -----------------------------------------
def demo_file_search() -> None:
    print("\n=== File Search (find) ===")
    base = Path(__file__).parent

    # All Python files
    py_files = sorted(base.rglob("*.py"))
    print(f"  *.py files under {base.name}/:")
    for f in py_files[:5]:
        print(f"    {f.relative_to(base)}")

    # fnmatch (shell-style pattern)
    all_files = list(base.iterdir())
    matched = [f for f in all_files if fnmatch.fnmatch(f.name, "working_*")]
    print(f"\n  fnmatch 'working_*': {[f.name for f in matched]}")

    # Find by size (>1 KB)
    big = [f for f in base.rglob("*") if f.is_file() and f.stat().st_size > 1024]
    print(f"  Files > 1KB: {len(big)}")


# -- 3. Text processing (grep/awk equivalent) ----------------------------------
def demo_text_processing() -> None:
    print("\n=== Text Processing (grep/awk) ===")

    # Download a small dataset to grep through
    dest = DATA / "titanic.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv",
                dest
            )
        except Exception:
            rows = ["PassengerId,Survived,Pclass,Name,Sex,Age\n"]
            for i in range(1, 30):
                rows.append(f"{i},{i%2},{(i%3)+1},Person {i},{'male' if i%2 else 'female'},{25+i%20}\n")
            dest.write_text("".join(rows))

    # grep: find lines matching a pattern
    keyword = "female"
    matches = [line.strip() for line in dest.read_text().splitlines() if keyword in line]
    print(f"  Lines with '{keyword}': {len(matches)}")

    # awk-equivalent: extract column 6 (Age) using csv
    with open(dest, newline="") as f:
        reader = csv.DictReader(f)
        ages = [float(r["Age"]) for r in reader if r.get("Age", "").strip()]
    print(f"  Age column: count={len(ages)}, mean={sum(ages)/len(ages):.2f}")

    # wc -l equivalent
    line_count = len(dest.read_text().splitlines())
    print(f"  wc -l {dest.name}: {line_count}")


# -- 4. Pipe pattern (chaining iterables) --------------------------------------
def demo_pipes() -> None:
    """Demonstrate Unix pipe pattern using Python generators."""
    print("\n=== Python Pipe Pattern ===")

    def read_lines(path: Path):
        yield from path.open(encoding="utf-8")

    def grep(lines, pattern: str):
        for line in lines:
            if pattern in line:
                yield line

    def cut_field(lines, field: int, sep: str = ","):
        for line in lines:
            parts = line.strip().split(sep)
            if field < len(parts):
                yield parts[field]

    def head(lines, n: int = 5):
        for i, line in enumerate(lines):
            if i >= n: break
            yield line

    # Pipe: cat titanic.csv | grep female | cut -f6 | head -5
    path = DATA / "titanic.csv"
    if path.exists():
        pipeline = head(cut_field(grep(read_lines(path), "female"), 5), 5)
        print("  cat titanic.csv | grep female | cut -f6 | head -5:")
        for val in pipeline:
            print(f"    {val!r}")


# -- 5. Task runner (Makefile-style) -------------------------------------------
class Task:
    """Simple Makefile-like task with dependency tracking."""
    registry: dict[str, "Task"] = {}

    def __init__(self, name: str, deps: list[str] = None):
        self.name = name
        self.deps = deps or []
        self.done = False
        Task.registry[name] = self

    def run(self, fn):
        """Decorator to attach a function to this task."""
        self._fn = fn
        return fn

    def execute(self, verbose: bool = False) -> None:
        if self.done: return
        for dep in self.deps:
            Task.registry[dep].execute(verbose)
        t0 = time.time()
        self._fn()
        dt = time.time() - t0
        self.done = True
        if verbose:
            print(f"    [{self.name}] done in {dt:.3f}s")


def demo_task_runner() -> None:
    print("\n=== Task Runner (Makefile-style) ===")
    task_download = Task("download")
    task_clean    = Task("clean", deps=["download"])
    task_report   = Task("report", deps=["clean"])

    @task_download.run
    def _():
        path = DATA / "titanic.csv"
        if not path.exists():
            try: urllib.request.urlretrieve(
                "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv", path)
            except Exception: path.write_text("PassengerId,Survived\n1,1\n2,0\n")
        print(f"  [download] {path.name} ready")

    @task_clean.run
    def _():
        print(f"  [clean] data cleaned")

    @task_report.run
    def _():
        print(f"  [report] report generated")

    task_report.execute(verbose=True)


# -- 6. Process info -----------------------------------------------------------
def demo_process_info() -> None:
    print("\n=== Process Info ===")
    print(f"  PID: {os.getpid()}")
    print(f"  Parent PID: {os.getppid()}")
    print(f"  Python exe: {sys.executable}")
    # List env vars (safe subset)
    safe_keys = ["PATH", "HOME", "USER", "COMPUTERNAME", "OS"]
    for k in safe_keys:
        val = os.environ.get(k, "")
        print(f"  {k}: {val[:60]}")


if __name__ == "__main__":
    demo_directory_ops()
    demo_file_search()
    demo_text_processing()
    demo_pipes()
    demo_task_runner()
    demo_process_info()
