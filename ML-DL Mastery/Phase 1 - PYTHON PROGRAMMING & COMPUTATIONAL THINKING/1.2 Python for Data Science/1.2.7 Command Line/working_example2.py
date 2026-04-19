"""
Working Example 2: Command Line Automation in Python
====================================================
Demonstrates argparse CLI scripts, subprocess system calls, and os/pathlib
file-system operations — the building blocks of ML pipeline CLIs.

Sections:
  1. argparse — build a proper CLI with flags and subcommands
  2. subprocess — call system tools (git, python, find/dir)
  3. shutil + pathlib — file management from scripts
  4. sys.argv parsing (manual, educational)
  5. Environment variables and dotenv-style config
  6. A mini pipeline runner that chains steps

Run:
  python working_example2.py --help
  python working_example2.py describe --path data/
  python working_example2.py run --steps download clean stats
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# -- 1. Utility: run shell command and stream output ---------------------------
def sh(cmd: list[str], cwd: Path | None = None) -> str:
    """Safe subprocess call — never passes shell=True."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.stdout.strip() if result.returncode == 0 else f"(error: {result.stderr.strip()})"


# -- 2. Environment config (dotenv-style) --------------------------------------
def load_env_config(path: Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines from a .env file (stdlib only)."""
    config: dict[str, str] = {}
    env_path = path or (Path(__file__).parent / ".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            config[key.strip()] = val.strip().strip('"').strip("'")
    # Also read from real environment (os.environ overrides .env)
    for k, v in os.environ.items():
        config[k] = v
    return config


# -- 3. Pipeline steps ---------------------------------------------------------
def step_download() -> None:
    import urllib.request
    dest = DATA / "titanic.csv"
    if dest.exists():
        print(f"  [download] already exists: {dest.name}")
        return
    try:
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv",
            dest
        )
        print(f"  [download] {dest.name} ({dest.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"  [download] WARNING: {e}")
        dest.write_text("PassengerId,Survived,Pclass,Name,Sex,Age\n1,1,1,Test,female,29\n")


def step_clean() -> None:
    import csv
    src = DATA / "titanic.csv"
    if not src.exists():
        print("  [clean] no file to clean — run download first"); return
    rows = []
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("Age", "").strip():  # drop rows with missing age
                rows.append(r)
    dest = DATA / "titanic_clean.csv"
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  [clean] {len(rows)} rows -> {dest.name}")


def step_stats() -> None:
    import csv, statistics
    src = DATA / "titanic_clean.csv"
    if not src.exists():
        src = DATA / "titanic.csv"
    if not src.exists():
        print("  [stats] no data available"); return
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ages = [float(r["Age"]) for r in rows if r.get("Age", "").strip()]
    surv = sum(int(r["Survived"]) for r in rows if r.get("Survived", "").strip())
    stats = {
        "n_rows": len(rows),
        "survival_rate": round(surv / max(len(rows), 1), 4),
        "age_mean": round(statistics.mean(ages), 2) if ages else None,
        "age_std":  round(statistics.stdev(ages), 2) if len(ages) > 1 else None,
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  [stats] {stats}")


STEPS = {"download": step_download, "clean": step_clean, "stats": step_stats}


# -- 4. describe: list directory ------------------------------------------------
def cmd_describe(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"Path not found: {path}"); return
    print(f"\nDirectory: {p.resolve()}")
    print(f"  {'Name':<30} {'Size':>10} {'Modified'}")
    print("  " + "-" * 60)
    for item in sorted(p.iterdir()):
        if item.is_file():
            size = f"{item.stat().st_size:,}"
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(item.stat().st_mtime))
            print(f"  {item.name:<30} {size:>10}  {mtime}")
        else:
            print(f"  {item.name:<30} {'<dir>':>10}")


# -- 5. sys info ----------------------------------------------------------------
def cmd_sysinfo() -> None:
    print("\n=== System Info ===")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  CWD: {Path.cwd()}")
    print(f"  Git: {sh(['git', '--version'])}")
    print(f"  PATH entries: {len(os.environ.get('PATH', '').split(os.pathsep))}")

    # Available Python packages (top-level)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=columns"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().splitlines()
    print(f"  Installed packages: {max(0, len(lines) - 2)}")


# -- 6. argparse CLI entry point -----------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="working_example2",
        description="ML Pipeline CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    sub = parser.add_subparsers(dest="command", required=False)

    # run subcommand
    run_p = sub.add_parser("run", help="Run pipeline steps")
    run_p.add_argument(
        "--steps", nargs="+",
        choices=list(STEPS), default=list(STEPS),
        help=f"Steps to run (default: all)"
    )

    # describe subcommand
    desc_p = sub.add_parser("describe", help="Describe a directory")
    desc_p.add_argument("--path", default=str(DATA), help="Path to describe")

    # info subcommand
    sub.add_parser("info", help="Print system info")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        steps = getattr(args, "steps", list(STEPS))
        print(f"=== Running pipeline steps: {steps} ===")
        for step in steps:
            t0 = time.time()
            STEPS[step]()
            dt = time.time() - t0
            if getattr(args, "verbose", False):
                print(f"    time: {dt:.3f}s")
    elif args.command == "describe":
        cmd_describe(args.path)
    elif args.command == "info":
        cmd_sysinfo()
