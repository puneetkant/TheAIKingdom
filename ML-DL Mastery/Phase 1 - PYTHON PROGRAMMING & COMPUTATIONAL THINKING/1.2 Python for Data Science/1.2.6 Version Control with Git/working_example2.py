"""
Working Example 2: Git Automation with subprocess
==================================================
Demonstrates calling Git from Python:
  - Check git version and current repo status
  - Create a temporary throwaway repo, make commits, inspect log
  - Parse `git log --oneline` to build a commit history
  - Show diff, branch list, stash usage
  - ML workflow: version a model artefact with git notes

Run:  python working_example2.py
Note: Requires git to be installed and accessible on PATH.
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


# -- Helpers --------------------------------------------------------------------
def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return stdout; raise on error."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True, text=True, cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)!r} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def git_safe(*args: str, cwd: Path | None = None) -> str:
    """Run a git command, return stdout or error message (never raises)."""
    try:
        return git(*args, cwd=cwd)
    except RuntimeError as e:
        return f"(error: {e})"


# -- 1. System check ------------------------------------------------------------
def check_git() -> bool:
    try:
        version = git("--version")
        print(f"Git available: {version}")
        return True
    except FileNotFoundError:
        print("Git not found — install git to run this demo.")
        return False


# -- 2. Temporary repo setup ----------------------------------------------------
def setup_temp_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="ml_git_demo_"))
    git("init", cwd=repo)
    git("config", "user.email", "demo@example.com", cwd=repo)
    git("config", "user.name",  "ML Demo Bot",       cwd=repo)
    print(f"\nTemp repo: {repo}")
    return repo


# -- 3. Simulate ML project commits --------------------------------------------
def make_ml_commits(repo: Path) -> None:
    """Create a realistic series of ML project commits."""
    # Commit 1: initial data pipeline
    (repo / "data_pipeline.py").write_text(
        'import csv\ndef load(path): return list(csv.DictReader(open(path)))\n'
    )
    (repo / "README.md").write_text("# ML Project\n\nData pipeline.\n")
    git("add", ".", cwd=repo)
    git("commit", "-m", "feat: initial data pipeline and README", cwd=repo)

    # Commit 2: add model
    (repo / "model.py").write_text(
        'class LinearModel:\n    def __init__(self): self.w = []\n'
        '    def fit(self, X, y): self.w = [0.0] * len(X[0])\n'
    )
    git("add", "model.py", cwd=repo)
    git("commit", "-m", "feat: add LinearModel skeleton", cwd=repo)

    # Commit 3: add training metrics JSON (model artifact)
    metrics = {"accuracy": 0.832, "loss": 0.214, "epoch": 10}
    (repo / "metrics.json").write_text(json.dumps(metrics, indent=2))
    git("add", "metrics.json", cwd=repo)
    git("commit", "-m", "chore: save training metrics v0.1", cwd=repo)

    # Commit 4: bugfix
    (repo / "model.py").write_text(
        'import math\nclass LinearModel:\n    def __init__(self): self.w = []\n'
        '    def fit(self, X, y):\n        n = len(X[0])\n        self.w = [0.0]*n\n'
        '    def predict(self, x): return sum(w*xi for w,xi in zip(self.w, x))\n'
    )
    git("add", "model.py", cwd=repo)
    git("commit", "-m", "fix: add predict method, import math", cwd=repo)

    print(f"  Created 4 commits in {repo.name}")


# -- 4. Inspect repo ------------------------------------------------------------
def inspect_repo(repo: Path) -> None:
    print("\n=== git log --oneline ===")
    log = git("log", "--oneline", cwd=repo)
    print(log)

    print("\n=== Parse commit history ===")
    commits = []
    for line in log.splitlines():
        m = re.match(r"^([0-9a-f]+)\s+(.+)$", line)
        if m:
            commits.append({"sha": m.group(1), "msg": m.group(2)})
    for i, c in enumerate(commits, 1):
        print(f"  {i}. [{c['sha']}] {c['msg']}")

    print("\n=== git show HEAD (short) ===")
    show = git("show", "--stat", "HEAD", cwd=repo)
    for line in show.splitlines()[:8]:
        print(f"  {line}")


# -- 5. Branch + diff ----------------------------------------------------------
def demo_branch_and_diff(repo: Path) -> None:
    print("\n=== Branch: experiment/new-features ===")
    git("checkout", "-b", "experiment/new-features", cwd=repo)

    # Modify model.py
    (repo / "model.py").write_text(
        'import math\nclass LinearModel:\n    def __init__(self, lr=0.01): self.w=[]; self.lr=lr\n'
        '    def fit(self, X, y): self.w=[0.0]*len(X[0])\n'
        '    def predict(self, x): return sum(w*xi for w,xi in zip(self.w,x))\n'
    )

    diff = git("diff", "model.py", cwd=repo)
    print(diff[:400] or "(no diff output)")

    print("\n=== Branches ===")
    print(git("branch", cwd=repo))

    # Return to main
    git("checkout", "-", cwd=repo)


# -- 6. Stash demo -------------------------------------------------------------
def demo_stash(repo: Path) -> None:
    print("\n=== Stash ===")
    (repo / "experiment.py").write_text('# work in progress\nresult = None\n')
    git("add", "experiment.py", cwd=repo)
    git("stash", "push", "-m", "WIP: experiment.py", cwd=repo)
    print("  Stash list:", git("stash", "list", cwd=repo))
    git("stash", "pop", cwd=repo)
    print("  Stash popped — file restored:", (repo / "experiment.py").exists())


# -- 7. Tag a model release ----------------------------------------------------
def demo_tag(repo: Path) -> None:
    print("\n=== Tag a model version ===")
    git("tag", "-a", "v0.1.0", "-m", "Model v0.1.0: baseline LinearModel", cwd=repo)
    print("  Tags:", git("tag", cwd=repo))
    print("  Tag description:", git("describe", "--tags", cwd=repo))


# -- 8. Status + clean up ------------------------------------------------------
def demo_status_and_cleanup(repo: Path) -> None:
    print("\n=== git status ===")
    print(git("status", "--short", cwd=repo))

    # Clean up temp repo
    shutil.rmtree(repo, ignore_errors=True)
    print(f"\n  Cleaned up temp repo.")


if __name__ == "__main__":
    if not check_git():
        raise SystemExit(1)

    repo = setup_temp_repo()
    try:
        make_ml_commits(repo)
        inspect_repo(repo)
        demo_branch_and_diff(repo)
        demo_stash(repo)
        demo_tag(repo)
        demo_status_and_cleanup(repo)
    except Exception as e:
        print(f"Error: {e}")
        shutil.rmtree(repo, ignore_errors=True)
