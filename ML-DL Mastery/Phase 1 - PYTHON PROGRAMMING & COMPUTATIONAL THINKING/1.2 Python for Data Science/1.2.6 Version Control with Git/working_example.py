"""
Working Example: Version Control with Git
Covers the most important Git concepts, commands, and workflows.
All examples run via subprocess so you can see real output.
A throwaway temp repo is created and cleaned up automatically.
"""
import subprocess
import tempfile
import shutil
import os
from pathlib import Path


def run(cmd, cwd=None, check=True):
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, text=True,
        capture_output=True, cwd=cwd
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return (result.stdout + result.stderr).strip()


def setup_git_identity():
    """Ensure Git has a usable identity (may already be global-configured)."""
    name  = run("git config --global user.name", check=False) or "Learner"
    email = run("git config --global user.email", check=False) or "learner@example.com"
    return name, email


# -- 1. Repository initialisation ---------------------------------------------
def init_demo(repo):
    print("=== 1. git init ===")
    print(run("git init", cwd=repo))
    # Ensure a consistent default branch name
    run("git config init.defaultBranch main", cwd=repo)
    run("git symbolic-ref HEAD refs/heads/main", cwd=repo)
    setup_git_identity()
    run('git config user.name "Learner"', cwd=repo)
    run('git config user.email "learner@example.com"', cwd=repo)
    print(f"  Repo at: {repo}")


# -- 2. Staging and committing -------------------------------------------------
def staging_demo(repo):
    print("\n=== 2. git add / commit ===")
    (Path(repo) / "README.md").write_text("# My Project\nWelcome!\n")
    (Path(repo) / "app.py").write_text('print("hello")\n')

    print(run("git status", cwd=repo))
    run("git add README.md", cwd=repo)
    print(run("git status --short", cwd=repo))
    run("git add .", cwd=repo)
    print(run('git commit -m "Initial commit: add README and app"', cwd=repo))
    print(run("git log --oneline", cwd=repo))


# -- 3. .gitignore -------------------------------------------------------------
def gitignore_demo(repo):
    print("\n=== 3. .gitignore ===")
    gitignore = Path(repo) / ".gitignore"
    gitignore.write_text("__pycache__/\n*.pyc\n.env\nvenv/\n*.log\n")
    (Path(repo) / "secret.log").write_text("token=abc123\n")
    (Path(repo) / "app.log").write_text("INFO started\n")

    run("git add .gitignore", cwd=repo)
    run('git commit -m "Add .gitignore"', cwd=repo)
    status = run("git status --short", cwd=repo)
    print(f"  status after adding logs (should be empty): {status!r}")


# -- 4. Branching and merging --------------------------------------------------
def branching_demo(repo):
    print("\n=== 4. Branching & Merging ===")
    print(run("git branch", cwd=repo))
    run("git checkout -b feature/login", cwd=repo)
    print(f"  switched to: {run('git branch --show-current', cwd=repo)}")

    (Path(repo) / "login.py").write_text('def login(user): return True\n')
    run("git add login.py", cwd=repo)
    run('git commit -m "feat: add login module"', cwd=repo)

    run("git checkout main", cwd=repo)
    print(f"  back on: {run('git branch --show-current', cwd=repo)}")
    print(run("git merge feature/login --no-ff -m 'Merge feature/login'", cwd=repo))
    print(run("git log --oneline --graph --all", cwd=repo))


# -- 5. Viewing history and diffs ----------------------------------------------
def history_demo(repo):
    print("\n=== 5. log / diff / show ===")
    (Path(repo) / "app.py").write_text('print("hello world")\n# updated\n')
    run("git add app.py", cwd=repo)
    run('git commit -m "chore: update app.py"', cwd=repo)

    print(run("git log --oneline --decorate --all", cwd=repo))
    print("\n  Last diff (HEAD~1..HEAD):")
    print(run("git diff HEAD~1 HEAD -- app.py", cwd=repo))


# -- 6. Undoing changes --------------------------------------------------------
def undo_demo(repo):
    print("\n=== 6. Undoing Changes ===")

    # -- restore unstaged change
    (Path(repo) / "app.py").write_text("oops!\n")
    print(run("git restore app.py", cwd=repo))
    print(f"  app.py after restore: {(Path(repo)/'app.py').read_text().strip()!r}")

    # -- revert a commit (safe for shared history)
    print(run("git revert HEAD --no-edit", cwd=repo))
    print(run("git log --oneline --all", cwd=repo))


# -- 7. Stashing ---------------------------------------------------------------
def stash_demo(repo):
    print("\n=== 7. git stash ===")
    (Path(repo) / "wip.py").write_text("# work in progress\n")
    run("git add wip.py", cwd=repo)
    run("git stash push -m 'WIP: new feature'", cwd=repo)
    print(run("git stash list", cwd=repo))
    print(run("git stash pop", cwd=repo))
    print(f"  wip.py restored: {(Path(repo)/'wip.py').exists()}")


# -- 8. Tagging ----------------------------------------------------------------
def tag_demo(repo):
    print("\n=== 8. git tag ===")
    run("git tag v1.0.0 -m 'Version 1.0.0'", cwd=repo)
    run("git tag v1.1.0-beta", cwd=repo)
    print(run("git tag", cwd=repo))
    print(run("git show v1.0.0 --stat", cwd=repo))


# -- 9. Command reference -----------------------------------------------------
def command_reference():
    print("\n=== 9. Quick Command Reference ===")
    commands = [
        ("git init",                       "Initialise a new repo"),
        ("git clone <url>",                "Clone a remote repo"),
        ("git status",                     "Working tree status"),
        ("git add <file>",                 "Stage file(s)"),
        ("git commit -m 'msg'",            "Commit staged changes"),
        ("git log --oneline --graph",      "Visual history"),
        ("git diff",                       "Unstaged changes"),
        ("git branch <name>",              "Create branch"),
        ("git checkout / switch <branch>", "Switch branch"),
        ("git merge <branch>",             "Merge branch into current"),
        ("git rebase <branch>",            "Rebase current on top of branch"),
        ("git stash / stash pop",          "Save / restore dirty work"),
        ("git reset HEAD~1 --soft",        "Undo last commit, keep changes"),
        ("git revert <sha>",               "Safe undo (new commit)"),
        ("git remote add origin <url>",    "Link to remote"),
        ("git push / pull",                "Sync with remote"),
        ("git fetch",                      "Download without merging"),
        ("git tag -a v1.0 -m 'msg'",       "Create annotated tag"),
        ("git cherry-pick <sha>",          "Apply single commit"),
        ("git bisect start/good/bad",      "Binary search for a bug"),
    ]
    for cmd, desc in commands:
        print(f"  {cmd:<42} # {desc}")


if __name__ == "__main__":
    repo = tempfile.mkdtemp(prefix="git_demo_")
    try:
        init_demo(repo)
        staging_demo(repo)
        gitignore_demo(repo)
        branching_demo(repo)
        history_demo(repo)
        undo_demo(repo)
        stash_demo(repo)
        tag_demo(repo)
        command_reference()
    finally:
        shutil.rmtree(repo, ignore_errors=True)
        print(f"\n  Temp repo cleaned up.")
