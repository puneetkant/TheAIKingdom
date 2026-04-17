# 1.2.6 Version Control with Git

Calling Git from Python via `subprocess`: init, commit, log parsing, branch, diff, stash, tag.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Core git commands explained with subprocess calls |
| `working_example2.py` | Full ML project simulation: commits → log parse → branch → stash → version tag |
| `working_example.ipynb` | Interactive: create temp repo, branch, diff, stash, tag, clean up |

## Run

```bash
python working_example.py
python working_example2.py   # requires git on PATH
jupyter lab working_example.ipynb
```

## Git Quick Reference

```bash
# Setup
git init
git config --global user.name "Name"
git config --global user.email "email@example.com"

# Daily workflow
git status                   # what changed?
git add <file>               # stage
git commit -m "feat: msg"    # commit
git log --oneline --graph    # history

# Branches
git checkout -b feature/x   # new branch
git merge feature/x          # merge back
git branch -d feature/x      # delete

# Stash
git stash push -m "WIP"
git stash pop

# Tags (for model versions)
git tag -a v1.0 -m "Model v1.0"
git describe --tags

# Undo
git restore <file>           # discard working changes
git revert HEAD              # safe undo with new commit
```

## From Python

```python
import subprocess

def git(*args, cwd=None):
    r = subprocess.run(['git', *args], capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    return r.stdout.strip()

log = git('log', '--oneline', cwd='/path/to/repo')
```

## Learning Resources
- [Git official docs](https://git-scm.com/doc)
- [Oh My Git! (interactive game)](https://ohmygit.org/)
- [Learn Git Branching](https://learngitbranching.js.org/)
- [Pro Git book (free)](https://git-scm.com/book/en/v2)
- [Conventional Commits spec](https://www.conventionalcommits.org/)
- **DVC**: [Data Version Control for ML](https://dvc.org/)

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
