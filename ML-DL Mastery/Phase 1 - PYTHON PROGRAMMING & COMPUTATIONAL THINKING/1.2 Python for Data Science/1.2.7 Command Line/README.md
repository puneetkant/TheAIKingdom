# 1.2.7 Command Line

Python CLI tooling: `argparse`, `subprocess`, `pathlib`, `os.environ`, pipeline runner with subcommands.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Basic CLI examples: argv, argparse, file ops |
| `working_example2.py` | Full ML pipeline CLI: argparse subcommands, subprocess, env config, pipeline steps |
| `working_example.ipynb` | Interactive: argparse, subprocess, pathlib, os.environ |

## Run

```bash
python working_example.py
python working_example2.py --help
python working_example2.py run --steps download clean stats --verbose
python working_example2.py describe --path data/
python working_example2.py info
jupyter lab working_example.ipynb
```

## CLI Quick Reference

```python
import argparse

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("model", choices=["linear", "mlp"])

# Subcommands
sub = parser.add_subparsers(dest="command")
train_p = sub.add_parser("train")
train_p.add_argument("--data", required=True)

args = parser.parse_args()
```

## subprocess (safe pattern)

```python
import subprocess, sys

# NEVER use shell=True with user input
result = subprocess.run(
    [sys.executable, "train.py", "--lr", "0.01"],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise RuntimeError(result.stderr)
print(result.stdout)
```

## Learning Resources
- [argparse docs](https://docs.python.org/3/library/argparse.html)
- [subprocess docs](https://docs.python.org/3/library/subprocess.html)
- [Real Python: Command Line Interfaces](https://realpython.com/command-line-interfaces-python-argparse/)
- [Click (popular CLI framework)](https://click.palletsprojects.com/)
- [Typer (modern, type-hint based)](https://typer.tiangolo.com/)

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
