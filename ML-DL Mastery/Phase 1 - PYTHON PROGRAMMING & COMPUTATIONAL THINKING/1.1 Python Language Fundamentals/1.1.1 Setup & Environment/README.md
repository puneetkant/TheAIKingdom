# 1.1.1 Setup & Environment

A solid Python environment is the foundation of every ML project. This module covers
virtual environments, package management, GPU detection, and reproducible project setup.

---

## Files in This Folder

| File | Description |
|------|-------------|
| `working_example.py` | Environment introspection, venv guidance, pyproject.toml, uv/conda workflows |
| `working_example2.py` | Real-world bootstrap: auto-install, HuggingFace dataset download, train + artifact logging |
| `working_example.ipynb` | Interactive notebook: setup checks, dataset download, EDA, model training |

---

## How to Run

```bash
# Basic environment checks
python working_example.py

# Full real-world workflow (downloads Iris from HuggingFace Hub)
python working_example2.py

# Interactive notebook
jupyter lab working_example.ipynb
```

---

## Key Concepts Covered

### Virtual Environments
```bash
# venv (built-in)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# conda
conda create -n ml-env python=3.11
conda activate ml-env

# uv (modern, fast)
uv init my-project
uv add numpy pandas scikit-learn
uv run python script.py
```

### Package Management
```bash
pip install numpy pandas scikit-learn torch transformers
pip freeze > requirements.txt
pip install -r requirements.txt

# Check for conflicts
pip check
```

### Modern Project Setup (pyproject.toml)
```toml
[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26", "pandas>=2.0", "scikit-learn>=1.4"]
```

### GPU Detection
```python
import torch
print(torch.cuda.is_available())           # CUDA (NVIDIA)
print(torch.backends.mps.is_available())   # Apple Silicon MPS
```

### Environment Reproducibility
```bash
pip freeze > requirements.txt       # exact pinned versions
uv lock                             # uv.lock (hash-verified)
conda env export > environment.yml  # conda environment
```

---

## Learning Resources

### Official Documentation
- [Python venv docs](https://docs.python.org/3/library/venv.html)
- [pip user guide](https://pip.pypa.io/en/stable/user_guide/)
- [conda docs](https://docs.conda.io/projects/conda/en/stable/)
- [uv documentation](https://docs.astral.sh/uv/)
- [pyproject.toml spec (PEP 517/518)](https://peps.python.org/pep-0518/)

### Courses & Tutorials
- **fast.ai setup guide** — https://course.fast.ai/ (highly practical ML setup)
- **Real Python: Virtual Environments** — https://realpython.com/python-virtual-environments-a-primer/
- **Full Stack Deep Learning** — https://fullstackdeeplearning.com/ (MLOps + env setup)
- **CS229 (Stanford) Setup** — https://cs229.stanford.edu/

### Books
- *Python for Data Analysis* — Wes McKinney (O'Reilly) — Ch. 1: environment setup
- *Fluent Python* — Luciano Ramalho — environment and tooling chapters
- *Python Crash Course* — Eric Matthes — beginner-friendly setup

### Tools
| Tool | Purpose | Link |
|------|---------|-------|
| `uv` | Fast package + project manager | https://docs.astral.sh/uv/ |
| `ruff` | Fast linter + formatter | https://docs.astral.sh/ruff/ |
| `pyenv` | Manage multiple Python versions | https://github.com/pyenv/pyenv |
| `direnv` | Per-directory env vars | https://direnv.net/ |
| `pre-commit` | Git hooks for code quality | https://pre-commit.com/ |

### Datasets Used
- **Iris** — [scikit-learn/iris on Hugging Face Hub](https://huggingface.co/datasets/scikit-learn/iris)
  - Classic 150-sample classification dataset (3 species, 4 features)
  - Downloaded in `working_example2.py` via direct URL

---

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `pip: command not found` | `python -m pip install --upgrade pip` |
| Wrong Python version runs | Use `python3` or activate venv first |
| CUDA not detected | Reinstall PyTorch with correct CUDA version from pytorch.org |
| Package conflict | `pip check` then resolve version pins |
| Slow pip installs | Use `uv` — 10-100x faster |
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
