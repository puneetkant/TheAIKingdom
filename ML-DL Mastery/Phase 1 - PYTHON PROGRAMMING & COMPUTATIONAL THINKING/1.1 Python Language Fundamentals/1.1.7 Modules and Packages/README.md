# 1.1.7 Modules and Packages

Build a reusable `mlutils` package, control exports with `__all__`, introspect with `inspect`, and dynamically load with `importlib`.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | import, from…import, __name__, standard library survey |
| `working_example2.py` | Creates mini `mlutils` package, introspection, dynamic imports, stdlib highlights |
| `working_example.ipynb` | Interactive: write package to disk, import, introspect, dynamic load |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Package Structure Pattern

```
mlutils/
├── __init__.py        # exports + __version__
├── metrics.py         # accuracy, f1, rmse
├── preprocessing.py   # normalize, standardize
└── plotting.py        # loss curves, confusion matrix
```

```python
# __init__.py
from .metrics import accuracy, f1
__all__ = ['accuracy', 'f1']
__version__ = '0.1.0'
```

## Key Concepts

```python
# Dynamic import
import importlib
mod = importlib.import_module('mlutils.metrics')
fn  = getattr(mod, 'accuracy')

# Introspection
import inspect
for name, obj in inspect.getmembers(mod, inspect.isfunction):
    print(name, inspect.signature(obj))

# __name__ guard
if __name__ == '__main__':
    ...   # only runs when script is executed directly
```

## Learning Resources
- [Python modules docs](https://docs.python.org/3/tutorial/modules.html)
- [importlib docs](https://docs.python.org/3/library/importlib.html)
- [Real Python: Python modules and packages](https://realpython.com/python-modules-packages/)
- [Real Python: importlib](https://realpython.com/python-import/)
- **Book:** *Fluent Python* Ch. 1 (Python Data Model), App. A (libraries)
- **Book:** *Python Cookbook* Ch. 10 (modules and packages)

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
