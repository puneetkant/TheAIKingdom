# 1.1.2 Basic Syntax & Data Types

Python's type system, string operations, and modern syntax — applied to real Titanic passenger data.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Built-in types, numeric, string, bool/None, type conversion, introspection |
| `working_example2.py` | Titanic CSV from HuggingFace, type coercion, Decimal, dataclasses, walrus, pattern matching |
| `working_example.ipynb` | Interactive notebook: download, pandas EDA, visualisation, pattern matching |

## Run

```bash
python working_example.py
python working_example2.py        # downloads data/titanic.csv
jupyter lab working_example.ipynb
```

## Key Concepts

| Type | Literal | Notes |
|------|---------|-------|
| `int` | `42` | Unlimited precision |
| `float` | `3.14` | IEEE 754 double |
| `complex` | `2+3j` | Real + imaginary |
| `Decimal` | `Decimal('0.1')` | Exact decimal arithmetic |
| `str` | `"hello"` | Immutable, Unicode |
| `bool` | `True`/`False` | Subclass of int |
| `None` | `None` | Singleton null value |

### String Formatting
```python
f"Score: {score:.1%}"          # f-string (preferred)
"Score: {:.1%}".format(score)  # .format()
"Score: %.1f%%" % (score*100)  # %-style (legacy)
```

### Modern Syntax (3.8-3.12)
```python
# Walrus operator (3.8+)
if (n := len(data)) > 10: print(f"Large: {n}")

# Pattern matching (3.10+)
match command:
    case "quit": quit()
    case "help": show_help()
    case _: print("Unknown")

# Union types (3.10+)
def fn(x: int | None) -> str: ...
```

## Datasets
- **Titanic** — [phihung/titanic on HuggingFace](https://huggingface.co/datasets/phihung/titanic) — 891 passengers, 12 columns

## Learning Resources
- [Python Built-in Types (official docs)](https://docs.python.org/3/library/stdtypes.html)
- [PEP 498 — f-strings](https://peps.python.org/pep-0498/)
- [PEP 634 — Structural pattern matching](https://peps.python.org/pep-0634/)
- [Real Python: Python Data Types](https://realpython.com/python-data-types/)
- [Python Decimal module](https://docs.python.org/3/library/decimal.html)
- **Book:** *Python Crash Course* — Ch. 2-4 (variables, strings, numbers)
- **Book:** *Fluent Python* (2nd ed.) — Ch. 1-2 (data model, sequences)
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
