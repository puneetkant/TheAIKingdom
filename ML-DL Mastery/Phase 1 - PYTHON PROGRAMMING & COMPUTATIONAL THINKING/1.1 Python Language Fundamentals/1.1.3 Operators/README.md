# 1.1.3 Operators

Python operators — from arithmetic and bitwise to operator overloading — applied to real data science problems.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | All operator categories, precedence table, identity/membership |
| `working_example2.py` | Vector class with `__dunder__` overloading, bitwise flags, California Housing feature engineering |
| `working_example.ipynb` | Interactive: download Cal Housing, feature engineering, plots |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Operator Quick Reference

| Category | Operators | Example |
|----------|-----------|---------|
| Arithmetic | `+ - * / // % **` | `17 // 5 = 3` |
| Comparison | `== != < > <= >=` | `3 < x <= 10` |
| Logical | `and or not` | `a > 0 and b > 0` |
| Bitwise | `& | ^ ~ << >>` | `flags & MASK` |
| Assignment | `+= -= *= /= //= **= &=` | `x += 1` |
| Identity | `is  is not` | `x is None` |
| Membership | `in  not in` | `"a" in "abc"` |
| Walrus | `:=` | `if (n := len(a)) > 10` |

### Operator Overloading
```python
class Vector:
    def __add__(self, other): return Vector(self.x+other.x, self.y+other.y)
    def __abs__(self): return sqrt(self.x**2 + self.y**2)
    def __mul__(self, scalar): return Vector(self.x*scalar, self.y*scalar)
    def __rmul__(self, scalar): return self.__mul__(scalar)
```

### Chained Comparisons (Pythonic)
```python
0 <= x < 100          # True if 0 <= x AND x < 100
a < b < c < d         # reads naturally, no 'and' needed
```

## Datasets
- **California Housing** — [scikit-learn/california-housing on HuggingFace](https://huggingface.co/datasets/scikit-learn/california-housing) — 20,640 rows, 9 features

## Learning Resources
- [Python Operator docs](https://docs.python.org/3/library/operator.html)
- [Python Data Model — emulating numeric types](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)
- [Real Python: Operator overloading](https://realpython.com/operator-function-overloading/)
- [PEP 572 — Walrus operator](https://peps.python.org/pep-0572/)
- **Book:** *Fluent Python* Ch. 16 — operator overloading
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
