# 1.1.10 Object-Oriented Programming (OOP)

Build an sklearn-style estimator hierarchy — abstract base, KNN, Naive Bayes, Pipeline, Mixins.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Classes, inheritance, dunder methods, properties |
| `working_example2.py` | BaseEstimator ABC, KNNClassifier, GaussianNaiveBayes, Pipeline (composition), Mixins, pickle serialization |
| `working_example.ipynb` | Interactive: build KNN from scratch, train on Iris, MRO exploration |

## Run

```bash
python working_example.py
python working_example2.py    # downloads iris.csv
jupyter lab working_example.ipynb
```

## Key OOP Patterns

```python
# Abstract Base Class
from abc import ABC, abstractmethod
class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X): ...
    def score(self, X, y):
        return sum(a==b for a,b in zip(self.predict(X), y)) / len(y)
    def __call__(self, X): return self.predict(X)   # callable instance

# Mixin (multiple inheritance)
class ReprMixin:
    def __repr__(self): return f"{type(self).__name__}({self.get_params()})"

class KNN(BaseEstimator, ReprMixin):
    def __init__(self, k=5): self.k = k
    def get_params(self): return {"k": self.k}
    def fit(self, X, y): ...
    def predict(self, X): ...
```

## Dunder Methods Cheat Sheet

| Dunder | Use case |
|--------|----------|
| `__init__` | Constructor |
| `__repr__` | Developer string (used in REPL) |
| `__str__` | User-facing string |
| `__len__` | `len(obj)` |
| `__call__` | `obj(args)` |
| `__iter__` | `for x in obj` |
| `__eq__`, `__lt__` | Comparison / sorting |
| `__add__`, `__mul__` | Arithmetic |
| `__enter__`, `__exit__` | `with` statement |

## Dataset
- **Iris** — [scikit-learn/iris on HuggingFace](https://huggingface.co/datasets/scikit-learn/iris)

## Learning Resources
- [Python classes docs](https://docs.python.org/3/tutorial/classes.html)
- [abc module](https://docs.python.org/3/library/abc.html)
- [Real Python: OOP](https://realpython.com/python3-object-oriented-programming/)
- [Real Python: Dunder methods](https://realpython.com/operator-function-overloading/)
- **Book:** *Fluent Python* Ch. 1, 11-14 (data model, interfaces, inheritance)
- **Book:** *Python Crash Course* Ch. 9 (classes)

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
