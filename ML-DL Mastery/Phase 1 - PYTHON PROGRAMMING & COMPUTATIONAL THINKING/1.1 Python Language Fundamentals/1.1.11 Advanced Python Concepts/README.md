# 1.1.11 Advanced Python Concepts

Metaclasses, descriptors, `__slots__`, generator `send()`, dataclasses, itertools — the internals powering ML frameworks like scikit-learn and HuggingFace.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Iterators, generators, comprehensions, context managers, decorators |
| `working_example2.py` | Metaclass registry, Descriptors, __slots__, generator coroutine, dataclass __post_init__, itertools grid search |
| `working_example.ipynb` | Interactive: all advanced patterns with ML-focused examples |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Advanced Patterns Reference

```python
# Metaclass registry (like AutoModel)
class Registry(type):
    _reg = {}
    def __new__(mcs, name, bases, ns, **kw):
        cls = super().__new__(mcs, name, bases, ns)
        if bases: mcs._reg[kw.get('tag', name)] = cls
        return cls

# Descriptor for validated attribute
class BoundedFloat:
    def __set_name__(self, owner, name): self.name = name
    def __set__(self, obj, v):
        if not (self.lo <= v <= self.hi): raise ValueError(...)
        obj.__dict__[self.name] = v

# __slots__
class Sample:
    __slots__ = ('feature', 'label', 'weight')

# Generator coroutine (send)
def running_mean():
    total=0; count=0; x = yield 0.0
    while True:
        total+=x; count+=1; x = yield total/count
gen = running_mean(); next(gen)
mean = gen.send(0.85)

# itertools for ML
import itertools
grid = list(itertools.product([0.1, 0.01], [32, 64]))  # hyperparameter grid
```

## Learning Resources
- [Python data model](https://docs.python.org/3/reference/datamodel.html)
- [Descriptors HowTo](https://docs.python.org/3/howto/descriptor.html)
- [itertools docs](https://docs.python.org/3/library/itertools.html)
- [dataclasses docs](https://docs.python.org/3/library/dataclasses.html)
- [Real Python: Metaclasses](https://realpython.com/python-metaclasses/)
- [Real Python: __slots__](https://realpython.com/python-slots/)
- **Book:** *Fluent Python* Ch. 14 (iterables), Ch. 21 (class metaprogramming)
- **Book:** *Python Cookbook* Ch. 8 (classes), Ch. 9 (metaprogramming)

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
