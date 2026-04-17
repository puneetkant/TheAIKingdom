# 1.1.6 Functions

Decorators, generators, closures, partial functions — the functional programming toolkit for ML engineers.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Args/kwargs, *args/**kwargs, lambdas, higher-order, recursion |
| `working_example2.py` | Timer/cache decorators, CSV generators, closure metric trackers, partial + composition |
| `working_example.ipynb` | Interactive: timer decorator, lru_cache fibonacci, batch generator, stateful tracker |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Key Patterns

```python
# Decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'{func.__name__}: {time.perf_counter()-t0:.3f}s')
        return result
    return wrapper

# Generator for streaming data
def csv_stream(path, chunk_size=100):
    with open(path) as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == chunk_size:
                yield chunk; chunk = []
        if chunk: yield chunk

# Closure
def make_tracker():
    history = []
    def track(val):
        history.append(val)
        return min(history)
    return track

# Partial
norm = functools.partial(lambda x, lo, hi: (x-lo)/(hi-lo), lo=0, hi=100)
```

## Learning Resources
- [Python functions docs](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [functools module](https://docs.python.org/3/library/functools.html)
- [Real Python: Decorators](https://realpython.com/primer-on-python-decorators/)
- [Real Python: Generators](https://realpython.com/introduction-to-python-generators/)
- **Book:** *Fluent Python* Ch. 7 (functions as objects), Ch. 9 (decorators)
- **Book:** *Python Cookbook* (O'Reilly) — Ch. 7 (functions and closures)
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
