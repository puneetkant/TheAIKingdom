# 1.1.9 Error and Exception Handling

Custom exception hierarchies, retry decorators, context managers, and `logging` — production patterns for ML pipelines.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | try/except/else/finally, built-in exception types, raise, assert |
| `working_example2.py` | Custom exceptions (PipelineError hierarchy), retry decorator, context manager, logging |
| `working_example.ipynb` | Interactive: logging setup, custom exceptions, retry, Timer context manager |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Exception Hierarchy Pattern

```python
class PipelineError(Exception): pass
class DataValidationError(PipelineError):
    def __init__(self, field, msg):
        super().__init__(f"'{field}': {msg}")
        self.field = field
class ModelTrainingError(PipelineError): pass

# Catch base class to handle all pipeline errors
try:
    ...
except PipelineError as e:
    logger.error(f"Pipeline error: {e}")
```

## Retry Decorator

```python
def retry(max_attempts=3, base_delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts+1):
                try: return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts: raise
                    time.sleep(base_delay * 2**(attempt-1))
        return wrapper
    return decorator
```

## Context Manager Pattern

```python
class PipelineSession:
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type: self._cleanup_on_error()
        return False   # don't suppress exceptions
```

## Learning Resources
- [Python exceptions docs](https://docs.python.org/3/tutorial/errors.html)
- [Built-in exceptions](https://docs.python.org/3/library/exceptions.html)
- [logging module](https://docs.python.org/3/library/logging.html)
- [Real Python: Exception handling](https://realpython.com/python-exceptions/)
- [Real Python: Context managers](https://realpython.com/python-with-statement/)
- **Book:** *Fluent Python* Ch. 18 (with statement), Ch. 24 (exceptions)
- **Book:** *Python Cookbook* Ch. 14 (testing / debugging / exceptions)

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
