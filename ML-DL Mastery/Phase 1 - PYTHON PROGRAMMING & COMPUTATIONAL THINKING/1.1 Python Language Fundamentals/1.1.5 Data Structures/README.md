# 1.1.5 Data Structures

Python built-in structures applied to MovieLens ratings — lists, dicts, sets, Counter, deque, namedtuple.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | List/dict/set/tuple fundamentals, comprehensions, sorting |
| `working_example2.py` | MovieLens data: list as stack/queue, defaultdict, set genre analysis, Counter |
| `working_example.ipynb` | Interactive: download ratings + movies, stats, genre sets, bar chart |

## Run

```bash
python working_example.py
python working_example2.py        # downloads ratings.csv + movies.csv
jupyter lab working_example.ipynb
```

## Cheat Sheet

| Structure | Creation | Ordered | Mutable | Unique |
|-----------|----------|---------|---------|--------|
| `list` | `[1,2,3]` | ✓ | ✓ | ✗ |
| `tuple` | `(1,2,3)` | ✓ | ✗ | ✗ |
| `dict` | `{"a":1}` | ✓ (3.7+) | ✓ | keys |
| `set` | `{1,2,3}` | ✗ | ✓ | ✓ |
| `frozenset` | `frozenset([1,2])` | ✗ | ✗ | ✓ |

### collections module
```python
from collections import defaultdict, Counter, deque, namedtuple, OrderedDict
Counter("banana")               # Counter({'a':3,'n':2,'b':1})
defaultdict(list)               # auto-creates missing keys
deque(maxlen=5)                 # O(1) append/pop both ends
namedtuple('Point', 'x y')      # lightweight typed tuple
```

## Datasets
- **MovieLens Small** — [Shahrukh0/MovieLens-Small on HuggingFace](https://huggingface.co/datasets/Shahrukh0/MovieLens-Small) — 100k ratings, 9k movies

## Learning Resources
- [Python data structures docs](https://docs.python.org/3/tutorial/datastructures.html)
- [collections module](https://docs.python.org/3/library/collections.html)
- [Real Python: Dictionaries](https://realpython.com/python-dicts/)
- [Real Python: Sets](https://realpython.com/python-sets/)
- **Book:** *Python Crash Course* Ch. 3, 5 (lists, dicts)
- **Book:** *Fluent Python* Ch. 2-3 (sequences, dicts)
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
