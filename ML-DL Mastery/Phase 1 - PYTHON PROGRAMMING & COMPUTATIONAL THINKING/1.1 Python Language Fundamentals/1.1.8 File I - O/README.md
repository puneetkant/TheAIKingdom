# 1.1.8 File I/O

Read, write, and manage files — CSV, JSON, Pickle, binary — with production-grade pathlib patterns.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | open/read/write, text/binary modes, context managers |
| `working_example2.py` | Titanic CSV parse → clean → JSON artifact → pickle → pathlib ops |
| `working_example.ipynb` | Interactive: download → CSV → JSON → pickle → directory listing |

## Run

```bash
python working_example.py
python working_example2.py    # creates data/ and output/ folders
jupyter lab working_example.ipynb
```

## Mode Reference

| Mode | Read | Write | Truncate | Create |
|------|------|-------|----------|--------|
| `r` | ✓ | ✗ | ✗ | ✗ |
| `w` | ✗ | ✓ | ✓ | ✓ |
| `a` | ✗ | ✓ | ✗ | ✓ |
| `r+` | ✓ | ✓ | ✗ | ✗ |
| `rb/wb` | binary | binary | — | — |

## Key Patterns

```python
# CSV
with open('data.csv', newline='') as f:
    rows = list(csv.DictReader(f))

# JSON round-trip
import json
json.dump(obj, open('out.json', 'w'), indent=2)
obj = json.load(open('out.json'))

# Atomic write (safe overwrite)
tmp = Path('file.txt.tmp')
tmp.write_text(content)
tmp.replace(Path('file.txt'))

# pathlib glob
list(Path('data').glob('*.csv'))
```

## Dataset
- **Titanic** — [phihung/titanic on HuggingFace](https://huggingface.co/datasets/phihung/titanic)

## Learning Resources
- [pathlib docs](https://docs.python.org/3/library/pathlib.html)
- [csv module docs](https://docs.python.org/3/library/csv.html)
- [json module docs](https://docs.python.org/3/library/json.html)
- [Real Python: Working with Files](https://realpython.com/working-with-files-in-python/)
- [Real Python: Reading/Writing CSV](https://realpython.com/python-csv/)
- **Book:** *Python for Data Analysis* Ch. 6 (data loading)
- **Book:** *Fluent Python* Ch. 18 (contextlib, context managers)

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
