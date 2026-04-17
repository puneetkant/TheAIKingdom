# 1.1.4 Control Flow

if/elif/else, loops, comprehensions — applied to Titanic data and ML training loop simulation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | if/elif, ternary, for/while, break/continue, enumerate, zip, match-case |
| `working_example2.py` | Titanic data validation pipeline, ML training loop with early stopping, survival matrix |
| `working_example.ipynb` | Interactive: Titanic EDA, training curve plot, comprehension patterns |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Key Patterns

```python
# Early stopping (common in ML training)
for epoch in range(max_epochs):
    val_loss = train_one_epoch()
    if val_loss < best - delta:
        best, patience_count = val_loss, 0
    else:
        patience_count += 1
    if patience_count >= patience:
        break

# Comprehension with filter
valid = [x for x in data if 0 < x < 120]

# Walrus + while for chunked processing
while (chunk := data[idx:idx+chunk_size]):
    process(chunk); idx += chunk_size
```

## Datasets
- **Titanic** — [phihung/titanic on HuggingFace](https://huggingface.co/datasets/phihung/titanic)

## Learning Resources
- [Python control flow docs](https://docs.python.org/3/tutorial/controlflow.html)
- [Real Python: for loops](https://realpython.com/python-for-loop/)
- [Real Python: List comprehensions](https://realpython.com/list-comprehension-python/)
- **Book:** *Python Crash Course* Ch. 5-7 (if statements, loops)
- **Book:** *Fluent Python* Ch. 17 (iterators, generators)
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
