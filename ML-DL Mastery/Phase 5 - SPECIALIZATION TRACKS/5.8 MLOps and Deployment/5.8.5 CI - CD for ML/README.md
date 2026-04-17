# 5.8.5 CI/CD for ML

Automated pipelines: linting → unit tests → training → evaluation gate → deployment.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | GitHub Actions / Jenkins setup |
| `working_example2.py` | Full CI/CD simulation: 5 stages with gate |
| `working_example.ipynb` | Interactive: training + eval gate + pipeline viz |

## Quick Reference

```yaml
# GitHub Actions CI for ML
name: ML CI
on: [push]
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Lint
        run: ruff check src/
      - name: Test
        run: pytest tests/ -v
      - name: Train
        run: python train.py --output model.pkl
      - name: Evaluate gate
        run: python evaluate.py --threshold 0.85 --model model.pkl
      - name: Deploy
        if: success()
        run: python deploy.py --model model.pkl
```

## Pipeline Stages

| Stage | Tool | Gate |
|-------|------|------|
| Lint | ruff, flake8 | Zero errors |
| Unit test | pytest | All pass |
| Train | Python, DVC | Completes |
| Eval gate | Custom script | acc ≥ threshold |
| Deploy | Docker push, cloud | Image built |

## Learning Resources
- [GitHub Actions for ML](https://docs.github.com/en/actions)
- [DVC pipelines](https://dvc.org/doc/start/data-pipelines)

Explore this topic with a small practical project or coding exercise.

## What to build

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
