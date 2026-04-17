# 5.8.1 ML System Design

Data pipelines, training loops, validation schemas, serving interfaces, latency vs throughput.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | ML system architecture patterns |
| `working_example2.py` | Full skeleton: validate → preprocess → train → serve |
| `working_example.ipynb` | Interactive: normaliser + validation |

## Quick Reference

```python
# Pipeline components
class Pipeline:
    def fit(self, X): ...
    def transform(self, X): ...

# Data validation
def validate(X, y, schema):
    assert X.shape[1] == schema['n_features']
    assert not np.isnan(X).any()

# Serving layer
class ModelServer:
    def predict(self, raw_input):
        X = preprocess(raw_input)
        return model.predict(X)

# Offline vs online serving
# Batch: process large datasets periodically
# Online: low-latency per-request prediction (< 100ms SLA)
```

## ML System Layers

| Layer | Component | Tool |
|-------|-----------|------|
| Data | Ingestion, validation | Great Expectations, TFDV |
| Training | Experiment runs | MLflow, W&B |
| Packaging | Docker, ONNX | Docker, torch.export |
| Serving | REST / gRPC | FastAPI, TF Serving, Triton |
| Monitoring | Drift, latency | Evidently, Prometheus |

## Learning Resources
- [Google ML Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)
- [Made With ML](https://madewithml.com/)

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
