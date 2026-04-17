# 5.8.3 Model Serving and Deployment

REST/gRPC APIs, batch inference, ONNX export, TF Serving, Triton, latency vs throughput.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | FastAPI serving template |
| `working_example2.py` | REST-like handler + batch latency benchmark |
| `working_example.ipynb` | Interactive: handler + latency curve |

## Quick Reference

```python
# FastAPI serving
from fastapi import FastAPI
import numpy as np

app = FastAPI()
model = load_model("model.pkl")

@app.post("/predict")
def predict(payload: dict):
    X = np.array(payload["instances"])
    return {"predictions": model.predict(X).tolist()}

# ONNX export (PyTorch)
torch.onnx.export(model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"])

# Load ONNX
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx")
out = sess.run(None, {"input": X.astype(np.float32)})
```

## Deployment Patterns

| Pattern | Latency | Throughput | Use Case |
|---------|---------|-----------|----------|
| Online REST | Low | Medium | Per-request |
| Batch inference | High | High | Offline scoring |
| Streaming | Medium | High | Real-time events |
| Edge | Very low | Low | Device |

## Learning Resources
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [ONNX Runtime](https://onnxruntime.ai/)

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
