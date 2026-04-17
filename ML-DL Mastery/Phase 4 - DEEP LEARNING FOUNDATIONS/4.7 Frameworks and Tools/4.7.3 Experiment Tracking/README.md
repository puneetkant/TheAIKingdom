# 4.7.3 Experiment Tracking

Log hyperparameters, metrics, and artifacts. Compare runs. MLflow / Weights & Biases patterns.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | MLflow run logging with `mlflow.start_run()` (real MLflow) |
| `working_example2.py` | Stdlib-only sweep: Ridge alpha sweep → JSON run logs → plot |
| `working_example.ipynb` | Interactive: param sweep → compare test MSE across runs |

## Quick Reference

```python
import mlflow, mlflow.sklearn

mlflow.set_experiment("housing-ridge")
for alpha in [0.001, 0.01, 0.1, 1.0]:
    with mlflow.start_run():
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        mse   = mean_squared_error(y_test, model.predict(X_test))
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("test_mse", mse)
        mlflow.sklearn.log_model(model, "model")
# View: mlflow ui  →  http://localhost:5000
```

## Tool Comparison

| Tool | OSS | Local UI | Cloud | Key strength |
|------|-----|----------|-------|--------------|
| MLflow | ✅ | ✅ | ✅ | Universal — any framework |
| W&B | Partial | ❌ | ✅ | Rich dashboards, sweeps |
| DVC | ✅ | ✅ | ✅ | Data versioning |
| TensorBoard | ✅ | ✅ | ❌ | Deep TF/PyTorch integration |

## Learning Resources
- [MLflow docs](https://mlflow.org/docs/latest/)
- [Weights & Biases](https://docs.wandb.ai/)

Log model training and metrics.

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
