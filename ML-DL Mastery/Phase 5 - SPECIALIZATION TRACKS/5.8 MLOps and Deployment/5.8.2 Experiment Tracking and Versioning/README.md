# 5.8.2 Experiment Tracking and Versioning

Run logging, hyperparameter sweeps, model registry, artifact versioning, Git + DVC.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | MLflow / W&B setup |
| `working_example2.py` | Lightweight experiment tracker + LR sweep + JSON log |
| `working_example.ipynb` | Interactive: run logging + best-run comparison |

## Quick Reference

```python
# MLflow
import mlflow
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("val_acc", 0.93)
    mlflow.sklearn.log_model(model, "model")

# Weights & Biases
import wandb
wandb.init(project="my-project", config={"lr": 0.01})
wandb.log({"val_acc": 0.93})

# DVC: data versioning
# dvc add data/train.csv
# dvc run -n train python train.py

# Model registry (MLflow)
mlflow.register_model("runs:/RUN_ID/model", "MyModel")
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Run | Single training execution with params + metrics |
| Experiment | Collection of related runs |
| Artifact | Model file, dataset, plot |
| Registry | Versioned model store |

## Learning Resources
- [MLflow docs](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)

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
