"""
Working Example: Experiment Tracking and Model Versioning
Covers MLflow, W&B patterns, hyperparameter search logging,
and model registry workflows.
"""
import numpy as np
import os, json, hashlib, time
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_experiment_tracking")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Experiment tracking concepts ───────────────────────────────────────────
def tracking_concepts():
    print("=== Experiment Tracking and Versioning ===")
    print()
    print("  What to track for every training run:")
    print()
    items = [
        ("Parameters",  "Hyperparameters: lr, batch_size, model architecture"),
        ("Metrics",     "Train/val/test loss, accuracy, F1, etc."),
        ("Artefacts",   "Model weights, tokeniser, config, plots"),
        ("Code version","Git commit hash"),
        ("Data version","DVC commit / dataset hash"),
        ("Environment", "Python version, library versions, hardware"),
        ("Tags",        "Project, team, experiment type"),
        ("Duration",    "Start time, end time, GPU hours"),
    ]
    for i, d in items:
        print(f"  {i:<14} {d}")


# ── 2. MLflow patterns ────────────────────────────────────────────────────────
def mlflow_patterns():
    print("\n=== MLflow Patterns ===")
    print()
    mlflow_code = '''
import mlflow
import mlflow.sklearn

# Start a run
with mlflow.start_run(run_name="lr_experiment"):
    # Log params
    mlflow.log_params({"lr": 0.01, "batch_size": 32, "epochs": 50})

    # Log metrics at each epoch
    for epoch in range(50):
        train_loss = train_one_epoch(model, loader)
        val_loss   = evaluate(model, val_loader)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss",   val_loss,   step=epoch)

    # Log artefacts
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("confusion_matrix.png")

# Query runs programmatically
client = mlflow.tracking.MlflowClient()
runs   = client.search_runs(experiment_ids=["1"],
                             filter_string="metrics.val_loss < 0.3",
                             order_by=["metrics.val_loss ASC"])
best_run = runs[0]

# Register model
mlflow.register_model(f"runs:/{best_run.info.run_id}/model",
                       name="ProductionClassifier")

# Transition to production
client.transition_model_version_stage(
    name="ProductionClassifier", version=1, stage="Production")
'''
    print(mlflow_code)
    print("  MLflow components:")
    components = [
        ("Tracking Server", "Stores runs; SQLite or PostgreSQL backend"),
        ("Artifact Store",  "S3/GCS/Azure Blob for model files"),
        ("Model Registry",  "Stage: None → Staging → Production → Archived"),
        ("Projects",        "Reproducible packaging with MLproject file"),
        ("Evaluate",        "Dataset-aware evaluation with custom metrics"),
    ]
    for c, d in components:
        print(f"  {c:<18} {d}")


# ── 3. W&B patterns ───────────────────────────────────────────────────────────
def wandb_patterns():
    print("\n=== Weights & Biases (W&B) Patterns ===")
    print()
    wandb_code = '''
import wandb

# Initialise
run = wandb.init(
    project="my-project",
    config={"lr": 0.01, "batch_size": 32, "architecture": "ResNet50"},
    tags=["baseline", "imagenet"],
)

# Log metrics (each call = 1 step by default)
for epoch in range(100):
    wandb.log({"train/loss": 0.5 - epoch*0.003,
               "val/accuracy": 0.7 + epoch*0.002,
               "epoch": epoch})

# Log model artifact
artifact = wandb.Artifact("model-weights", type="model")
artifact.add_file("model.pt")
run.log_artifact(artifact)

# Sweeps (hyperparameter search)
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [16, 32, 64, 128]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train, count=20)
'''
    print(wandb_code)


# ── 4. Minimal experiment logger (pure Python) ────────────────────────────────
class SimpleExperimentLogger:
    """Lightweight experiment tracker that saves JSON logs."""
    def __init__(self, name, output_dir):
        self.name    = name
        self.run_id  = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        self.params  = {}
        self.metrics = {}
        self.start_t = datetime.now().isoformat()
        self.path    = os.path.join(output_dir, f"run_{self.run_id}.json")

    def log_params(self, d):
        self.params.update(d)

    def log_metric(self, key, value, step=None):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({"step": step, "value": float(value)})

    def save(self):
        record = {"run_id": self.run_id, "name": self.name,
                  "start": self.start_t, "params": self.params,
                  "metrics": self.metrics}
        with open(self.path, 'w') as f:
            json.dump(record, f, indent=2)
        return self.path


def demo_logger():
    print("\n=== Minimal Experiment Logger Demo ===")
    rng  = np.random.default_rng(0)
    exp  = SimpleExperimentLogger("demo_run", OUTPUT_DIR)
    exp.log_params({"lr": 0.01, "batch_size": 32, "epochs": 20})

    losses = []
    for epoch in range(20):
        loss = 1.0 * np.exp(-0.15*epoch) + rng.normal(0, 0.03)
        acc  = 0.5 + 0.4*(1 - np.exp(-0.2*epoch)) + rng.normal(0, 0.02)
        exp.log_metric("train_loss", loss, step=epoch)
        exp.log_metric("val_acc",    acc,  step=epoch)
        losses.append(loss)

    path = exp.save()
    print(f"  Run ID: {exp.run_id}")
    print(f"  Final train loss: {losses[-1]:.4f}")
    print(f"  Log saved: {path}")

    # Reload and query
    with open(path) as f:
        loaded = json.load(f)
    final_acc = loaded["metrics"]["val_acc"][-1]["value"]
    print(f"  Reloaded val_acc: {final_acc:.4f}")


if __name__ == "__main__":
    tracking_concepts()
    mlflow_patterns()
    wandb_patterns()
    demo_logger()
