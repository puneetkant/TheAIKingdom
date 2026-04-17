"""
Working Example: Experiment Tracking
Covers MLflow, Weights & Biases, and TensorBoard concepts,
hyperparameter logging, metric tracking, model registry,
and reproducibility — runnable with or without those libraries.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os, time, json, hashlib, warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

# Optional: check for MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# ── Minimal in-memory experiment tracker ──────────────────────────────────────
@dataclass
class Run:
    run_id:    str
    name:      str
    params:    Dict[str, Any]
    metrics:   Dict[str, float]
    tags:      Dict[str, str]
    artifacts: List[str]
    start_time: float
    end_time:   float = 0.0

    @property
    def duration(self): return self.end_time - self.start_time


class Tracker:
    """Lightweight in-memory experiment tracker (MLflow-like API)."""
    def __init__(self, experiment_name="default"):
        self.experiment = experiment_name
        self.runs: List[Run] = []
        self._active: Run | None = None

    def start_run(self, name=""):
        rid = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        self._active = Run(run_id=rid, name=name or f"run_{len(self.runs)}",
                           params={}, metrics={}, tags={}, artifacts=[],
                           start_time=time.time())

    def log_param(self, key, value):
        if self._active: self._active.params[key] = value

    def log_params(self, d):
        for k, v in d.items(): self.log_param(k, v)

    def log_metric(self, key, value):
        if self._active: self._active.metrics[key] = value

    def log_metrics(self, d):
        for k, v in d.items(): self.log_metric(k, v)

    def set_tag(self, key, value):
        if self._active: self._active.tags[key] = value

    def log_artifact(self, path):
        if self._active: self._active.artifacts.append(path)

    def end_run(self):
        if self._active:
            self._active.end_time = time.time()
            self.runs.append(self._active)
            self._active = None

    def best_run(self, metric, higher_is_better=True):
        fn = max if higher_is_better else min
        return fn(self.runs, key=lambda r: r.metrics.get(metric, -1e9))

    def to_dataframe_rows(self):
        rows = []
        for r in self.runs:
            row = {"run_id": r.run_id, "name": r.name, "duration_s": f"{r.duration:.2f}"}
            row.update({f"param_{k}": v for k, v in r.params.items()})
            row.update({f"metric_{k}": f"{v:.4f}" for k, v in r.metrics.items()})
            rows.append(row)
        return rows


# ── 1. Basic tracking demo ────────────────────────────────────────────────────
def basic_tracking():
    print("=== Basic Experiment Tracking Demo ===")

    X, y = make_classification(n_samples=800, n_features=20, random_state=42)
    X    = StandardScaler().fit_transform(X)
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

    tracker = Tracker("classification_study")

    configs = [
        {"model": "logistic",    "C": 1.0,   "max_iter": 200},
        {"model": "logistic",    "C": 0.1,   "max_iter": 200},
        {"model": "rf",          "n_est": 100, "max_depth": 5},
        {"model": "rf",          "n_est": 200, "max_depth": None},
        {"model": "gbm",         "n_est": 100, "lr": 0.1},
        {"model": "gbm",         "n_est": 200, "lr": 0.05},
    ]

    for cfg in configs:
        tracker.start_run(name=f"{cfg['model']}_{len(tracker.runs)}")
        tracker.log_params(cfg)
        tracker.set_tag("framework", "sklearn")

        t0 = time.time()
        if cfg["model"] == "logistic":
            clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"],
                                     random_state=0).fit(Xtr, ytr)
        elif cfg["model"] == "rf":
            clf = RandomForestClassifier(n_estimators=cfg["n_est"],
                                         max_depth=cfg.get("max_depth"),
                                         random_state=0).fit(Xtr, ytr)
        else:
            clf = GradientBoostingClassifier(n_estimators=cfg["n_est"],
                                              learning_rate=cfg["lr"],
                                              random_state=0).fit(Xtr, ytr)
        train_time = time.time() - t0
        yp = clf.predict(Xts)
        yp_prob = clf.predict_proba(Xts)[:, 1]

        tracker.log_metrics({
            "accuracy": accuracy_score(yts, yp),
            "f1":       f1_score(yts, yp),
            "auc":      roc_auc_score(yts, yp_prob),
            "train_sec": train_time
        })
        tracker.end_run()

    # Print summary
    rows = tracker.to_dataframe_rows()
    keys = list(rows[0].keys())
    metric_cols = [k for k in keys if k.startswith("metric_")]
    param_cols  = [k for k in keys if k.startswith("param_")]

    print(f"  {'Name':<20} {'Acc':>6} {'F1':>6} {'AUC':>6} | Params")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6}   {'─'*30}")
    for r in tracker.runs:
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())[:40]
        print(f"  {r.name:<20} {r.metrics['accuracy']:>6.4f} "
              f"{r.metrics['f1']:>6.4f} {r.metrics['auc']:>6.4f} | {params_str}")

    best = tracker.best_run("auc")
    print(f"\n  Best run (AUC): {best.name}  AUC={best.metrics['auc']:.4f}")
    return tracker


# ── 2. Hyperparameter sweep ───────────────────────────────────────────────────
def hyperparameter_sweep():
    print("\n=== Hyperparameter Sweep ===")
    X, y = make_classification(n_samples=600, n_features=15, random_state=7)
    X    = StandardScaler().fit_transform(X)
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=7)

    grid = {
        "n_estimators":   [50, 100, 200],
        "max_depth":      [3, 5, 7],
        "learning_rate":  [0.05, 0.1, 0.2],
    }
    tracker = Tracker("gbm_sweep")

    best_auc, best_cfg = 0, None
    for params in ParameterGrid(grid):
        tracker.start_run()
        tracker.log_params(params)
        clf = GradientBoostingClassifier(**params, random_state=0).fit(Xtr, ytr)
        auc = roc_auc_score(yts, clf.predict_proba(Xts)[:, 1])
        tracker.log_metric("auc", auc)
        tracker.end_run()
        if auc > best_auc:
            best_auc, best_cfg = auc, params

    print(f"  Grid size:  {len(list(ParameterGrid(grid)))} runs")
    print(f"  Best AUC:   {best_auc:.4f}")
    print(f"  Best config: {best_cfg}")

    # Distribution of AUC
    aucs = sorted([r.metrics["auc"] for r in tracker.runs], reverse=True)
    print(f"  AUC stats: min={min(aucs):.4f}  median={np.median(aucs):.4f}  max={max(aucs):.4f}")
    print(f"  Top-5 AUCs: {[f'{a:.4f}' for a in aucs[:5]]}")


# ── 3. MLflow concepts ────────────────────────────────────────────────────────
def mlflow_concepts():
    print("\n=== MLflow Concepts ===")
    concepts = [
        ("Experiment",   "A named collection of runs (e.g., 'ResNet-CIFAR10')"),
        ("Run",          "Single execution with logged params, metrics, artifacts"),
        ("Parameters",   "Hyperparameters: learning_rate=0.01, batch_size=64"),
        ("Metrics",      "Scalar values per step: {'train_loss': 0.23, 'val_acc': 0.91}"),
        ("Artifacts",    "Files: model.pkl, confusion_matrix.png, tokenizer.json"),
        ("Model",        "MLflow model format: unified interface for multiple backends"),
        ("Registry",     "Versioned model store with staging/production/archived"),
        ("Autolog",      "mlflow.sklearn.autolog() — automatic param/metric capture"),
    ]
    print(f"  {'Concept':<15} Description")
    print(f"  {'─'*15} {'─'*50}")
    for c, d in concepts:
        print(f"  {c:<15} {d}")

    if HAS_MLFLOW:
        print("\n  MLflow is installed — minimal live demo:")
        mlflow.set_experiment("demo")
        with mlflow.start_run(run_name="demo_run"):
            mlflow.log_param("lr", 0.01)
            mlflow.log_metric("accuracy", 0.92)
        print("  Run logged to MLflow tracking URI:", mlflow.get_tracking_uri())
    else:
        print("\n  [MLflow not installed — code pattern below]")
        print("""
  import mlflow
  mlflow.set_experiment("my_experiment")

  with mlflow.start_run(run_name="run_1"):
      mlflow.log_params({"lr": 0.01, "batch_size": 64})
      for epoch in range(n_epochs):
          # ... train ...
          mlflow.log_metrics({"train_loss": loss, "val_acc": acc}, step=epoch)
      mlflow.sklearn.log_model(model, artifact_path="model")
        """)


# ── 4. Weights & Biases overview ──────────────────────────────────────────────
def wandb_overview():
    print("\n=== Weights & Biases (W&B) Overview ===")
    features = [
        ("wandb.init()",         "Start a run; project/entity/config kwargs"),
        ("wandb.log({})",        "Log dict of scalars, images, tables, audio"),
        ("wandb.config",         "Hyperparameter config synced to cloud"),
        ("wandb.watch(model)",   "Track gradients and parameters over training"),
        ("wandb.Artifact",       "Version datasets, models, code as artefacts"),
        ("wandb.Sweep",          "Bayesian/grid/random hyper-param sweeps"),
        ("wandb.Table",          "Interactive table of predictions, metrics"),
        ("wandb.Image",          "Log images; compare across runs visually"),
    ]
    print(f"  {'API':<28} Description")
    print(f"  {'─'*28} {'─'*40}")
    for api, desc in features:
        print(f"  {api:<28} {desc}")

    print("""
  Typical training loop:
  import wandb
  wandb.init(project="my_project", config={"lr": 0.01, "epochs": 50})

  for epoch in range(config.epochs):
      train(model, loader)
      metrics = evaluate(model, val_loader)
      wandb.log({"epoch": epoch, **metrics})

  wandb.finish()
    """)


# ── 5. TensorBoard overview ───────────────────────────────────────────────────
def tensorboard_overview():
    print("=== TensorBoard Overview ===")
    print("  Launch: tensorboard --logdir=runs/")
    print()
    panels = [
        ("Scalars",     "Loss/accuracy curves across epochs and runs"),
        ("Images",      "Input samples, reconstructions, attention maps"),
        ("Histograms",  "Weight / gradient distributions per layer"),
        ("Projector",   "High-dim embedding visualisation (t-SNE, PCA, UMAP)"),
        ("HParams",     "Hyperparameter sweep comparison dashboard"),
        ("Graph",       "Computation graph / model architecture"),
        ("Text",        "Sample generated text per epoch"),
        ("PR Curves",   "Precision-recall curves at different thresholds"),
    ]
    print(f"  {'Panel':<14} Description")
    print(f"  {'─'*14} {'─'*45}")
    for panel, desc in panels:
        print(f"  {panel:<14} {desc}")

    print("""
  PyTorch usage:
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter("runs/exp1")
  writer.add_scalar("loss/train", loss, global_step)
  writer.add_histogram("fc1.weight", model.fc1.weight, epoch)
  writer.add_images("inputs", imgs, epoch)
  writer.close()

  Keras/TF usage:
  tb_cb = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
  model.fit(..., callbacks=[tb_cb])
    """)


# ── 6. Reproducibility checklist ─────────────────────────────────────────────
def reproducibility():
    print("=== Reproducibility Checklist ===")
    checks = [
        ("Random seeds",    "Set numpy / torch / tf / random / CUDA seeds"),
        ("Environment",     "Log Python, library versions (pip freeze)"),
        ("Data versioning", "Hash or version datasets; use DVC or MLflow artifacts"),
        ("Code versioning", "Git commit hash logged with every run"),
        ("Determinism",     "torch.backends.cudnn.deterministic = True"),
        ("Config files",    "Store all hyperparams in YAML/JSON, not hardcoded"),
        ("Model checkpoints", "Save weights + optimizer state + epoch"),
        ("Containers",      "Docker image pins OS + CUDA + Python environment"),
    ]
    print(f"  {'Area':<20} Action")
    print(f"  {'─'*20} {'─'*45}")
    for area, action in checks:
        print(f"  {area:<20} {action}")

    # Example seed function
    print("""
  def set_seed(seed=42):
      import random, os
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      try:
          import torch
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
      except ImportError: pass
    """)


if __name__ == "__main__":
    basic_tracking()
    hyperparameter_sweep()
    mlflow_concepts()
    wandb_overview()
    tensorboard_overview()
    reproducibility()
