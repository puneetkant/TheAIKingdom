"""
Working Example: CI/CD for ML
Covers continuous integration, testing strategies, automated pipelines,
and GitOps patterns for ML systems.
"""
import numpy as np
import os, json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_cicd")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. CI/CD overview ---------------------------------------------------------
def cicd_overview():
    print("=== CI/CD for ML Systems ===")
    print()
    print("  Traditional CI/CD + MLOps additions:")
    print()
    pipeline_stages = [
        ("1. Code CI",        "lint, unit tests, type check on code commit"),
        ("2. Data CI",        "validate schema, stats, volume on data change"),
        ("3. Training",       "automated training on new data or code change"),
        ("4. Model testing",  "accuracy gates, bias checks, latency benchmarks"),
        ("5. Container build","build Docker image; push to registry"),
        ("6. Staging deploy", "deploy to staging; integration tests"),
        ("7. Prod deploy",    "canary -> full rollout; automated or approval gate"),
        ("8. Monitoring",     "drift detection triggers retraining"),
    ]
    for s, d in pipeline_stages:
        print(f"  {s:<18} {d}")

    print()
    print("  CI/CD platforms for ML:")
    platforms = [
        ("GitHub Actions",  "Built-in; self-hosted runners with GPU"),
        ("GitLab CI",       "On-prem option; strong Docker support"),
        ("Jenkins",         "Self-hosted; flexible; large plugin ecosystem"),
        ("CircleCI",        "Fast; parallelism; resource classes for GPU"),
        ("Tekton",          "Kubernetes-native; used by Kubeflow Pipelines"),
    ]
    for p, d in platforms:
        print(f"  {p:<18} {d}")


# -- 2. GitHub Actions workflow ------------------------------------------------
def github_actions_pattern():
    print("\n=== GitHub Actions ML Pipeline ===")
    print()
    workflow_yaml = '''\
name: ML CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # -- Step 1: Code Quality ----------------------------------------------------
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: "3.11"}
    - run: pip install -r requirements-dev.txt
    - run: ruff check src/
    - run: mypy src/
    - run: pytest tests/unit/ -v --cov=src

  # -- Step 2: Data Validation -------------------------------------------------
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - run: pip install great_expectations dvc
    - run: dvc pull data/validation_set
    - run: python scripts/validate_data.py

  # -- Step 3: Training --------------------------------------------------------
  train:
    needs: [lint-and-test, data-validation]
    runs-on: [self-hosted, gpu]
    steps:
    - uses: actions/checkout@v4
    - run: dvc repro train
    - uses: actions/upload-artifact@v4
      with:
        name: model-artefacts
        path: models/

  # -- Step 4: Model Evaluation Gate ------------------------------------------
  evaluate:
    needs: train
    runs-on: ubuntu-latest
    steps:
    - uses: actions/download-artifact@v4
      with: {name: model-artefacts}
    - run: python scripts/evaluate.py --threshold 0.92
    - run: python scripts/check_bias.py
    - run: python scripts/benchmark_latency.py --max-p99-ms 100

  # -- Step 5: Build and Push Image -------------------------------------------
  build-image:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - uses: docker/build-push-action@v5
      with:
        push: true
        tags: ghcr.io/myorg/model-service:${{ github.sha }}

  # -- Step 6: Deploy to Staging ----------------------------------------------
  deploy-staging:
    needs: build-image
    environment: staging
    steps:
    - run: |
        kubectl set image deployment/model-service \\
          model-server=ghcr.io/myorg/model-service:${{ github.sha }}
    - run: pytest tests/integration/ --env staging
'''
    print(workflow_yaml)


# -- 3. ML testing strategies --------------------------------------------------
def ml_testing_strategies():
    print("=== ML Testing Strategies ===")
    print()
    test_types = [
        ("Unit tests",           "Test individual functions: preprocessing, metrics, utils"),
        ("Model interface tests","Correct input/output shapes and types"),
        ("Data validation",      "Schema, range, uniqueness, no target leakage"),
        ("Training smoke test",  "Run 1 epoch on tiny dataset; loss decreases"),
        ("Evaluation gate",      "Fail CI if val_accuracy < threshold"),
        ("Bias/fairness tests",  "Performance parity across demographic groups"),
        ("Latency benchmark",    "p99 < SLA; regression vs previous version"),
        ("Differential tests",   "New model within 2% of champion on all slices"),
        ("Integration tests",    "Full API call; correct JSON response schema"),
        ("Load tests",           "Locust/k6; 1000 RPS sustained; check degradation"),
    ]
    for t, d in test_types:
        print(f"  {t:<24} {d}")

    # Demonstrate a simple model evaluation gate
    print()
    print("  Example evaluation gate:")
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) > 0.5).astype(int)
    y_pred = y_true.copy()
    # Introduce some noise
    flip_idx = rng.choice(500, 40, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    accuracy = (y_true == y_pred).mean()
    threshold = 0.90
    passed = accuracy >= threshold
    print(f"  Accuracy: {accuracy:.4f}  |  Threshold: {threshold}")
    print(f"  Gate: {'PASSED [OK]' if passed else 'FAILED [X]'}")
    print(f"  CI exit code: {0 if passed else 1}")


if __name__ == "__main__":
    cicd_overview()
    github_actions_pattern()
    ml_testing_strategies()
