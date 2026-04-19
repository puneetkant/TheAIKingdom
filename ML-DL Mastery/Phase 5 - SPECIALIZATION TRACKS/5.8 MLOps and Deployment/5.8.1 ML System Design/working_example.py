"""
Working Example: ML System Design
Covers end-to-end ML system architecture, feature stores,
training infrastructure, and production design patterns.
"""
import numpy as np
import os, time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ml_system_design")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. ML system components ---------------------------------------------------
def ml_system_overview():
    print("=== ML System Design ===")
    print()
    print("  End-to-end ML system layers:")
    layers = [
        ("Data ingestion",   "Batch/streaming; Kafka, Kinesis, Flink"),
        ("Data storage",     "Data lake (S3/GCS), data warehouse (BigQuery/Snowflake)"),
        ("Feature store",    "Online (Redis, DynamoDB) + offline (Parquet, Hive)"),
        ("Training",         "Distributed training; GPU clusters; Vertex AI, SageMaker"),
        ("Experiment track", "MLflow, W&B, Neptune; params, metrics, artefacts"),
        ("Model registry",   "Version control for models; staging/prod lifecycle"),
        ("Serving",          "REST/gRPC; batching; Triton, TorchServe, TF Serving"),
        ("Monitoring",       "Data drift, model drift, latency, error rate"),
        ("Feedback loop",    "Capture labels, detect drift, trigger retraining"),
    ]
    for layer, desc in layers:
        print(f"  {layer:<22} {desc}")

    print()
    print("  Scaling approaches:")
    scaling = [
        ("Horizontal scale", "Add more instances; load balancer"),
        ("Vertical scale",   "Larger instances; more GPU memory"),
        ("Batch vs online",  "Asynchronous batch scoring vs real-time inference"),
        ("Caching",          "Cache predictions for popular inputs (Redis)"),
        ("Model distillation","Compress large model for fast serving"),
        ("Quantisation",     "INT8/FP16 inference; ~4× speedup, ~4× smaller"),
    ]
    for s, d in scaling:
        print(f"  {s:<22} {d}")


# -- 2. Feature store patterns -------------------------------------------------
def feature_store():
    print("\n=== Feature Store ===")
    print()
    print("  Problem: features computed differently at training vs serving")
    print("           -> training-serving skew -> silent performance degradation")
    print()
    print("  Feature store solves this by:")
    print("    1. Computing features once (no duplicate logic)")
    print("    2. Point-in-time correct feature retrieval (no data leakage)")
    print("    3. Sharing features across teams / models")
    print()
    print("  Dual-database architecture:")
    print("    Offline store: historical features for training (Parquet, Hive)")
    print("    Online store:  latest features for inference (<10ms) (Redis, DynamoDB)")
    print()
    print("  Tools:")
    tools = [
        ("Feast",       "Open-source; Kubernetes-based; GCP/AWS"),
        ("Tecton",      "Enterprise; real-time + batch; AWS-native"),
        ("Hopsworks",   "Open-source; feature group versioning"),
        ("AWS SageMaker Feature Store", "Managed; tight AWS integration"),
        ("Vertex AI Feature Store",     "Google Cloud; serverless"),
    ]
    for t, d in tools:
        print(f"  {t:<35} {d}")

    # Simulate feature serving latency
    print()
    print("  Simulated feature serving latency:")
    rng = np.random.default_rng(0)
    for backend in ["Redis (online)", "Parquet (offline)", "DB (naive)"]:
        latencies = rng.exponential(3 if "Redis" in backend else
                                    50 if "Parquet" in backend else 200, 100)
        print(f"  {backend:<22}: mean={latencies.mean():.1f}ms  p99={np.percentile(latencies,99):.1f}ms")


# -- 3. Data versioning and reproducibility ------------------------------------
def data_versioning():
    print("\n=== Data Versioning and Reproducibility ===")
    print()
    print("  Key principle: every training run must be fully reproducible")
    print("  Requirements: code version + data version + config version")
    print()
    print("  Data versioning tools:")
    tools = [
        ("DVC (Data Version Control)", "Git-like for datasets; S3/GCS backends"),
        ("Delta Lake",                 "ACID transactions on data lake; versioned tables"),
        ("Iceberg",                    "Table format; schema evolution; time-travel"),
        ("lakeFS",                     "Git-for-data layer on object storage"),
    ]
    for t, d in tools:
        print(f"  {t:<34} {d}")
    print()
    print("  DVC workflow:")
    dvc_code = """
git init; dvc init
dvc add data/training_set.csv       # track file with DVC
git add .dvc data/.gitignore
git commit -m 'Add training data'
dvc push                             # push to remote (S3/GCS)

# Reproduce pipeline
dvc repro                            # reruns stages if inputs changed
dvc dag                              # visualise dependency graph
"""
    print(dvc_code)


# -- 4. Production design patterns ---------------------------------------------
def production_patterns():
    print("=== Production ML Patterns ===")
    patterns = [
        ("Shadow mode",      "Run new model in parallel; compare without affecting users"),
        ("Canary deploy",    "Route X% traffic to new model; ramp up gradually"),
        ("A/B testing",      "Statistical comparison; measure business metrics"),
        ("Blue-green",       "Two identical envs; instant rollback by switching"),
        ("Champion-Challenger","Current champion vs new challenger; promote on win"),
        ("Multi-armed bandit","Adaptive traffic routing; maximise online metric"),
        ("Ensemble serving", "Multiple models averaged; better than any single"),
        ("Fallback chains",  "Primary -> secondary -> heuristic on failure"),
    ]
    print(f"  {'Pattern':<22} {'Description'}")
    print(f"  {'-'*22} {'-'*50}")
    for p, d in patterns:
        print(f"  {p:<22} {d}")

    print()
    print("  Model versioning checklist:")
    checklist = [
        "□ Semantic version: major.minor.patch",
        "□ Link to training run ID (MLflow/W&B)",
        "□ Link to dataset version (DVC commit)",
        "□ Performance metrics on eval set",
        "□ Bias audit results",
        "□ Inference latency benchmarks (p50/p95/p99)",
        "□ Memory footprint (RAM + GPU)",
        "□ Dependencies pinned (requirements.txt)",
    ]
    for item in checklist:
        print(f"  {item}")


if __name__ == "__main__":
    ml_system_overview()
    feature_store()
    data_versioning()
    production_patterns()
