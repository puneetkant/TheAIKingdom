"""
Working Example: Cloud Platforms for ML
Covers AWS SageMaker, Google Vertex AI, Azure ML, and
managed service patterns for training and deployment.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_cloud_ml")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Platform comparison ────────────────────────────────────────────────────
def platform_comparison():
    print("=== Cloud ML Platforms Comparison ===")
    print()
    platforms = {
        "AWS SageMaker": {
            "Training":    "SageMaker Training Jobs; Spot Instances; distributed",
            "Serving":     "Real-time Endpoints; Serverless; Batch Transform",
            "Pipelines":   "SageMaker Pipelines; Step Functions",
            "Feature store":"SageMaker Feature Store (online+offline)",
            "Experiments": "SageMaker Experiments",
            "Registry":    "SageMaker Model Registry",
            "AutoML":      "SageMaker Autopilot",
        },
        "Google Vertex AI": {
            "Training":    "Custom Training; Distributed; TPU pods",
            "Serving":     "Online Prediction; Batch Prediction; Matching Engine",
            "Pipelines":   "Vertex AI Pipelines (Kubeflow backend)",
            "Feature store":"Vertex Feature Store",
            "Experiments": "Vertex Experiments + Tensorboard",
            "Registry":    "Vertex Model Registry",
            "AutoML":      "AutoML Tables, Vision, NLP, Video",
        },
        "Azure ML": {
            "Training":    "Azure ML Jobs; GPU clusters; low-priority VMs",
            "Serving":     "Managed Online Endpoints; Batch Endpoints",
            "Pipelines":   "Azure ML Pipelines; Designer (GUI)",
            "Feature store":"Azure ML Feature Store (Preview)",
            "Experiments": "Azure ML Studio; MLflow integration",
            "Registry":    "Azure ML Model Registry",
            "AutoML":      "Azure AutoML",
        },
    }
    for platform, features in platforms.items():
        print(f"\n  ── {platform} ──────────────────────────────────────────────")
        for k, v in features.items():
            print(f"    {k:<16} {v}")


# ── 2. SageMaker training job pattern ─────────────────────────────────────────
def sagemaker_pattern():
    print("\n=== AWS SageMaker: Key Patterns ===")
    print()
    print("  Training job (Python SDK):")
    sm_training = '''
import sagemaker
from sagemaker.pytorch import PyTorch

session    = sagemaker.Session()
role       = sagemaker.get_execution_role()

estimator = PyTorch(
    entry_point  = "train.py",
    source_dir   = "src/",
    role         = role,
    framework_version = "2.1.0",
    py_version   = "py311",
    instance_type= "ml.p3.2xlarge",    # 1× V100
    instance_count = 1,
    hyperparameters= {"lr": 0.001, "epochs": 50},
    use_spot_instances   = True,        # up to 90% cheaper
    max_wait             = 7200,        # seconds
    checkpoint_s3_uri    = "s3://my-bucket/checkpoints/",
)

estimator.fit({
    "train": "s3://my-bucket/data/train/",
    "val":   "s3://my-bucket/data/val/",
})
'''
    print(sm_training)

    print("  Real-time endpoint deployment:")
    sm_deploy = '''
# Deploy after training
predictor = estimator.deploy(
    initial_instance_count = 2,
    instance_type          = "ml.m5.xlarge",
    endpoint_name          = "my-model-v1",
)

# Autoscaling
import boto3
asg = boto3.client("application-autoscaling")
asg.register_scalable_target(
    ServiceNamespace = "sagemaker",
    ResourceId       = f"endpoint/my-model-v1/variant/AllTraffic",
    ScalableDimension= "sagemaker:variant:DesiredInstanceCount",
    MinCapacity = 1, MaxCapacity = 20,
)
'''
    print(sm_deploy)


# ── 3. Vertex AI pattern ──────────────────────────────────────────────────────
def vertex_ai_pattern():
    print("=== Google Vertex AI: Key Patterns ===")
    print()
    print("  Custom training job:")
    vertex_training = '''
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name  = "my-training-job",
    script_path   = "train.py",
    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
    requirements  = ["transformers", "datasets"],
    model_serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
)

model = job.run(
    dataset      = vertex_dataset,
    model_display_name = "my-model-v1",
    args         = ["--lr=0.001", "--epochs=50"],
    replica_count     = 1,
    machine_type      = "n1-standard-8",
    accelerator_type  = "NVIDIA_TESLA_V100",
    accelerator_count = 1,
    sync = True,
)

endpoint = model.deploy(
    machine_type         = "n1-standard-4",
    min_replica_count    = 1,
    max_replica_count    = 10,
    traffic_percentage   = 100,
)
response = endpoint.predict(instances=[{"input": [0.5, 1.2, 3.4]}])
'''
    print(vertex_training)


# ── 4. Cost optimisation ──────────────────────────────────────────────────────
def cost_optimisation():
    print("=== Cloud ML Cost Optimisation ===")
    print()
    strategies = [
        ("Spot / Preemptible VMs",  "Up to 90% discount; handle interruptions with checkpoints"),
        ("Right-sizing",            "Profile GPU util; downgrade if <70% used"),
        ("Auto-stop",               "Shut down notebooks/clusters when idle"),
        ("Batching inference",      "Batch Transform instead of always-on endpoint"),
        ("Serverless endpoints",    "Pay-per-request; 0 cost when not serving"),
        ("Storage tiering",         "S3 Intelligent-Tiering; move old data to Glacier"),
        ("On-demand vs reserved",   "1yr reserved saves ~40%; 3yr ~60%"),
        ("Multi-region routing",    "Route to cheapest region for non-latency tasks"),
    ]
    print(f"  {'Strategy':<26} {'Notes'}")
    for s, d in strategies:
        print(f"  {s:<26} {d}")

    print()
    print("  GPU instance comparison (relative cost, 2024 approximate):")
    gpus = [
        ("ml.g4dn.xlarge",  "T4",    1.0,   "Inference; video analytics"),
        ("ml.g5.xlarge",    "A10G",  1.7,   "LLM inference; diffusion"),
        ("ml.p3.2xlarge",   "V100",  3.2,   "Training; FP16/FP32"),
        ("ml.p4d.24xlarge", "A100×8",42.0,  "Large model training"),
        ("ml.p5.48xlarge",  "H100×8",97.0,  "LLM training; highest perf"),
    ]
    print(f"  {'Instance':<20} {'GPU':<10} {'Relative $'} {'Use case'}")
    for inst, gpu, cost, use in gpus:
        print(f"  {inst:<20} {gpu:<10} {cost:>8.1f}x   {use}")


if __name__ == "__main__":
    platform_comparison()
    sagemaker_pattern()
    vertex_ai_pattern()
    cost_optimisation()
