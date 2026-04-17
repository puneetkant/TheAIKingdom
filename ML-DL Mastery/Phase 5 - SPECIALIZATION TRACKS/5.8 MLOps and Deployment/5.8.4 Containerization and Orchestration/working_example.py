"""
Working Example: Containerisation and Orchestration
Covers Docker, Kubernetes, Helm, and scaling patterns for ML workloads.
"""
import os, json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_containers")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Docker for ML ──────────────────────────────────────────────────────────
def docker_patterns():
    print("=== Docker for ML Services ===")
    print()
    print("  Best-practice Dockerfile for a Python ML service:")
    print()
    dockerfile = '''\
# ── Stage 1: build dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime ────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/

# Security: run as non-root
RUN useradd -m appuser
USER appuser

ENV PYTHONUNBUFFERED=1 \\
    PATH=/root/.local/bin:$PATH

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s \\
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "src/main.py"]
'''
    print(dockerfile)

    print("  Key Docker concepts for ML:")
    concepts = [
        ("Multi-stage build",   "Separate build/runtime; smaller image (2GB→200MB)"),
        ("Layer caching",       "COPY requirements.txt first; pip install cached"),
        ("Non-root user",       "Security; never run as root in production"),
        (".dockerignore",       "Exclude data/, .git/, __pycache__, *.pt"),
        ("Volume mounts",       "Mount model weights; avoid baking large files"),
        ("GPU support",         "nvidia/cuda base image; --gpus all flag"),
        ("Health checks",       "Kubernetes liveness/readiness probes use these"),
    ]
    for c, d in concepts:
        print(f"  {c:<22} {d}")


# ── 2. Kubernetes for ML ──────────────────────────────────────────────────────
def kubernetes_patterns():
    print("\n=== Kubernetes for ML Serving ===")
    print()
    print("  Deployment manifest (inference service):")
    print()
    manifest = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prediction-service
  template:
    metadata:
      labels:
        app: prediction-service
    spec:
      containers:
      - name: model-server
        image: myrepo/prediction-service:v1.2.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
        env:
        - name: MODEL_VERSION
          valueFrom:
            configMapKeyRef:
              name: model-config
              key: version
        livenessProbe:
          httpGet: {path: /health, port: 8080}
          initialDelaySeconds: 30
        readinessProbe:
          httpGet: {path: /ready, port: 8080}
          initialDelaySeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: prediction-service
spec:
  selector:
    app: prediction-service
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prediction-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
    print(manifest)

    print("  Key K8s concepts:")
    k8s_concepts = [
        ("Pod",               "Smallest deployable unit; 1+ containers"),
        ("Deployment",        "Manages ReplicaSets; rolling updates; rollback"),
        ("Service",           "Stable DNS + IP; load balancing across pods"),
        ("HPA",               "Horizontal Pod Autoscaler; scale on CPU/custom metrics"),
        ("ConfigMap",         "Non-secret config; model version, feature flags"),
        ("Secret",            "Encrypted config; API keys, DB passwords"),
        ("PersistentVolume",  "Shared model storage; ReadWriteMany NFS"),
        ("ResourceQuota",     "Limit GPU usage per namespace"),
        ("Namespace",         "Isolation; dev/staging/prod separation"),
    ]
    for c, d in k8s_concepts:
        print(f"  {c:<22} {d}")


# ── 3. Kubeflow Pipelines ─────────────────────────────────────────────────────
def kubeflow_patterns():
    print("\n=== Kubeflow Pipelines ===")
    print()
    print("  Orchestration framework for ML workflows on Kubernetes")
    print()
    kfp_code = '''\
import kfp
from kfp import dsl

@dsl.component(base_image="python:3.11")
def preprocess(data_path: str, output_path: str) -> None:
    # preprocessing logic
    pass

@dsl.component(base_image="myrepo/training:latest")
def train(data_path: str, model_path: str, lr: float = 0.01) -> float:
    # training logic; return val_loss
    return val_loss

@dsl.component
def evaluate(model_path: str, threshold: float = 0.9) -> bool:
    # return True if model meets quality bar
    return accuracy > threshold

@dsl.pipeline(name="ml-pipeline")
def pipeline(data_path: str, lr: float = 0.01):
    prep  = preprocess(data_path=data_path, output_path="/tmp/data")
    tr    = train(data_path=prep.output, lr=lr, model_path="/tmp/model")
    ev    = evaluate(model_path=tr.outputs["model_path"])
    # conditional: only deploy if evaluation passes
    with dsl.If(ev.output == True):
        deploy(model_path=tr.outputs["model_path"])

client = kfp.Client(host="http://kubeflow-pipelines.example.com")
client.create_run_from_pipeline_func(pipeline, arguments={"lr": 0.001})
'''
    print(kfp_code)

    print("  Alternatives:")
    alts = [
        ("Argo Workflows", "Kubernetes-native; YAML DAGs; not ML-specific"),
        ("Prefect",        "Python-first; cloud or self-hosted; rich UI"),
        ("Airflow",        "DAG-based; mature; Astronomer for managed"),
        ("Metaflow",       "Netflix; Python-native; S3-backed; Kubernetes step"),
        ("ZenML",          "MLOps framework with pipeline abstraction; stack-agnostic"),
    ]
    for a, d in alts:
        print(f"  {a:<16} {d}")


if __name__ == "__main__":
    docker_patterns()
    kubernetes_patterns()
    kubeflow_patterns()
