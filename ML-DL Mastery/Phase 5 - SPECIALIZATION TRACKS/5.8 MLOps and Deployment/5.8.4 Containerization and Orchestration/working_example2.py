"""
Working Example 2: Containerisation — Docker build steps, health check, resource sizing
=========================================================================================
Simulates Docker-compose config generation and resource estimation.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def estimate_resources(batch_size, model_params_M, precision_bytes=4):
    """
    Rough GPU memory estimate for inference.
    Activation memory ~= batch_size * hidden_dim * n_layers * precision_bytes
    Model memory ~= params_M * 1e6 * precision_bytes
    """
    model_mb = model_params_M * 1e6 * precision_bytes / (1024**2)
    activation_mb = batch_size * 512 * 12 * precision_bytes / (1024**2)  # proxy
    return model_mb, activation_mb

def generate_dockerfile(model_name="my_model", port=8000):
    return f"""FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE {port}
HEALTHCHECK --interval=30s --timeout=5s \\
    CMD curl -f http://localhost:{port}/health || exit 1
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "{port}"]
# Model: {model_name}
"""

def generate_compose(services):
    lines = ["version: '3.9'", "services:"]
    for svc in services:
        lines += [
            f"  {svc['name']}:",
            f"    image: {svc['image']}",
            f"    ports:",
            f"      - '{svc['port']}:{svc['port']}'",
            f"    deploy:",
            f"      resources:",
            f"        limits:",
            f"          memory: {svc['mem_limit']}",
        ]
    return "\n".join(lines)

def demo():
    print("=== Containerisation Demo ===")
    dockerfile = generate_dockerfile("classifier_v2", port=8080)
    print("--- Dockerfile ---")
    print(dockerfile)

    services = [
        {"name": "api",    "image": "ml-api:latest",  "port": 8080, "mem_limit": "512m"},
        {"name": "worker", "image": "ml-worker:latest","port": 8081, "mem_limit": "2g"},
        {"name": "redis",  "image": "redis:7",         "port": 6379, "mem_limit": "256m"},
    ]
    compose_yaml = generate_compose(services)
    print("--- docker-compose.yml ---")
    print(compose_yaml)

    # Resource estimation across model sizes
    sizes = [10, 50, 100, 300, 700]  # millions of params
    batch = 32
    model_mbs = [estimate_resources(batch, s)[0] for s in sizes]
    act_mbs   = [estimate_resources(batch, s)[1] for s in sizes]

    print("\n--- Resource Estimates (batch=32) ---")
    for s, m, a in zip(sizes, model_mbs, act_mbs):
        print(f"  {s:4d}M params -> model {m:.0f}MB  activations {a:.0f}MB  total ~{m+a:.0f}MB")

    plt.figure(figsize=(6, 3))
    plt.plot(sizes, model_mbs, "o-", label="Model weights")
    plt.plot(sizes, act_mbs,   "s--", label="Activations")
    plt.xlabel("Model params (M)"); plt.ylabel("GPU Memory (MB)")
    plt.title("Memory Estimation"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "containerisation.png"); plt.close()
    print("  Saved containerisation.png")

def demo_kubernetes_manifest():
    """Generate a minimal Kubernetes Deployment and HPA YAML."""
    print("\n=== Kubernetes Manifest Generation ===")
    def k8s_deployment(name, image, replicas=2, cpu_req="500m", mem_req="512Mi"):
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    spec:
      containers:
      - name: {name}
        image: {image}
        resources:
          requests:
            cpu: {cpu_req}
            memory: {mem_req}"""

    def k8s_hpa(name, min_rep=1, max_rep=10, cpu_target=70):
        return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}-hpa
spec:
  scaleTargetRef:
    name: {name}
  minReplicas: {min_rep}
  maxReplicas: {max_rep}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: {cpu_target}"""

    print(k8s_deployment("ml-inference", "ml-api:v2", replicas=3))
    print(); print(k8s_hpa("ml-inference", min_rep=2, max_rep=8))


def demo_scaling_estimation():
    """Estimate required replicas for given throughput requirements."""
    print("\n=== Autoscaling Estimation ===")
    latency_ms = 50   # avg inference latency
    qps_per_pod = 1000 / latency_ms  # requests/sec per pod
    print(f"  Latency: {latency_ms}ms -> {qps_per_pod:.0f} RPS per pod")
    print(f"\n  {'Target RPS':>12}  {'Min Pods':>10}  {'w/ 2x headroom':>16}")
    for target_rps in [10, 100, 500, 2000, 10000]:
        min_pods = int(np.ceil(target_rps / qps_per_pod))
        with_headroom = min_pods * 2
        print(f"  {target_rps:>12}  {min_pods:>10}  {with_headroom:>16}")


if __name__ == "__main__":
    demo()
    demo_kubernetes_manifest()
    demo_scaling_estimation()
