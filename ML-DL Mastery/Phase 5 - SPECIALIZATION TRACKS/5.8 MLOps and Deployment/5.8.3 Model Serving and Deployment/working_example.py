"""
Working Example: Model Serving and Deployment
Covers REST API design, batching strategies, model optimisation,
and serving framework patterns.
"""
import numpy as np
import os, time, json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_model_serving")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# ── 1. Serving approaches ─────────────────────────────────────────────────────
def serving_overview():
    print("=== Model Serving and Deployment ===")
    print()
    print("  Serving patterns:")
    patterns = [
        ("Online (real-time)", "One request at a time; <100ms SLA; REST/gRPC"),
        ("Batch inference",    "Process large datasets; nightly; Spark / Ray"),
        ("Streaming",          "Flink/Kafka with embedded models; continuous"),
        ("Edge deployment",    "On-device; ONNX, TFLite, Core ML; no network"),
    ]
    for p, d in patterns:
        print(f"  {p:<22} {d}")
    print()
    print("  Serving frameworks:")
    frameworks = [
        ("TorchServe",       "PyTorch official; gRPC + REST; model registry"),
        ("TF Serving",       "TensorFlow official; Docker; gRPC; batching"),
        ("Triton Inference", "NVIDIA; multi-framework; GPU batching; concurrent"),
        ("BentoML",          "Python-first; auto Docker; Kubernetes operators"),
        ("Ray Serve",        "Distributed; composable; Python native"),
        ("FastAPI",          "Custom REST; flexible; manual batching"),
        ("Seldon Core",      "Kubernetes-native; canary; A/B; monitoring"),
    ]
    for f, d in frameworks:
        print(f"  {f:<18} {d}")


# ── 2. REST API design ────────────────────────────────────────────────────────
def rest_api_design():
    print("\n=== REST API Design for ML Models ===")
    print()
    api_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

app = FastAPI(title="Prediction API", version="1.0.0")

class PredictRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = "latest"

class PredictResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str
    latency_ms: float

@app.post("/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if len(req.features) != EXPECTED_DIM:
        raise HTTPException(400, f"Expected {EXPECTED_DIM} features")
    start = time.time()
    x     = np.array(req.features)
    prob  = model.predict_proba(x.reshape(1,-1))[0, 1]
    return PredictResponse(
        prediction=float(prob > 0.5),
        probability=float(prob),
        model_version="v1.2.0",
        latency_ms=(time.time()-start)*1000
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return {"requests_total": counter, "avg_latency_ms": avg_lat}
'''
    print(api_code)

    print("  API design principles:")
    principles = [
        "Versioned endpoints: /v1/predict, /v2/predict",
        "Idempotent: same input always returns same output",
        "Health check: /health endpoint for load balancer probes",
        "Input validation: Pydantic schemas; reject malformed requests",
        "Output schema: include model version, confidence, latency",
        "Rate limiting: protect against DDoS; token bucket algorithm",
        "Authentication: API keys or OAuth2; never embed in URL",
    ]
    for p in principles:
        print(f"  • {p}")


# ── 3. Batching for throughput ────────────────────────────────────────────────
def batching_demo():
    print("\n=== Dynamic Batching for Throughput ===")
    print()
    print("  Dynamic batching: accumulate requests for a window, then infer together")
    print("  Trades latency for throughput; essential for GPU efficiency")
    print()

    class SimpleModel:
        def __init__(self, W):
            self.W = W
        def predict(self, X):
            # simulate per-item overhead vs batch
            return sigmoid(X @ self.W)

    rng = np.random.default_rng(0)
    n_features = 16
    W = rng.normal(0, 0.1, (n_features, 1))
    model = SimpleModel(W)

    def simulate_latency(batch_size, n_batches=50):
        """Returns throughput (items/s)."""
        total_items = 0
        start = time.perf_counter()
        for _ in range(n_batches):
            X = rng.standard_normal((batch_size, n_features))
            _ = model.predict(X)
            total_items += batch_size
            time.sleep(0.0001)   # simulate I/O overhead per batch
        elapsed = time.perf_counter() - start
        return total_items / elapsed

    print(f"  {'Batch size':<12} {'Throughput (items/s)'}")
    for bs in [1, 4, 16, 64, 256]:
        tp = simulate_latency(bs)
        print(f"  {bs:<12} {tp:>18.0f}")

    print()
    print("  Triton batching config (model.config.pbtxt snippet):")
    triton_cfg = '''
dynamic_batching {
  preferred_batch_size: [32, 64, 128]
  max_queue_delay_microseconds: 5000  # 5ms window
}
'''
    print(triton_cfg)


# ── 4. Model optimisation for serving ─────────────────────────────────────────
def model_optimisation():
    print("=== Model Optimisation for Serving ===")
    print()
    techniques = [
        ("Quantisation (INT8)",  "4× smaller; ~3× faster; minimal accuracy drop"),
        ("Pruning",              "Remove low-magnitude weights; 50-90% sparsity"),
        ("Knowledge distillation","Train small student to mimic large teacher"),
        ("ONNX export",          "Framework-agnostic; inference with ORT"),
        ("TensorRT",             "NVIDIA; layer fusion; FP16/INT8; optimised kernels"),
        ("torch.compile",        "PyTorch 2.0; graph compilation; ~20% speedup"),
        ("Speculative decoding", "LLM; draft model + verify; 2-3× speedup"),
        ("FlashAttention",       "Memory-efficient attention; I/O aware"),
    ]
    print(f"  {'Technique':<25} {'Notes'}")
    print(f"  {'─'*25} {'─'*50}")
    for t, d in techniques:
        print(f"  {t:<25} {d}")

    print()
    print("  Quantisation workflow (PyTorch):")
    quant_code = '''
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert

model.eval()
model.qconfig = get_default_qconfig("fbgemm")    # CPU
model_prepared = prepare(model)
# calibrate on representative data:
with torch.no_grad():
    for x, _ in calibration_loader:
        model_prepared(x)
model_quantized = convert(model_prepared)
torch.save(model_quantized.state_dict(), "model_int8.pt")
'''
    print(quant_code)


if __name__ == "__main__":
    serving_overview()
    rest_api_design()
    batching_demo()
    model_optimisation()
