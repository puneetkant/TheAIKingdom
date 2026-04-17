# 6.1.5 Inference Optimization

Deploying large language models efficiently requires techniques that reduce memory and latency without degrading quality. Key strategies include KV-cache reuse, quantisation (INT8/INT4), speculative decoding, and continuous batching. This folder benchmarks throughput, memory savings, and quality trade-offs across these methods.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | KV-cache speedup model, quantisation memory curves, speculative decoding acceptance-rate simulation |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `inference_optimization.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| KV-Cache | Reuse attention keys/values across steps; avoids recomputation |
| Quantisation | INT8/INT4 weights; 2-4× memory reduction |
| Speculative decoding | Draft model + verifier; speeds up sampling |
| Continuous batching | Dynamic request batching for higher GPU utilisation |
| FlashAttention | IO-aware exact attention; up to 3× faster |

## Learning Resources

- Leviathan et al. *Speculative Decoding* (2023)
- Dao et al. *FlashAttention* (2022)
- HuggingFace TGI documentation
