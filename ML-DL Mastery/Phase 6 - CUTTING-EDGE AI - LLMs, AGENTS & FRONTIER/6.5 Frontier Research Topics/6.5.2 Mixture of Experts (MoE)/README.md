# 6.5.2 Mixture of Experts (MoE)

Mixture-of-Experts models (Mixtral, Switch Transformer, GPT-4) use learned sparse routing to activate only a subset of N expert FFN sub-networks per token, achieving massive parameter counts at constant inference compute. Top-k gating with load-balance loss prevents expert collapse. This folder implements top-k gating, load-balance loss, and expert utilisation analysis.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Top-k gating with softmax, load balance loss, expert utilisation bar chart, gate weight heatmap |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `moe_gating.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Top-k routing | Each token routed to k of N experts; k=2 is common |
| Load balance loss | Penalises routing all tokens to same expert |
| Expert parallelism | Different experts on different GPUs |
| Switch Transformer | k=1 routing; simpler but less expressive |
| Expert capacity | Max tokens per expert per batch; overflow dropped |

## Learning Resources

- Shazeer et al. *Outrageously Large NNs* (2017)
- Fedus et al. *Switch Transformer* (2021)
- Jiang et al. *Mixtral of Experts* (2024)
