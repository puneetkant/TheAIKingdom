# 5.7.2 GNN Architectures

GCN, GraphSAGE, GAT, GIN — message passing, aggregation, update, expressivity.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | PyG/DGL intro |
| `working_example2.py` | GCN + GraphSAGE layer from scratch on toy graph |
| `working_example.ipynb` | Interactive: one GCN layer visualisation |

## Quick Reference

```python
import numpy as np

# GCN layer: H = σ( D̃^{-½} Ã D̃^{-½} X W )
def gcn_layer(A, X, W):
    A_hat = A + np.eye(len(A))               # add self-loops
    D_inv_sqrt = np.diag(1/np.sqrt(A_hat.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return np.tanh(A_norm @ X @ W)

# GraphSAGE (mean aggregation)
def sage_layer(A, X, W_self, W_neigh):
    A_norm = A / A.sum(axis=1, keepdims=True).clip(1)
    return np.maximum(0, X @ W_self.T + (A_norm @ X) @ W_neigh.T)

# GAT: softmax attention weights
# α_ij = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
```

## Architecture Comparison

| Model | Aggregation | Key Idea |
|-------|-------------|----------|
| GCN | Symmetric norm | Spectral |
| GraphSAGE | Mean/max/LSTM | Inductive |
| GAT | Attention | Adaptive weights |
| GIN | Sum + MLP | Most expressive |

## Learning Resources
- [GCN paper (Kipf & Welling)](https://arxiv.org/abs/1609.02907)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

Work with graphs and GNN concepts.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
