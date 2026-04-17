# 5.7.4 Advanced GNN Topics

GAT, over-smoothing, hierarchical pooling, equivariance, scalable GNNs, PNA.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Advanced PyG patterns |
| `working_example2.py` | GAT attention + over-smoothing curve |
| `working_example.ipynb` | Interactive: attention weights + over-smoothing |

## Quick Reference

```python
# GAT: attention weight between node i and j
# e_ij = LeakyReLU( a^T [ W h_i || W h_j ] )
# α_ij = exp(e_ij) / Σ_{k∈N(i)} exp(e_ik)
# h_i  = σ( Σ_j α_ij W h_j )

# Over-smoothing: feature convergence after many layers
# Measured as pairwise distance std → 0 with more layers

# DiffPool (hierarchical pooling)
# S = softmax(GNN_pool(A, X))   # soft cluster assignment (N→K)
# X_new = S^T X                 # K × F
# A_new = S^T A S               # K × K

# Graph transformer: attention via QKV
# Q, K, V = H @ Wq, H @ Wk, H @ Wv
# α = softmax(Q K^T / sqrt(d)) masked to edges
```

## Challenges and Solutions

| Problem | Solution |
|---------|---------|
| Over-smoothing | DropEdge, JK-Net, residuals |
| Over-squashing | Graph rewiring, MPNN alternatives |
| Scalability | Mini-batch sampling (SAGE, ClusterGCN) |
| Long-range deps | Graph transformers, virtual nodes |

## Learning Resources
- [GAT paper](https://arxiv.org/abs/1710.10903)
- [Over-smoothing survey](https://arxiv.org/abs/2006.13318)

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
