# 5.7.5 Applications

GNNs applied to social networks, molecules, knowledge graphs, traffic, RecSys, fraud detection.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Real-world GNN application examples |
| `working_example2.py` | Community detection via spectral embedding on SBM graph |
| `working_example.ipynb` | Interactive: SBM graph + spectral clustering |

## Quick Reference

```python
# Stochastic Block Model (synthetic community graph)
# p_in >> p_out → clear communities

# Spectral embedding (Laplacian Eigenmaps)
L = D - A
eigvals, eigvecs = np.linalg.eigh(L)
Z = eigvecs[:, 1:k+1]   # skip trivial eigenvector (λ=0)

# Molecular graph: atoms = nodes, bonds = edges
# GCN predicts property from graph-level readout

# Knowledge graph embedding: TransE
# score(h, r, t) = -||h + r - t||
```

## Application Domains

| Domain | Input Graph | Prediction |
|--------|-------------|-----------|
| Social network | Users + friendships | Community, influence |
| Molecules | Atoms + bonds | Property, toxicity |
| Knowledge graph | Entities + relations | Link prediction |
| Traffic | Road network | Speed, ETA |
| RecSys | User-item bipartite | Click, purchase |

## Learning Resources
- [GNN for drug discovery](https://arxiv.org/abs/2012.01981)
- [PyG applications](https://pytorch-geometric.readthedocs.io/)

Explore this topic with a small practical project or coding exercise.

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
