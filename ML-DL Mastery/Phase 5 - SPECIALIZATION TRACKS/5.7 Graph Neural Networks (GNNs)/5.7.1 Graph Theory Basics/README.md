# 5.7.1 Graph Theory Basics

Nodes, edges, adjacency matrix, Laplacian, BFS/DFS, degree, clustering coefficient, spectral properties.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | NetworkX graph properties demo |
| `working_example2.py` | Synthetic Erdős–Rényi graph: degree, BFS, Laplacian eigenvalues |
| `working_example.ipynb` | Interactive: adjacency + Laplacian visualisation |

## Quick Reference

```python
import numpy as np

# Adjacency matrix → degree matrix → Laplacian
A = np.array([[0,1,1],[1,0,1],[1,1,0]], float)
D = np.diag(A.sum(axis=1))
L = D - A                          # unnormalized Laplacian
L_sym = np.diag(1/np.sqrt(D.diagonal())) @ L @ np.diag(1/np.sqrt(D.diagonal()))

# Eigenvalues of Laplacian
eigvals = np.linalg.eigvalsh(L)    # sorted ascending
# λ₀ = 0 always; λ₁ (algebraic connectivity) > 0 if connected

# NetworkX
import networkx as nx
G = nx.karate_club_graph()
print(nx.average_clustering(G))
print(nx.average_shortest_path_length(G))
```

## Key Concepts

| Concept | Definition |
|---------|-----------|
| Degree | Number of neighbours |
| Clustering coeff | Triangle density |
| Algebraic connectivity | λ₂ of Laplacian |
| Diameter | Longest shortest path |

## Learning Resources
- [NetworkX docs](https://networkx.org/)
- [Graph theory intro (Coursera)](https://www.coursera.org/learn/graphs)

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
