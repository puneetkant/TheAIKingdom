# 5.7.6 GNN Tools

PyTorch Geometric, DGL, NetworkX — data objects, message passing, mini-batching, GPU training.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | PyG / DGL setup examples |
| `working_example2.py` | Minimal graph data object + message passing + mini-batch |
| `working_example.ipynb` | Interactive: message passing layer demo |

## Quick Reference

```python
# PyTorch Geometric data object
from torch_geometric.data import Data
data = Data(
    x=node_features,          # [N, F]
    edge_index=edges_coo,     # [2, E]  (source, target)
    y=labels                  # [N] or scalar
)

# Mini-batching multiple graphs
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# DGL graph
import dgl
g = dgl.graph((src, dst))
g.ndata['h'] = node_features
# Message passing:
g.update_all(dgl.function.copy_u('h','m'), dgl.function.mean('m','h'))

# NetworkX
import networkx as nx
G = nx.from_numpy_array(A)
nx.draw(G, with_labels=True)
```

## Tool Comparison

| Tool | Backend | Strength |
|------|---------|---------|
| PyG | PyTorch | Flexible, research |
| DGL | PyTorch/MXNet | Efficient batching |
| NetworkX | Pure Python | Analysis, viz |
| Spektral | TensorFlow | Keras-style |

## Learning Resources
- [PyG docs](https://pytorch-geometric.readthedocs.io/)
- [DGL docs](https://www.dgl.ai/)
- [NetworkX](https://networkx.org/)

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
