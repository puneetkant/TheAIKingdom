# 5.7.3 GNN Tasks

Node classification, link prediction, graph classification — loss functions and readout strategies.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | PyG task setup |
| `working_example2.py` | All 3 tasks from scratch on synthetic graphs |
| `working_example.ipynb` | Interactive: GCN embeddings → link scores → readout |

## Quick Reference

```python
# Node classification: cross-entropy on node embeddings
def cross_entropy(probs, labels):
    return -np.log(probs[range(len(labels)), labels] + 1e-8).mean()

# Link prediction: inner product decoder
def decode(Z, i, j):           # Z: node embedding matrix
    return 1 / (1 + np.exp(-Z[i] @ Z[j]))

# Graph classification: readout + MLP
def readout(H, mode='sum'):
    if mode == 'sum':  return H.sum(axis=0)
    if mode == 'mean': return H.mean(axis=0)
    if mode == 'max':  return H.max(axis=0)
```

## Task Overview

| Task | Level | Loss | Output |
|------|-------|------|--------|
| Node classification | Node | Cross-entropy | Softmax per node |
| Link prediction | Edge | Binary CE | Sigmoid score |
| Graph classification | Graph | Cross-entropy | Softmax after readout |

## Learning Resources
- [PyG tasks tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)

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
