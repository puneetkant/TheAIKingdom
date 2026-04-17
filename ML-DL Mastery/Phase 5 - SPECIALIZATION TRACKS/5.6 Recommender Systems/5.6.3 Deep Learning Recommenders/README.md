# 5.6.3 Deep Learning Recommenders

NCF, two-tower models, BERT4Rec, session-based, embedding tables, negative sampling.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Two-tower retrieval model |
| `working_example2.py` | NCF from scratch: embedding + MLP → binary BPR loss |
| `working_example.ipynb` | Interactive: embedding dot-product scoring |

## Quick Reference

```python
# Two-tower (retrieval)
class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)

# NCF (concatenate + MLP)
class NCF(nn.Module):
    def forward(self, u, i):
        h = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(h)

# BPR loss (pairwise)
bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
```

## Model Comparison

| Model | Type | Input | Key idea |
|-------|------|-------|---------|
| NCF | Pointwise | User+Item | MLP on embeddings |
| Two-tower | Retrieval | User+Item | ANN search |
| SASRec | Seq attention | Item history | Causal transformer |
| BERT4Rec | Seq masked | Item history | MLM on sequences |

## Learning Resources
- [NCF paper](https://arxiv.org/abs/1708.05031)
- [RecBole library](https://recbole.io/)

Build a simple recommendation engine.

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
