# 5.6.5 Advanced Topics

Context-aware, diversity, fairness, multi-objective, exploration-exploitation, bandits.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Contextual bandits for RecSys |
| `working_example2.py` | MMR diversity re-ranking vs naive top-k |
| `working_example.ipynb` | Interactive: ε-greedy bandit + MMR |

## Quick Reference

```python
# Maximal Marginal Relevance (MMR)
def mmr_rerank(scores, sim_matrix, lam=0.5, k=5):
    selected = []; candidates = list(range(len(scores)))
    while len(selected) < k and candidates:
        if not selected:
            best = max(candidates, key=lambda i: scores[i])
        else:
            best = max(candidates, key=lambda i:
                lam * scores[i] - (1-lam) * max(sim_matrix[i,j] for j in selected))
        selected.append(best); candidates.remove(best)
    return selected

# Diversity metric: intra-list diversity (ILD)
ILD = mean(1 - sim[i,j] for all pairs (i,j) in list)

# Bandits for exploration
class UCBBandit:
    def select(self, t):
        return (self.Q + np.sqrt(2*np.log(t)/(self.N+1))).argmax()
```

## Advanced Topics

| Topic | Technique | Goal |
|-------|-----------|------|
| Diversity | MMR, DPP | Reduce filter bubble |
| Fairness | Regularisation, re-ranking | Reduce bias |
| Exploration | ε-greedy, UCB, Thompson | Cold-start |
| Context | Time, location, device | Personalise |
| Multi-objective | Pareto, scalarisation | Engagement + satisfaction |

## Learning Resources
- [Diversity in RecSys survey](https://arxiv.org/abs/2010.01525)
- [Bandits for RecSys](https://arxiv.org/abs/1003.0146)

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
