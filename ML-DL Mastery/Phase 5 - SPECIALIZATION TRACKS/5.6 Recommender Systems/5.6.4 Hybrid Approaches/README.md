# 5.6.4 Hybrid Approaches

Weighted hybrid, cascade, switching, feature augmentation, LightFM, two-stage systems.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | LightFM hybrid demo |
| `working_example2.py` | Content + CF weighted ensemble comparison |
| `working_example.ipynb` | Interactive: alpha sweep RMSE curve |

## Quick Reference

```python
# Weighted linear hybrid
def hybrid_score(u, i, alpha=0.5):
    return (1-alpha) * content_score(u, i) + alpha * cf_score(u, i)

# Feature augmentation: use CF embeddings as content features
user_emb = cf_model.user_factors[u]   # learned CF embedding
content_vec = tfidf_matrix[i]
hybrid_input = np.concatenate([user_emb, content_vec.toarray()[0]])
prediction = meta_model.predict([hybrid_input])

# LightFM (factorisation machines with side info)
from lightfm import LightFM
model = LightFM(loss="warp", no_components=32)
model.fit(interactions, user_features=user_feat, item_features=item_feat, epochs=30)
```

## Hybrid Strategies

| Strategy | Description | Use when |
|----------|-----------|---------|
| Weighted | Linear blend | Both signals available |
| Switching | Pick best signal | New vs returning users |
| Cascade | CF then content rerank | High precision needed |
| Mixed | Show both lists | Discovery mode |

## Learning Resources
- [LightFM docs](https://making.lyst.com/lightfm/docs/)
- [Netflix two-stage system](https://netflixtechblog.com/)

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
