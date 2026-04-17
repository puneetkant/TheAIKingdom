# 5.6.1 Content-Based Filtering

Item profiles, TF-IDF features, cosine similarity, user profiles, cold-start advantage.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Genre-based item similarity |
| `working_example2.py` | TF-IDF movie profiles + user preference vector → recommendations |
| `working_example.ipynb` | Interactive: content similarity matrix |

## Quick Reference

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Build item profiles
tfidf = TfidfVectorizer()
item_matrix = tfidf.fit_transform(item_descriptions)   # (n_items, n_terms)

# Item-item similarity
sim_matrix = cosine_similarity(item_matrix)            # (n_items, n_items)

# Recommend for a query item
query_sims = sim_matrix[query_idx]
top_k = np.argsort(query_sims)[::-1][1:k+1]

# User profile = weighted avg of liked items
user_vec = item_matrix[liked_ids].mean(axis=0)
user_sims = cosine_similarity(user_vec, item_matrix).ravel()
```

## Pros and Cons

| Aspect | Detail |
|--------|--------|
| Pros | Handles cold-start (new users), no data sparsity |
| Cons | Feature engineering needed, filter bubble |
| Best for | News, articles, job postings |

## Learning Resources
- [Google RecSys Crash Course](https://developers.google.com/machine-learning/recommendation)

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
