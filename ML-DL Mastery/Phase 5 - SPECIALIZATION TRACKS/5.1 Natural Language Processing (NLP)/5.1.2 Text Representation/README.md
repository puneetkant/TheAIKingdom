# 5.1.2 Text Representation

Bag of Words, TF-IDF, N-grams, cosine similarity, sparse vs dense representation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | BoW matrix + vocabulary inspection |
| `working_example2.py` | BoW → TF-IDF → bigrams → save feature matrix |
| `working_example.ipynb` | Interactive: TF-IDF → cosine similarity matrix |

## Quick Reference

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),    # unigrams + bigrams
    sublinear_tf=True,     # log(TF) instead of raw count
    stop_words="english",
)
X = vectorizer.fit_transform(corpus)  # sparse matrix

sim = cosine_similarity(X[0], X)  # compare doc 0 to all
```

## TF-IDF Formula

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{df(t)}$$

| Representation | Dense | Semantic | Notes |
|----------------|-------|----------|-------|
| BoW | No | No | Simple, interpretable |
| TF-IDF | No | Weak | Sparse, fast |
| Word2Vec | Yes | Yes | Fixed 300d |
| BERT | Yes | Yes | Contextual |

## Learning Resources
- [sklearn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Understanding TF-IDF](https://www.tfidf.com/)

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
