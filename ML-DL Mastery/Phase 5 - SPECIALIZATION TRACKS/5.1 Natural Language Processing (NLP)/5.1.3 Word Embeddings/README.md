# 5.1.3 Word Embeddings

Word2Vec (CBOW/Skip-gram), GloVe, FastText. Dense word representations with semantic properties.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | GloVe-like co-occurrence matrix |
| `working_example2.py` | CBOW from scratch → PCA 2D plot → nearest neighbours |
| `working_example.ipynb` | Interactive: co-occurrence matrix → nearest neighbour lookup |

## Quick Reference

```python
# Using gensim Word2Vec
from gensim.models import Word2Vec
tokenized = [sentence.split() for sentence in corpus]
model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1, epochs=10)
vector = model.wv["cat"]           # shape (100,)
model.wv.most_similar("cat", topn=5)
model.wv.similarity("cat", "dog")

# Using pre-trained GloVe (gensim-data)
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-50")
glove.most_similar(positive=["king","woman"], negative=["man"])  # → queen
```

## Model Comparison

| Model | Method | Advantages | Notes |
|-------|--------|-----------|-------|
| Word2Vec | Neural (CBOW/Skip-gram) | Fast, analogy tasks | OOV problem |
| GloVe | Co-occurrence matrix | Global statistics | OOV problem |
| FastText | Character n-grams | Handles OOV | Larger model |
| ELMo/BERT | Contextual | Polysemy-aware | Heavy |

## Learning Resources
- [Word2Vec explained](https://arxiv.org/abs/1301.3781)
- [GloVe paper](https://nlp.stanford.edu/projects/glove/)

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
