# 5.1.9 NLP Tools and Libraries

NLTK, spaCy, sklearn, HuggingFace Transformers, Gensim — ecosystem overview and quick starts.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | spaCy pipeline demo |
| `working_example2.py` | Library comparison table, regex patterns, sklearn pipeline |
| `working_example.ipynb` | Interactive: library detection → sklearn NLP pipeline |

## Quick Reference

```python
# NLTK
import nltk; nltk.download("punkt", "stopwords", "wordnet")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# spaCy
import spacy; nlp = spacy.load("en_core_web_sm")
doc = nlp("Text to process."); [token.lemma_ for token in doc]

# sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])

# HuggingFace
from transformers import pipeline
clf = pipeline("text-classification")
```

## Library Comparison

| Library | Best For | Speed | Notes |
|---------|---------|-------|-------|
| NLTK | Learning, research | Slow | Many corpora |
| spaCy | Production pipelines | Fast | Industrial-grade |
| sklearn | ML workflows | Medium | Integrates with ML |
| HuggingFace | SOTA models | Slow (GPU) | 100k+ models |
| Gensim | Word2Vec, LDA | Fast | Topic modeling |

## Learning Resources
- [spaCy 101](https://spacy.io/usage/spacy-101)
- [NLTK book](https://www.nltk.org/book/)
- [HuggingFace course](https://huggingface.co/learn/nlp-course/)

Process text and build simple NLP pipelines.

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
