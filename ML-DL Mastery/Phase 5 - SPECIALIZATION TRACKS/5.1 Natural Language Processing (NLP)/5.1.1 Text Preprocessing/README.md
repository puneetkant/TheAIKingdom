# 5.1.1 Text Preprocessing

Cleaning, tokenization, stopword removal, stemming/lemmatization pipeline for NLP.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Regex-based pipeline with frequency analysis |
| `working_example2.py` | Full pipeline: clean → tokenize → filter → stem → term freq |
| `working_example.ipynb` | Interactive: corpus cleaning → token frequency |

## Quick Reference

```python
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)     # remove punctuation
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw]
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]
```

## Pipeline Steps

| Step | Purpose | Tool |
|------|---------|------|
| Lowercasing | Normalise case | `str.lower()` |
| Punctuation removal | Reduce noise | `re.sub` |
| Tokenization | Split into words | `nltk.word_tokenize` |
| Stopword removal | Remove function words | `nltk.corpus.stopwords` |
| Stemming | Reduce to root form | `PorterStemmer` |
| Lemmatization | Linguistic root | `WordNetLemmatizer` |

## Learning Resources
- [NLTK book](https://www.nltk.org/book/)
- [spaCy docs](https://spacy.io/usage/linguistic-features)

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
