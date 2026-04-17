# 5.1.5 Classical NLP Models

Naive Bayes, Logistic Regression, SVM for text classification. Probabilistic and discriminative baselines.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Naive Bayes sentiment analysis with smoothing |
| `working_example2.py` | NB vs LR vs LinearSVC on 20 newsgroups — accuracy comparison |
| `working_example.ipynb` | Interactive: TF-IDF + NB → LR accuracy comparison |

## Quick Reference

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2))),
    ("clf",   LinearSVC(C=0.5, max_iter=2000)),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

## Model Comparison

| Model | Strengths | Weaknesses |
|-------|-----------|-----------|
| Naive Bayes | Fast, probabilistic | Assumes independence |
| Logistic Regression | Interpretable, sparse | Linear decision boundary |
| LinearSVC | High accuracy, scalable | No probability output |
| Gradient Boosting | Strong baseline | Slow on large vocab |

## Learning Resources
- [sklearn text classification tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Naive Bayes for text](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)

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
