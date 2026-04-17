# 3.2.4 Naive Bayes

Gaussian NB, Bernoulli NB, MultinomialNB, Laplace smoothing, text classification.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Bayes theorem by hand, class priors/likelihoods |
| `working_example2.py` | GaussianNB (Cal Housing), Laplace smoothing, MultinomialNB text |
| `working_example.ipynb` | Interactive: GaussianNB → Laplace → text classification |

## Quick Reference

```python
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Continuous features
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Text (sparse counts)
pipe = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=1.0))  # alpha=Laplace smoothing
pipe.fit(texts_train, y_train)
```

## Assumptions
- Features are **conditionally independent** given class label
- GaussianNB: features ~ N(μ_k, σ_k²) per class
- MultinomialNB: word counts follow multinomial distribution

## Learning Resources
- [StatQuest: Naive Bayes](https://youtu.be/O2L2Uv9pdDA)
- [sklearn Naive Bayes guide](https://scikit-learn.org/stable/modules/naive_bayes.html)

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
