# 5.1.8 NLP Tasks

NER, POS tagging, text classification, sentiment analysis, QA, summarization.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | spaCy NER + dependency parsing |
| `working_example2.py` | Rule-based NER + multi-class classification benchmark |
| `working_example.ipynb` | Interactive: NER patterns → text classification |

## Quick Reference

```python
# spaCy NER
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino.")
for ent in doc.ents:
    print(ent.text, ent.label_)  # Apple Inc. ORG, Steve Jobs PERSON

# HuggingFace pipeline for QA
from transformers import pipeline
qa = pipeline("question-answering")
result = qa(question="Who founded Apple?",
            context="Apple was founded by Steve Jobs in 1976.")
print(result["answer"])  # Steve Jobs
```

## Task Taxonomy

| Task | Input | Output | Metric |
|------|-------|--------|--------|
| Text classification | Document | Label | Accuracy, F1 |
| NER | Sentence | Span+label | F1 (entity) |
| POS tagging | Sentence | Tag per token | Accuracy |
| QA (extractive) | Context+Q | Answer span | EM, F1 |
| Summarization | Long text | Summary | ROUGE |

## Learning Resources
- [spaCy docs](https://spacy.io/api)
- [HuggingFace NLP course](https://huggingface.co/learn/nlp-course/)

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
