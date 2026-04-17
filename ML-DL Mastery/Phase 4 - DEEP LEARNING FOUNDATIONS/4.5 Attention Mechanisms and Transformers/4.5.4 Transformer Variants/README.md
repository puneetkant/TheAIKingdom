# 4.5.4 Transformer Variants

BERT (encoder-only), GPT (decoder-only), T5/BART (encoder-decoder) — architecture families.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | HuggingFace quick inference demo |
| `working_example2.py` | Architecture comparison table + causal/bidi mask demo |
| `working_example.ipynb` | Interactive: comparison table → mask visualisation |

## Architecture Families

| Family | Attention | Pre-training | Best For |
|--------|-----------|-------------|----------|
| Encoder-only (BERT) | Bidirectional | MLM | Classification, NER, QA |
| Decoder-only (GPT) | Causal | CLM | Generation, chat, code |
| Encoder-Decoder (T5) | Both | Span masking | Seq2seq tasks |

## Quick Reference

```python
from transformers import pipeline

# BERT: text classification
clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
print(clf("I love this!"))

# GPT: text generation
gen = pipeline("text-generation", model="gpt2")
print(gen("Once upon a time", max_length=30))
```

## Learning Resources
- [BERT paper (Devlin 2018)](https://arxiv.org/abs/1810.04805)
- [GPT-3 paper (Brown 2020)](https://arxiv.org/abs/2005.14165)
- [HuggingFace model hub](https://huggingface.co/models)

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
