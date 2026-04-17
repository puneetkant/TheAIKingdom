# 5.1.7 Pre-trained Language Models

BERT, GPT-2, T5, DistilBERT. Fine-tuning, feature extraction, zero-shot with HuggingFace.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | BERT tokenizer output inspection |
| `working_example2.py` | HuggingFace sentiment pipeline (falls back to sklearn) |
| `working_example.ipynb` | Interactive: pipeline demo → fine-tuning pattern |

## Quick Reference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("I love NLP!", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred = logits.argmax(-1).item()  # 0=NEG, 1=POS

# Fine-tuning pattern
from transformers import Trainer, TrainingArguments
args = TrainingArguments("output", num_train_epochs=3, per_device_train_batch_size=16)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
```

## Model Family

| Model | Architecture | Pre-train task | Best for |
|-------|-------------|----------------|---------|
| BERT | Encoder | MLM + NSP | Classification, NER |
| GPT-2 | Decoder | CLM | Text generation |
| T5 | Enc-Dec | Span masking | Seq2seq tasks |
| DistilBERT | Encoder | Knowledge distillation | Fast inference |

## Learning Resources
- [HuggingFace docs](https://huggingface.co/docs/transformers/)
- [BERT paper](https://arxiv.org/abs/1810.04805)

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
