# 4.4.3 RNN Applications

Sequence classification, time series forecasting, seq2seq encoder-decoder sketch.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Character-level language model sketch |
| `working_example2.py` | Sin vs Cos sequence classifier — RNN trained with output-only backprop |
| `working_example.ipynb` | Interactive: make sequences → RNN inference → accuracy |

## Quick Reference

```python
import torch, torch.nn as nn

# Many-to-one (classification)
class RNNClassifier(nn.Module):
    def __init__(self, inp, hid, n_class):
        super().__init__()
        self.rnn = nn.GRU(inp, hid, batch_first=True)
        self.fc  = nn.Linear(hid, n_class)
    def forward(self, x):
        _, h = self.rnn(x)   # h: (1, B, hid)
        return self.fc(h.squeeze(0))
```

## RNN Application Types

| Type | Example | Output |
|------|---------|--------|
| Many-to-one | Sentiment classification | Single label |
| Many-to-many | NER tagging | Label per token |
| Seq2seq | Machine translation | Variable-length output |
| One-to-many | Image captioning | Caption tokens |

## Learning Resources
- [Sequence modelling (PyTorch tutorial)](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [Seq2seq paper (Sutskever 2014)](https://arxiv.org/abs/1409.3215)

Work with sequence data and recurrent models.

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
