# 5.1.6 Neural NLP Models

TextCNN, GRU/LSTM text classifiers, attention pooling. Neural baselines before Transformers.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | TextCNN with max-over-time pooling (PyTorch) |
| `working_example2.py` | MLP depth comparison: 1/2/3-layer on 20 newsgroups |
| `working_example.ipynb` | Interactive: TF-IDF + MLP classifier |

## Quick Reference

```python
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)           # (B, E, L)
        x = [F.relu(conv(x)).max(dim=2).values      # max-over-time
             for conv in self.convs]
        return self.fc(torch.cat(x, dim=1))
```

## Model Family

| Model | Input | Key idea |
|-------|-------|---------|
| TextCNN | Token IDs | Local n-gram filters |
| Bi-LSTM | Sequence | Long-range context |
| Attention LSTM | Sequence | Weighted pooling |
| Transformer | Sequence | Full attention |

## Learning Resources
- [TextCNN paper (Kim 2014)](https://arxiv.org/abs/1408.5882)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

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
