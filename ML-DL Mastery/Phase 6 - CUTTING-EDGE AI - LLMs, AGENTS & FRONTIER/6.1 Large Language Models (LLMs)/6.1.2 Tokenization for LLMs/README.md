# 6.1.2 Tokenisation for LLMs

BPE, WordPiece, SentencePiece — vocabulary, encode/decode, fertility, special tokens.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | tiktoken / HuggingFace tokeniser examples |
| `working_example2.py` | BPE from scratch on mini corpus |
| `working_example.ipynb` | Interactive: BPE merges + fertility chart |

## Quick Reference

```python
# HuggingFace tokeniser
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello, world!")
print(tok.decode(ids))         # "Hello, world!"

# tiktoken (OpenAI)
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("Hello world"); print(ids)

# BPE core loop
def bpe_step(vocab):
    pairs = get_pairs(vocab)
    best  = pairs.most_common(1)[0][0]
    return merge(best, vocab), best

# Fertility = tokens / words (lower is better compression)
fertility = n_tokens / n_words
```

## Tokeniser Comparison

| Method | Vocab size | Splits at | Used by |
|--------|-----------|-----------|---------|
| BPE | 50k-100k | Byte pairs | GPT-2/3/4 |
| WordPiece | 30k | Subwords | BERT |
| SentencePiece | configurable | Unicode | LLaMA, Gemma |
| Character | 256 | Character | Byte-level |

## Learning Resources
- [BPE paper (Sennrich 2016)](https://arxiv.org/abs/1508.07909)
- [tiktoken](https://github.com/openai/tiktoken)

Inspect tokenization and sampling.

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
