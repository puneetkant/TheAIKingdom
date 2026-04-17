# 5.1.4 Subword Tokenization

Byte-Pair Encoding (BPE), WordPiece, SentencePiece. Handles OOV and morphology.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | WordPiece tokenization concept |
| `working_example2.py` | BPE from scratch: merge algorithm → subword vocabulary |
| `working_example.ipynb` | Interactive: BPE merge steps trace |

## Quick Reference

```python
# Using HuggingFace tokenizers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello, world! This is NLP.")
print(tokens["input_ids"])
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
# ['[CLS]', 'hello', ',', 'world', '!', 'this', 'is', 'nl', '##p', '.', '[SEP]']

# Training a new tokenizer
from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer()
tok.train(["corpus.txt"], vocab_size=30000, min_frequency=2)
```

## Algorithm Comparison

| Algorithm | Used by | Strategy |
|-----------|---------|---------|
| BPE | GPT-2, RoBERTa | Merge most frequent pair |
| WordPiece | BERT | Maximize likelihood |
| SentencePiece | T5, LLaMA | Language-agnostic BPE |
| Unigram | XLNet | Prune vocabulary |

## Learning Resources
- [BPE paper (Sennrich 2016)](https://arxiv.org/abs/1508.07909)
- [HuggingFace tokenizers](https://huggingface.co/docs/tokenizers/)

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
