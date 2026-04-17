# 5.4.5 Autoregressive Models for Generation

GPT, WaveNet, PixelCNN, autoregressive factorisation, temperature, top-k/p sampling.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Token-level autoregressive LM |
| `working_example2.py` | Character bigram/trigram model + perplexity + temperature sampling |
| `working_example.ipynb` | Interactive: bigram model + temperature comparison |

## Quick Reference

```python
# Autoregressive factorisation
p(x) = prod_{t} p(x_t | x_1, ..., x_{t-1})

# GPT-style training (next-token prediction)
logits = model(input_ids)            # (B, T, vocab)
loss   = F.cross_entropy(logits[:, :-1].reshape(-1, V), input_ids[:, 1:].reshape(-1))

# Temperature sampling
probs = softmax(logits / temperature)
next_token = torch.multinomial(probs, 1)

# Top-k sampling
top_k = 50
indices_to_remove = logits < logits.topk(top_k)[0][..., -1, None]
logits[indices_to_remove] = -inf

# Top-p (nucleus) sampling
sorted_probs, _ = torch.sort(softmax(logits), descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)
remove = cumulative > p  # p=0.9
```

## Sampling Strategy Comparison

| Strategy | Diversity | Coherence | Use case |
|----------|-----------|-----------|---------|
| Greedy | Low | High | Short answers |
| Temperature | Medium | Medium | Creative |
| Top-k | Medium | High | General |
| Top-p | High | Good | LLMs |
| Beam search | Low | Very high | Translation |

## Learning Resources
- [GPT-2 paper](https://openai.com/blog/better-language-models/)
- [Karpathy makemore](https://github.com/karpathy/makemore)

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
