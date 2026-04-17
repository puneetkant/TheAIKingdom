# 4.5.3 Efficient Attention Variants

Linear attention O(n), sliding window O(n·w), FlashAttention — solving the O(n²) bottleneck.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Memory complexity comparison |
| `working_example2.py` | Full / Linear / Window attention timing across sequence lengths |
| `working_example.ipynb` | Interactive: complexity ops plot → timing comparison |

## Complexity Comparison

| Variant | Time | Memory | Model |
|---------|------|--------|-------|
| Full attention | O(n²) | O(n²) | BERT, GPT |
| Sliding window | O(n·w) | O(n·w) | Longformer |
| Linear attention | O(n) | O(n) | Performer |
| FlashAttention | O(n²) time, O(n) mem | IO-aware | GPT-4, LLaMA |

## Quick Reference

```python
# PyTorch 2.0 FlashAttention (automatic)
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    out = F.scaled_dot_product_attention(Q, K, V)

# Longformer-style window
from transformers import LongformerModel
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
```

## Learning Resources
- [FlashAttention (Dao 2022)](https://arxiv.org/abs/2205.14135)
- [Longformer (Beltagy 2020)](https://arxiv.org/abs/2004.05150)

Inspect attention and transformer architecture.

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
