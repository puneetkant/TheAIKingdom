# 6.1.6 Context Length Extension

Extending a transformer's context window beyond its training length requires careful handling of positional encodings. Techniques like RoPE scaling, ALiBi, YaRN, and positional interpolation allow models pre-trained on 4 k tokens to handle 32 k–1 M tokens at inference. This folder visualises positional encoding similarity matrices and attention decay patterns.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | RoPE vs ALiBi positional bias comparison, attention decay curves, PE similarity heatmap |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `context_length.png`, `pe_similarity.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| RoPE | Rotary position embedding; relative positions via rotation |
| ALiBi | Attention with linear biases; length-generalising |
| YaRN | Yet Another RoPE extensioN; NTK-aware interpolation |
| Positional interpolation | Scale existing RoPE freqs to longer context |
| Lost-in-the-middle | Models attend poorly to middle of long contexts |

## Learning Resources

- Su et al. *RoFormer* (2021)
- Press et al. *ALiBi* (2022)
- Peng et al. *YaRN* (2023)
