# 6.5.4 Mechanistic Interpretability

Mechanistic interpretability (MI) reverse-engineers how neural networks implement algorithms internally. Key techniques include attention head analysis, activation patching (causal tracing), logit lens, and sparse autoencoders for finding interpretable features. This folder implements causal attention pattern analysis, activation patching, and logit-lens layer-by-layer prediction tracking.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Causal attention patterns, attention entropy, logit lens simulation, activation patching |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `mechanistic_interp.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Attention head analysis | What does each head attend to? induction, copying, etc. |
| Activation patching | Swap activations between runs; locate responsible components |
| Logit lens | Read off model’s predictions at each intermediate layer |
| Sparse autoencoder | Learn interpretable features from dense activations |
| Superposition | Model encodes more features than dimensions via interference |

## Learning Resources

- Elhage et al. *A Mathematical Framework for Transformer Circuits* (2021)
- Meng et al. *ROME* (2022)
- Anthropic *Scaling Monosemanticity* (2024)
