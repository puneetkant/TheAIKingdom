# 6.1.4 Key LLM Families

The modern LLM landscape spans encoder-only models (BERT), decoder-only models (GPT family, LLaMA, Mistral, Gemma), and encoder-decoder models (T5, BART). Each family has distinct architectural choices, pre-training objectives, and strengths. This folder visualises family relationships, parameter counts, and benchmark comparisons.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Bubble chart of model families by params & benchmark score, architecture comparison bar chart |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `llm_families.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Decoder-only | Autoregressive generation; GPT, LLaMA, Mistral |
| Encoder-only | Bidirectional; BERT, RoBERTa; suited for classification |
| Encoder-decoder | Seq2seq; T5, BART; suited for translation/summarisation |
| MoE architecture | Sparse mixture of experts; Mixtral |
| RLHF alignment | Post-training alignment with human feedback |

## Learning Resources

- Touvron et al. *LLaMA 2* (2023)
- Jiang et al. *Mistral* (2023)
- Raffel et al. *T5* (2020)
