# 6.2.5 LLM Evaluation

Evaluating LLMs requires a multi-dimensional toolkit: perplexity for fluency, BLEU/ROUGE for overlap with references, LLM-as-judge for quality, and holistic benchmarks (MMLU, HumanEval, MT-Bench) for capability. This folder computes these metrics from scratch and visualises trade-offs between automatic and human evaluation.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Perplexity, BLEU-1/2, ROUGE-L, radar chart of model scores across benchmarks |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `llm_evaluation.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Perplexity | 2^(average NLL); lower = better language model fit |
| BLEU | n-gram precision vs reference; standard MT metric |
| ROUGE-L | Longest common subsequence; used for summarisation |
| MMLU | Massive Multitask Language Understanding benchmark |
| LLM-as-judge | Use GPT-4 / Claude to score outputs pair-wise |

## Learning Resources

- Liang et al. *HELM* (2022)
- Hendrycks et al. *MMLU* (2020)
- Zheng et al. *MT-Bench / Chatbot Arena* (2023)
