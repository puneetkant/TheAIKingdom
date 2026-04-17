# 6.3.7 Prompt Engineering

Prompt engineering extracts maximum capability from frozen LLMs through careful input design. Core techniques include zero-shot, few-shot, chain-of-thought, self-consistency, and structured output prompting. This folder implements a prompt accuracy scorer, compares few-shot vs zero-shot performance, and visualises the accuracy–examples curve.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Few-shot accuracy simulation, chain-of-thought template, accuracy vs #examples curve |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `prompt_engineering.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Zero-shot | Task described in prompt; no examples |
| Few-shot | 1–32 (input, output) examples in the prompt |
| Chain-of-Thought | Elicit step-by-step reasoning before answer |
| Self-consistency | Sample multiple CoT paths; majority vote |
| System prompt | High-level persona and instructions for the model |

## Learning Resources

- Wei et al. *Chain-of-Thought* (2022)
- Wang et al. *Self-Consistency* (2022)
- DAIR.AI Prompt Engineering Guide
