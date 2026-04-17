# 6.3.2 Tool Use and Function Calling

Tool use allows LLMs to call external APIs, search engines, code interpreters, and databases by generating structured JSON function calls. OpenAI's function-calling API and Anthropic's tool-use specification formalise this pattern. This folder simulates a tool dispatcher, argument extraction, and multi-tool chaining pipelines.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Tool registry, JSON argument extraction sim, multi-tool chain trace, success rate chart |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `tool_use.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Function calling | Model outputs structured JSON to invoke a tool |
| Tool schema | JSON Schema describing tool name, description, params |
| Parallel tool calls | Multiple tools invoked in a single LLM turn |
| Tool result injection | Results fed back as context for final answer |
| Grounding | Reduces hallucination by anchoring to real data |

## Learning Resources

- OpenAI Function Calling docs
- Qin et al. *ToolLLM* (2023)
- Schick et al. *Toolformer* (2023)
