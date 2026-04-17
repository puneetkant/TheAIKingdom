# 6.3.1 Agent Architectures

AI agents combine an LLM reasoning core with memory, tools, and an action loop. Key architectures include ReAct (Reason + Act), Reflexion (self-reflection), Plan-and-Execute, and Tree-of-Thought. This folder implements a minimal ReAct-style loop in pure Python, simulating thought-action-observation traces and success rates.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | ReAct trace simulator, action success heatmap, architecture comparison bar chart |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `agent_architectures.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| ReAct | Interleave reasoning traces with tool actions |
| Reflexion | Agent critiques its own outputs and retries |
| Plan-and-Execute | Planner decomposes task; executor runs sub-tasks |
| Tree-of-Thought | Explore branching reasoning paths with search |
| Observation loop | Thought → Action → Observation → repeat |

## Learning Resources

- Yao et al. *ReAct* (2022)
- Shinn et al. *Reflexion* (2023)
- Wei et al. *Chain-of-Thought* (2022)
