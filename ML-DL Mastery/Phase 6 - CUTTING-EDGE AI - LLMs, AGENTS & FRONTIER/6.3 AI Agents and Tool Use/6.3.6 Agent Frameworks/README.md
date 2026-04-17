# 6.3.6 Agent Frameworks

Agent frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, LangGraph) provide abstractions for building production-grade agentic pipelines with routing, memory, tool integration, and multi-agent coordination. This folder benchmarks framework design patterns: single-agent, hierarchical, and collaborative multi-agent task decomposition.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Agent framework pattern simulator, task decomposition tree, framework latency comparison |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `agent_frameworks.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| LangChain | Chains, agents, and tool integrations |
| LangGraph | Graph-based stateful agent workflows |
| AutoGen | Multi-agent conversation framework |
| CrewAI | Role-based collaborative agent teams |
| DAG orchestration | Directed acyclic graph of agent tasks |

## Learning Resources

- LangChain documentation
- Microsoft AutoGen (2023)
- LangGraph state machine docs
