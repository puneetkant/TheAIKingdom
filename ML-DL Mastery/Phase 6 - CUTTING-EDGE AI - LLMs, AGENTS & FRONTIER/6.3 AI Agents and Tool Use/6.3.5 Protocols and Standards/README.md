# 6.3.5 Protocols and Standards

Open protocols standardise how agents, tools, and models communicate. The Model Context Protocol (MCP) defines a JSON-RPC interface between host applications and AI models. OpenAPI / Swagger enables LLMs to call REST APIs. This folder simulates a minimal MCP-style message dispatcher and measures latency vs payload size trade-offs.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | MCP-style JSON-RPC message sim, latency vs payload chart, protocol throughput comparison |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `protocols_standards.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| MCP | Model Context Protocol; standardised host↔tool interface |
| JSON-RPC | Lightweight RPC over JSON; stateless |
| OpenAPI | REST API schema; enables LLM plugin calling |
| SSE | Server-Sent Events; streaming from server to client |
| Tool manifest | Declares tool capabilities to an LLM host |

## Learning Resources

- Anthropic Model Context Protocol spec
- OpenAI Plugin / GPT Actions documentation
- OpenAPI Specification 3.1
