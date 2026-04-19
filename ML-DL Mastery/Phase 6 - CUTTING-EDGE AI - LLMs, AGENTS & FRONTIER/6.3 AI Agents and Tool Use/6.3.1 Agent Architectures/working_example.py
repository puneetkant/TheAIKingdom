"""
Working Example: AI Agent Architectures
Covers ReAct, Reflexion, Plan-and-Execute, AutoGPT-style,
and multi-agent patterns.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_agent_arch")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. What is an AI agent? ---------------------------------------------------
def agent_overview():
    print("=== AI Agent Architectures ===")
    print()
    print("  Agent = LLM + Memory + Tools + Execution Loop")
    print()
    print("  Core agent loop:")
    print("    1. Observe: receive goal / environment state")
    print("    2. Think:   LLM generates reasoning and action plan")
    print("    3. Act:     execute tool calls")
    print("    4. Update:  integrate results into context")
    print("    5. Repeat until task complete or budget exhausted")
    print()
    print("  Agent taxonomy:")
    agent_types = [
        ("Reactive",        "Direct perception -> action; no planning; simple"),
        ("Deliberative",    "World model; plan then act; classical AI"),
        ("ReAct",           "Reason + Act interleaved; LLM-native"),
        ("Plan-Execute",    "High-level plan first; then subagents execute"),
        ("Multi-agent",     "Specialist subagents; orchestrator coordinates"),
        ("Hierarchical",    "Tree of agents; top-down delegation"),
    ]
    for t, d in agent_types:
        print(f"  {t:<18} {d}")


# -- 2. ReAct ------------------------------------------------------------------
def react_pattern():
    print("\n=== ReAct (Reasoning + Acting) ===")
    print()
    print("  Yao et al. 2022 — interleave thought and action traces")
    print()
    print("  Trace format:")
    react_trace = """
Question: Who is the current CEO of the company that makes the iPhone?

Thought 1: I need to find out who makes the iPhone, and then find their CEO.
Action 1: Search["iPhone manufacturer"]
Observation 1: Apple Inc. manufactures the iPhone.

Thought 2: Now I need to find the current CEO of Apple Inc.
Action 2: Search["Apple CEO 2024"]
Observation 2: Tim Cook has been CEO of Apple since 2011.

Thought 3: I now have the answer.
Action 3: Finish["Tim Cook is the CEO of Apple Inc."]
"""
    print(react_trace)
    print("  Benefits of ReAct:")
    print("  • Transparent reasoning trace (explainable)")
    print("  • Handles multi-step tasks requiring external information")
    print("  • Handles errors by updating beliefs from observations")


# -- 3. Reflexion --------------------------------------------------------------
def reflexion_pattern():
    print("\n=== Reflexion ===")
    print()
    print("  Shinn et al. 2023 — agents that learn from past failures")
    print()
    print("  Key idea: store verbal reflection after failure -> guide future attempts")
    print()
    print("  Algorithm:")
    steps = [
        ("1. Trial",       "Run task; receive outcome (success/fail)"),
        ("2. Evaluate",    "Check if task succeeded; extract failure signal"),
        ("3. Reflect",     "LLM writes verbal summary of what went wrong"),
        ("4. Add to memory","Reflection added to long-term memory store"),
        ("5. Retry",       "Next attempt uses reflections as context"),
    ]
    for s, d in steps:
        print(f"  {s:<14} {d}")
    print()
    print("  Example reflection:")
    reflection = """
[Previous attempt]
  Task: Sort a list in Python. Submitted: list.sorted() -> AttributeError

[Reflection]
  list.sorted() is wrong. The correct methods are:
  - sorted(list) returns a new sorted list
  - list.sort() sorts in-place
  I should use sorted(lst) when I need the original unchanged.

[Next attempt]
  Used sorted(numbers) -> correct output
"""
    print(reflection)


# -- 4. Multi-agent patterns ---------------------------------------------------
def multi_agent():
    print("\n=== Multi-Agent Patterns ===")
    print()
    print("  Why multi-agent?")
    print("  • Parallelism: run subtasks concurrently")
    print("  • Specialisation: coding agent, research agent, critic agent")
    print("  • Context window: each agent has fresh context")
    print()
    print("  Topology patterns:")
    topologies = [
        ("Orchestrator-worker",   "Orchestrator decomposes task; workers execute"),
        ("Pipeline",              "Agent A output -> Agent B -> Agent C (sequential)"),
        ("Debate",                "Multiple agents argue positions; judge decides"),
        ("Ensemble",              "N agents vote or merge outputs"),
        ("Hierarchical",          "Manager -> team leads -> workers (tree)"),
        ("Peer-to-peer",          "Agents communicate directly via shared state"),
    ]
    for t, d in topologies:
        print(f"  {t:<24} {d}")
    print()
    print("  Frameworks:")
    frameworks = [
        ("LangGraph",    "Graph-based; stateful; multi-agent; cycles allowed"),
        ("AutoGen",      "Microsoft; multi-agent conversation; flexible"),
        ("CrewAI",       "Role-based crews; sequential/hierarchical; popular"),
        ("Swarm (OpenAI)","Lightweight; handoff pattern; educational"),
        ("CAMEL",        "Communicative agents; role-play; research"),
    ]
    for f, d in frameworks:
        print(f"  {f:<16} {d}")

    print()
    print("  Challenges:")
    challenges = [
        "Error propagation: one agent's mistake cascades",
        "Cost: multiple LLM calls per task; expensive",
        "Latency: sequential agents slow for real-time",
        "Coordination: who decides when to hand off?",
        "Eval: hard to evaluate composite multi-agent systems",
    ]
    for c in challenges:
        print(f"  • {c}")


if __name__ == "__main__":
    agent_overview()
    react_pattern()
    reflexion_pattern()
    multi_agent()
