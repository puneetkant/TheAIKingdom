"""
Working Example: Agent Frameworks
Covers LangChain, LangGraph, CrewAI, AutoGen, and patterns
for building production-grade agents.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_frameworks")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Framework comparison ---------------------------------------------------
def framework_comparison():
    print("=== Agent Frameworks Comparison ===")
    print()
    frameworks = [
        ("LangChain",   "Chains, agents, tools; LCEL; massive ecosystem; verbose"),
        ("LangGraph",   "Stateful graph-based; LangChain extension; best for complex flows"),
        ("CrewAI",      "Role-based crews; sequential/hierarchical; beginner-friendly"),
        ("AutoGen",     "Microsoft; multi-agent conversations; GroupChat; research-friendly"),
        ("Swarm (OAI)", "Lightweight; handoffs; stateless; educational"),
        ("Haystack",    "NLP/RAG focused; pipeline-first; production-grade"),
        ("Semantic Kernel","Microsoft; C#/Python; plugins; enterprise"),
        ("Pydantic AI",  "Type-safe; Pydantic models; production-focused"),
    ]
    print(f"  {'Framework':<18} {'Notes'}")
    for f, d in frameworks:
        print(f"  {f:<18} {d}")


# -- 2. LangGraph pattern ------------------------------------------------------
def langgraph_pattern():
    print("\n=== LangGraph: Stateful Agent Graphs ===")
    print()
    code = '''
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages:   Annotated[list, operator.add]   # append-only
    tool_calls: list
    final_answer: str | None

# Define nodes
def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def run_tools(state: AgentState) -> AgentState:
    results = [execute_tool(tc) for tc in state["tool_calls"]]
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"    # route to tool node
    return "end"          # route to END

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent",  call_llm)
graph.add_node("tools",  run_tools)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue,
                             {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")   # loop back after tools

app = graph.compile()

# Run
result = app.invoke({
    "messages": [("user", "What is the weather in Paris?")]
})
'''
    print(code)
    print("  LangGraph strengths:")
    strengths = [
        "Cycles and loops (not just DAGs)",
        "Persistent state across turns (checkpointing)",
        "Human-in-the-loop (interrupt nodes)",
        "First-class streaming",
        "Built-in debugging with LangSmith",
    ]
    for s in strengths:
        print(f"  • {s}")


# -- 3. CrewAI pattern ---------------------------------------------------------
def crewai_pattern():
    print("\n=== CrewAI: Role-Based Crews ===")
    print()
    code = '''
from crewai import Agent, Task, Crew, Process

# Define specialised agents
researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information about {topic}",
    backstory="Expert at web research and synthesis.",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm="gpt-4o",
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging reports based on research",
    backstory="Skilled at turning research into clear prose.",
    llm="gpt-4o-mini",
)

# Define tasks
research_task = Task(
    description="Research recent AI developments in {topic}",
    agent=researcher,
    expected_output="Bullet-point summary of key developments",
)

writing_task = Task(
    description="Write a 500-word blog post based on the research",
    agent=writer,
    context=[research_task],    # depends on research output
    expected_output="Polished blog post in markdown",
)

# Assemble crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "LLM reasoning"})
'''
    print(code)


# -- 4. Building production agents ---------------------------------------------
def production_agents():
    print("=== Production Agent Best Practices ===")
    print()
    practices = [
        ("Timeout limits",    "Every tool call has a max wait; no infinite loops"),
        ("Step budget",       "Max N tool calls per task; prevent runaway agents"),
        ("Error handling",    "Catch tool failures; retry with backoff; graceful fallback"),
        ("Cost tracking",     "Track tokens per run; alert on expensive tasks"),
        ("Tracing",           "Log every LLM call, tool call, decision (LangSmith/Phoenix)"),
        ("Human-in-the-loop", "Interrupt for high-stakes actions; approval gates"),
        ("Idempotency",       "Tools should be safe to retry (read-preferred, not write)"),
        ("Sandboxing",        "Execute code in isolated env (E2B, Modal, Docker)"),
        ("Testing",           "Unit test tools; integration test agent flows; evals"),
    ]
    print(f"  {'Practice':<24} {'Notes'}")
    for p, d in practices:
        print(f"  {p:<24} {d}")

    print()
    print("  Evaluation:")
    evals = [
        ("Task success rate",    "% tasks completed correctly end-to-end"),
        ("Step efficiency",      "Avg tool calls per success; fewer = better"),
        ("Error recovery",       "Can agent recover from tool failures?"),
        ("Hallucination rate",   "Does agent make up tool arguments?"),
        ("AgentBench/WebArena",  "Standardised benchmarks; web/OS/code tasks"),
    ]
    for e, d in evals:
        print(f"  {e:<24} {d}")


if __name__ == "__main__":
    framework_comparison()
    langgraph_pattern()
    crewai_pattern()
    production_agents()
