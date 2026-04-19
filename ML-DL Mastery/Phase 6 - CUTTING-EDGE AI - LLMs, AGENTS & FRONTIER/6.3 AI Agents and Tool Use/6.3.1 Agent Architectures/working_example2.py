"""
Working Example 2: Agent Architectures
Simulates a ReAct-style (Reason+Act) agent loop with a tool dispatcher
and trajectory logging.
Run: python working_example2.py
"""
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


# --- Simulated tools ---
def tool_search(query: str) -> str:
    results = {"weather": "Sunny, 22°C", "capital france": "Paris",
               "pi value": "3.14159", "population china": "1.4 billion"}
    for k, v in results.items():
        if k in query.lower():
            return v
    return "No result found."


def tool_calc(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return "Calc error"


TOOLS = {"search": tool_search, "calc": tool_calc}


def react_agent(goal: str, max_steps: int = 6):
    """Simulated ReAct agent loop."""
    trajectory = []
    # Scripted plan for demo purposes
    plan = [
        ("think", f"I need to find: {goal}"),
        ("act:search", goal),
        ("observe", None),
        ("think", "Now I have the info, checking calculation."),
        ("act:calc", "3 * 7"),
        ("observe", None),
        ("think", "Done."),
        ("finish", goal),
    ]
    result = None
    for step_type, arg in plan[:max_steps]:
        if step_type == "think":
            entry = {"step": step_type, "content": arg, "result": None}
        elif step_type.startswith("act:"):
            tool_name = step_type.split(":")[1]
            res = TOOLS.get(tool_name, lambda x: "unknown")(arg)
            entry = {"step": step_type, "content": arg, "result": res}
            result = res
        elif step_type == "observe":
            entry = {"step": step_type, "content": result, "result": None}
        else:
            entry = {"step": step_type, "content": arg, "result": None}
        trajectory.append(entry)
    return trajectory


def demo():
    print("=== Agent Architectures: ReAct Loop ===")
    trajectory = react_agent("capital france")
    for i, step in enumerate(trajectory):
        print(f"  Step {i+1} [{step['step']}]: {step['content']}"
              + (f" -> {step['result']}" if step["result"] else ""))

    # Visualise step-type distribution across many runs
    step_types = ["think", "act:search", "act:calc", "observe", "finish"]
    counts = np.array([3, 2, 1, 2, 1])  # from demo trajectory

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(step_types, counts, color=["#3498db", "#e74c3c", "#e67e22", "#2ecc71", "#9b59b6"])
    axes[0].set(xlabel="Step Type", ylabel="Count",
                title="ReAct Agent: Step Type Distribution")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Simulated: success rate vs max steps
    max_steps_range = range(1, 10)
    success_rate = [0.1, 0.3, 0.55, 0.72, 0.83, 0.89, 0.92, 0.94, 0.95]
    axes[1].plot(list(max_steps_range), success_rate, "o-", color="steelblue", lw=2)
    axes[1].set(xlabel="Max Steps Allowed", ylabel="Task Success Rate",
                title="Agent Performance vs Max Steps")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "agent_architectures.png", dpi=100)
    plt.close()
    print("  Saved agent_architectures.png")


if __name__ == "__main__":
    demo()
