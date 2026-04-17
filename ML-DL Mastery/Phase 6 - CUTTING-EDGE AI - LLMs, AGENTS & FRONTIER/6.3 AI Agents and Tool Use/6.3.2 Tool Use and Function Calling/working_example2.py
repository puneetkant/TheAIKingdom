"""
Working Example 2: Tool Use and Function Calling
Demonstrates a function registry, schema validation, and simulated
tool call dispatch (JSON function calling format).
Run: python working_example2.py
"""
from pathlib import Path
import json

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


# --- Function registry ---
REGISTRY = {}


def register(name, description, params):
    def decorator(fn):
        REGISTRY[name] = {"fn": fn, "description": description, "params": params}
        return fn
    return decorator


@register("get_weather", "Get current weather for a city",
          {"city": {"type": "string", "required": True},
           "units": {"type": "string", "required": False, "default": "celsius"}})
def get_weather(city, units="celsius"):
    temps = {"london": 15, "paris": 18, "tokyo": 22, "sydney": 25}
    t = temps.get(city.lower(), 20)
    return {"city": city, "temp": t, "units": units}


@register("calculate", "Evaluate a math expression",
          {"expression": {"type": "string", "required": True}})
def calculate(expression):
    allowed = {k: v for k, v in vars(__builtins__).items()
               if k in {"abs", "round", "min", "max"}} if hasattr(__builtins__, "items") else {}
    try:
        return {"result": eval(expression, {"__builtins__": {}}, allowed)}  # noqa: S307
    except Exception as e:
        return {"error": str(e)}


@register("list_files", "List files in a directory",
          {"path": {"type": "string", "required": True},
           "extension": {"type": "string", "required": False}})
def list_files(path, extension=None):
    p = Path(path)
    if not p.exists():
        return {"files": [], "error": "path not found"}
    files = list(p.iterdir())
    if extension:
        files = [f for f in files if f.suffix == extension]
    return {"files": [f.name for f in files[:10]]}


def validate_call(name, args):
    if name not in REGISTRY:
        return False, f"Unknown function: {name}"
    schema = REGISTRY[name]["params"]
    for p, meta in schema.items():
        if meta.get("required") and p not in args:
            return False, f"Missing required param: {p}"
    return True, "OK"


def dispatch(name, args):
    valid, msg = validate_call(name, args)
    if not valid:
        return {"error": msg}
    fn = REGISTRY[name]["fn"]
    return fn(**args)


def demo():
    print("=== Tool Use and Function Calling ===")
    print(f"  Registered functions: {list(REGISTRY.keys())}")

    # Test calls
    calls = [
        ("get_weather", {"city": "Paris"}),
        ("get_weather", {"city": "Tokyo", "units": "fahrenheit"}),
        ("calculate", {"expression": "2 ** 10 + 42"}),
        ("calculate", {}),             # missing required param
        ("unknown_fn", {"x": 1}),     # unknown function
        ("list_files", {"path": str(OUTPUT.parent), "extension": ".py"}),
    ]

    results = []
    for name, args in calls:
        res = dispatch(name, args)
        results.append({"call": f"{name}({args})", "result": res})
        print(f"  {name}: {res}")

    # Visualise: validation success/failure
    success = [1 if "error" not in r["result"] else 0 for r in results]
    labels = [r["call"][:25] + "..." for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["mediumseagreen" if s else "tomato" for s in success]
    axes[0].barh(labels, success, color=colors)
    axes[0].set(xlabel="Success (1) / Failure (0)",
                title="Tool Call Validation Results")
    axes[0].set_xlim(0, 1.4)
    axes[0].grid(True, axis="x", alpha=0.3)

    # Schema complexity per function
    fn_names = list(REGISTRY.keys())
    n_params = [len(v["params"]) for v in REGISTRY.values()]
    n_required = [sum(1 for p in v["params"].values() if p.get("required"))
                  for v in REGISTRY.values()]
    x = np.arange(len(fn_names))
    axes[1].bar(x - 0.2, n_params, 0.4, label="Total Params", color="steelblue")
    axes[1].bar(x + 0.2, n_required, 0.4, label="Required", color="darkorange")
    axes[1].set(xticks=x, xticklabels=fn_names, ylabel="Count",
                title="Function Schema Complexity")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "tool_use.png", dpi=100)
    plt.close()
    print("  Saved tool_use.png")


if __name__ == "__main__":
    demo()
