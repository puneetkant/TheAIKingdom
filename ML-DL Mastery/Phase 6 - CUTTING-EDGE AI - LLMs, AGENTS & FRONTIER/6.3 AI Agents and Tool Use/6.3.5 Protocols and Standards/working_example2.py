"""
Working Example 2: Protocols and Standards
Demonstrates OpenAI-style function call JSON schema generation and
MCP (Model Context Protocol) tool definition validation.
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


def make_openai_schema(name, description, params):
    """Generate OpenAI function calling JSON schema."""
    properties = {}
    required = []
    for param, meta in params.items():
        properties[param] = {"type": meta["type"]}
        if "description" in meta:
            properties[param]["description"] = meta["description"]
        if meta.get("enum"):
            properties[param]["enum"] = meta["enum"]
        if meta.get("required", True):
            required.append(param)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def validate_tool_call(schema, args):
    """Validate a tool call against its schema."""
    errors = []
    required = schema["parameters"].get("required", [])
    properties = schema["parameters"].get("properties", {})
    for param in required:
        if param not in args:
            errors.append(f"Missing required: {param}")
    for param, val in args.items():
        if param not in properties:
            errors.append(f"Unknown param: {param}")
        elif properties[param]["type"] == "integer" and not isinstance(val, int):
            errors.append(f"Type error: {param} must be integer")
        elif properties[param]["type"] == "string" and not isinstance(val, str):
            errors.append(f"Type error: {param} must be string")
        elif "enum" in properties[param] and val not in properties[param]["enum"]:
            errors.append(f"Enum error: {param}={val!r} not in {properties[param]['enum']}")
    return errors


def demo():
    print("=== Protocols and Standards: Function Call Schemas ===")

    # Define tools
    tools_def = [
        ("get_weather", "Get weather for a city", {
            "city": {"type": "string", "description": "City name", "required": True},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "required": False},
        }),
        ("search_web", "Search the internet", {
            "query": {"type": "string", "description": "Search query", "required": True},
            "max_results": {"type": "integer", "description": "Max results", "required": False},
        }),
        ("create_file", "Create a file", {
            "path": {"type": "string", "description": "File path", "required": True},
            "content": {"type": "string", "description": "File content", "required": True},
        }),
    ]

    schemas = [make_openai_schema(n, d, p) for n, d, p in tools_def]
    for s in schemas:
        print(f"\n  Schema: {s['name']}")
        print(f"    {json.dumps(s, indent=4)[:200]}...")

    # Test validation
    test_calls = [
        (schemas[0], {"city": "Paris", "unit": "celsius"}, "Valid"),
        (schemas[0], {"unit": "celsius"}, "Missing required"),
        (schemas[0], {"city": "Paris", "unit": "kelvin"}, "Enum violation"),
        (schemas[1], {"query": "machine learning", "max_results": 5}, "Valid"),
        (schemas[1], {"query": "ml", "max_results": "five"}, "Type error"),
    ]

    results = []
    for schema, args, expected in test_calls:
        errors = validate_tool_call(schema, args)
        valid = len(errors) == 0
        results.append({"fn": schema["name"], "valid": valid, "errors": errors, "expected": expected})
        status = "PASS" if (valid == (expected == "Valid")) else "FAIL"
        print(f"  [{status}] {schema['name']}({list(args.keys())}): {errors or 'OK'}")

    # Visualise validation summary
    valid_counts = sum(1 for r in results if r["valid"])
    error_counts = len(results) - valid_counts

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].pie([valid_counts, error_counts],
                labels=["Valid Calls", "Invalid Calls"],
                colors=["mediumseagreen", "tomato"],
                autopct="%1.0f%%", startangle=90)
    axes[0].set_title("Tool Call Validation Results")

    # Schema parameter counts
    fn_names = [s["name"] for s in schemas]
    total_p = [len(s["parameters"]["properties"]) for s in schemas]
    req_p = [len(s["parameters"].get("required", [])) for s in schemas]
    x = np.arange(len(fn_names))
    axes[1].bar(x - 0.2, total_p, 0.4, label="Total Params", color="steelblue")
    axes[1].bar(x + 0.2, req_p, 0.4, label="Required", color="darkorange")
    axes[1].set(xticks=x, xticklabels=fn_names, ylabel="Count",
                title="Function Schema Complexity")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "protocols_standards.png", dpi=100)
    plt.close()
    print("  Saved protocols_standards.png")


if __name__ == "__main__":
    demo()
