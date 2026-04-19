"""
Working Example: Tool Use and Function Calling
Covers OpenAI/Anthropic function calling APIs, tool schemas,
parallel tool use, and structured output patterns.
"""
import os, json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_tool_use")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Function calling overview ----------------------------------------------
def function_calling_overview():
    print("=== Tool Use and Function Calling ===")
    print()
    print("  Function calling = structured mechanism for LLMs to invoke external tools")
    print("  Model emits JSON with function name + arguments; caller executes; result returned")
    print()
    print("  Providers and formats:")
    providers = [
        ("OpenAI",      "tools=[{type, function: {name, description, parameters}}]"),
        ("Anthropic",   "tools=[{name, description, input_schema}]; use_tools"),
        ("Gemini",      "tools=[FunctionDeclaration]; function_calling_config"),
        ("Ollama",      "OpenAI-compatible; most models via Hermes/Llama3.1"),
        ("LangChain",   ".bind_tools(tools); @tool decorator; uniform API"),
    ]
    for p, d in providers:
        print(f"  {p:<12} {d}")


# -- 2. Tool schema ------------------------------------------------------------
def tool_schema_demo():
    print("\n=== Tool Schema Definition ===")
    print()
    print("  OpenAI-compatible tool definition (JSON Schema):")
    tool_def = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'London, UK'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }
        }
    }
    print(json.dumps(tool_def, indent=2))
    print()

    # Simulate tool execution
    def get_weather(location: str, units: str = "celsius") -> dict:
        """Simulated weather API."""
        temps = {"London, UK": 15, "New York, US": 22, "Tokyo, JP": 28}
        temp  = temps.get(location, 20)
        if units == "fahrenheit":
            temp = temp * 9/5 + 32
        return {"location": location, "temperature": temp,
                "units": units, "condition": "partly cloudy"}

    # Simulate a model calling the tool
    tool_call = {"name": "get_weather", "arguments": '{"location": "London, UK", "units": "celsius"}'}
    args = json.loads(tool_call["arguments"])
    result = get_weather(**args)
    print(f"  Model tool call: {tool_call['name']}({args})")
    print(f"  Tool result: {result}")


# -- 3. Parallel tool use ------------------------------------------------------
def parallel_tool_use():
    print("\n=== Parallel Tool Use ===")
    print()
    print("  Modern models can call multiple tools in one response")
    print("  Execute concurrently -> lower latency")
    print()

    # Simulate parallel tool calls
    class ToolCall:
        def __init__(self, id, name, args):
            self.id   = id
            self.name = name
            self.args = args

    tool_calls = [
        ToolCall("call_1", "search_web",   {"query": "capital of France"}),
        ToolCall("call_2", "search_web",   {"query": "capital of Germany"}),
        ToolCall("call_3", "get_weather",  {"location": "Paris, FR"}),
    ]

    def search_web(query: str) -> str:
        results = {
            "capital of France":  "Paris is the capital of France.",
            "capital of Germany": "Berlin is the capital of Germany.",
        }
        return results.get(query, "No results found.")

    def get_weather(location: str) -> dict:
        return {"temperature": 18, "condition": "sunny"}

    print(f"  {'Call ID':<10} {'Tool':<14} {'Args':<35} {'Result'}")
    print(f"  {'-'*10} {'-'*14} {'-'*35} {'-'*30}")
    for tc in tool_calls:
        if tc.name == "search_web":
            result = search_web(**tc.args)
        else:
            result = str(get_weather(**tc.args))
        args_str = str(tc.args)
        print(f"  {tc.id:<10} {tc.name:<14} {args_str:<35} {result[:30]}")


# -- 4. Structured output ------------------------------------------------------
def structured_output():
    print("\n=== Structured Output (JSON Mode) ===")
    print()
    print("  Force LLM to output valid JSON matching a schema")
    print("  Two approaches:")
    print()
    print("  1. JSON mode (OpenAI):  response_format={type: json_object}")
    print("     - Guarantees valid JSON but any schema")
    print()
    print("  2. Structured outputs:  response_format={type: json_schema, json_schema: ...}")
    print("     - Guarantees schema compliance (constrained decoding)")
    print()

    schema = {
        "type": "object",
        "properties": {
            "summary":  {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence":{"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["summary", "sentiment", "confidence"],
        "additionalProperties": False,
    }

    # Simulated model output
    model_output = json.dumps({
        "summary":    "The product review was generally positive with minor complaints.",
        "sentiment":  "positive",
        "confidence": 0.85,
    }, indent=2)

    print(f"  Schema:\n{json.dumps(schema, indent=2)}")
    print(f"\n  Model output (guaranteed valid):\n{model_output}")

    # Validate
    import json as _json
    parsed = _json.loads(model_output)
    valid  = all(k in parsed for k in schema["required"])
    print(f"\n  Validated: {valid}")


if __name__ == "__main__":
    function_calling_overview()
    tool_schema_demo()
    parallel_tool_use()
    structured_output()
