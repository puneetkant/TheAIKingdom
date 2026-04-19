"""
Working Example 2: Agent Frameworks
Minimal LangChain-style chain: prompt template -> LLM call stub -> output parser.
Demonstrates chain composition and pipeline patterns.
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


# --- Minimal chain components ---
class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, **kwargs):
        result = self.template
        for k, v in kwargs.items():
            result = result.replace(f"{{{k}}}", str(v))
        return result


class LLMStub:
    """Stub LLM: echo + simulated latency and token count."""
    def __init__(self, name="gpt-stub"):
        self.name = name
        self._call_log = []

    def call(self, prompt):
        # Simulated response based on prompt keywords
        responses = {
            "classify": "positive",
            "summarize": "A brief summary of the provided text.",
            "translate": "Translated text goes here.",
            "extract": "entity1, entity2, entity3",
        }
        for kw, resp in responses.items():
            if kw in prompt.lower():
                response = resp
                break
        else:
            response = f"[{self.name}] Response to: {prompt[:40]}..."
        self._call_log.append({"prompt_len": len(prompt), "response_len": len(response)})
        return response


class OutputParser:
    def __init__(self, mode="strip"):
        self.mode = mode

    def parse(self, text):
        if self.mode == "strip":
            return text.strip()
        elif self.mode == "list":
            return [item.strip() for item in text.split(",")]
        elif self.mode == "bool":
            return "positive" in text.lower()
        return text


class Chain:
    """LangChain-style sequential chain."""
    def __init__(self, steps):
        self.steps = steps  # list of (name, callable)

    def run(self, **kwargs):
        state = kwargs
        log = []
        for name, fn in self.steps:
            if isinstance(fn, PromptTemplate):
                out = fn.format(**state)
                state["prompt"] = out
            elif isinstance(fn, LLMStub):
                out = fn.call(state.get("prompt", ""))
                state["llm_output"] = out
            elif isinstance(fn, OutputParser):
                out = fn.parse(state.get("llm_output", ""))
                state["final"] = out
            else:
                out = fn(**state)
            log.append({"step": name, "output": str(out)[:60]})
        return state.get("final"), log


def demo():
    print("=== Agent Frameworks: Minimal LangChain-Style Chain ===")

    llm = LLMStub("gpt-stub")

    # Chain 1: Sentiment classification
    classify_chain = Chain([
        ("prompt_template", PromptTemplate(
            "Classify the sentiment of: '{text}'. Answer: classify this text.")),
        ("llm_call", llm),
        ("output_parser", OutputParser("bool")),
    ])

    # Chain 2: Entity extraction
    extract_chain = Chain([
        ("prompt_template", PromptTemplate("Extract entities from: '{text}'. extract them.")),
        ("llm_call", llm),
        ("output_parser", OutputParser("list")),
    ])

    texts = ["I love this product!", "The service was terrible.", "Machine learning is fascinating."]
    for text in texts:
        result, log = classify_chain.run(text=text)
        print(f"  Classify: '{text[:30]}...' -> {result}")

    result2, log2 = extract_chain.run(text="Apple CEO Tim Cook visited Paris.")
    print(f"\n  Extract entities: {result2}")
    print("\n  Chain execution log:")
    for entry in log2:
        print(f"    [{entry['step']}] {entry['output']}")

    # Visualise: chain step latencies (simulated), call log
    steps = ["PromptTemplate", "LLM Call", "OutputParser"]
    sim_latency_ms = [2, 850, 1]  # ms
    throughputs = [1000 / (l + 1) for l in sim_latency_ms]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(steps, sim_latency_ms, color=["#3498db", "#e74c3c", "#2ecc71"])
    axes[0].set(ylabel="Simulated Latency (ms)", title="Chain Step Latency")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Call log: prompt vs response lengths
    if llm._call_log:
        p_lens = [c["prompt_len"] for c in llm._call_log]
        r_lens = [c["response_len"] for c in llm._call_log]
        x = np.arange(len(p_lens))
        axes[1].bar(x - 0.2, p_lens, 0.4, label="Prompt Tokens (chars)", color="steelblue")
        axes[1].bar(x + 0.2, r_lens, 0.4, label="Response Tokens (chars)", color="darkorange")
        axes[1].set(xlabel="LLM Call #", ylabel="Length (chars)",
                    title="LLM Call: Prompt vs Response Length")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "agent_frameworks.png", dpi=100)
    plt.close()
    print("  Saved agent_frameworks.png")


if __name__ == "__main__":
    demo()
