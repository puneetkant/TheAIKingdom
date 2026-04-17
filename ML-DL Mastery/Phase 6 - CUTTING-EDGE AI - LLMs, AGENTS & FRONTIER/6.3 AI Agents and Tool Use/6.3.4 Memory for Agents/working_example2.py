"""
Working Example 2: Memory for Agents
Demonstrates three memory types: in-context sliding window buffer,
episodic key-value store, and semantic search over memory embeddings.
Run: python working_example2.py
"""
from pathlib import Path
from collections import deque

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


class SlidingWindowMemory:
    """In-context buffer with fixed window size."""
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)

    def add(self, entry):
        self.buffer.append(entry)

    def get_context(self):
        return list(self.buffer)


class EpisodicMemory:
    """Key-value episodic memory store."""
    def __init__(self):
        self.store = {}

    def remember(self, key, value):
        self.store[key] = value

    def recall(self, key):
        return self.store.get(key, None)


class SemanticMemory:
    """Semantic memory with random vector embeddings (proxy for real embeddings)."""
    def __init__(self, dim=32):
        self.dim = dim
        self.memories = []
        self.embeddings = []
        self.rng = np.random.default_rng(42)

    def embed(self, text):
        # Proxy: hash-based deterministic random vector
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim)
        return v / (np.linalg.norm(v) + 1e-10)

    def store(self, text):
        self.memories.append(text)
        self.embeddings.append(self.embed(text))

    def search(self, query, top_k=3):
        if not self.memories:
            return []
        qvec = self.embed(query)
        embs = np.array(self.embeddings)
        scores = embs @ qvec
        top = np.argsort(scores)[::-1][:top_k]
        return [(self.memories[i], scores[i]) for i in top]


def demo():
    print("=== Memory for Agents ===")

    # Sliding window
    sw = SlidingWindowMemory(max_size=4)
    for msg in ["Hello", "What's the weather?", "It's sunny.", "Great!", "Let's go outside."]:
        sw.add(msg)
    print(f"  Sliding window (size=4): {sw.get_context()}")

    # Episodic memory
    ep = EpisodicMemory()
    ep.remember("user_name", "Alice")
    ep.remember("last_city", "Paris")
    ep.remember("preference", "vegetarian")
    print(f"  Episodic recall 'user_name': {ep.recall('user_name')}")
    print(f"  Episodic recall 'missing': {ep.recall('missing_key')}")

    # Semantic memory
    sm = SemanticMemory(dim=64)
    facts = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Machine learning uses data to train models.",
        "Paris has the Eiffel Tower.",
        "Neural networks are inspired by the brain.",
        "The Louvre is a museum in Paris.",
    ]
    for fact in facts:
        sm.store(fact)
    query = "Tell me about Paris"
    results = sm.search(query, top_k=3)
    print(f"\n  Semantic search: '{query}'")
    for mem, score in results:
        print(f"    [{score:.3f}] {mem}")

    # Visualise memory types
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sliding window - buffer over time
    msgs_all = ["Hello", "What's the weather?", "It's sunny.", "Great!", "Let's go outside.",
                "The park is nearby.", "Perfect day!"]
    window = 4
    buffer_sizes = [min(i + 1, window) for i in range(len(msgs_all))]
    axes[0].step(range(len(msgs_all)), buffer_sizes, where="post", color="steelblue", lw=2)
    axes[0].axhline(window, color="red", linestyle="--", label=f"Max window={window}")
    axes[0].set(xlabel="Message Index", ylabel="Buffer Size",
                title="Sliding Window Memory Buffer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episodic memory — key lookup times (simulated)
    keys = list(ep.store.keys()) + ["missing"]
    hit = [1, 1, 1, 0]
    axes[1].bar(keys, hit, color=["mediumseagreen"] * 3 + ["tomato"])
    axes[1].set(xlabel="Memory Key", ylabel="Hit (1) / Miss (0)",
                title="Episodic Memory Lookup")
    axes[1].grid(True, axis="y", alpha=0.3)

    # Semantic search scores
    all_scores = sm.embeddings @ sm.embed(query)
    axes[2].barh(range(len(facts)), all_scores, color="steelblue", alpha=0.7)
    top_idx = [facts.index(r[0]) for r in results]
    for i in top_idx:
        axes[2].barh(i, all_scores[i], color="tomato")
    axes[2].set(yticks=range(len(facts)),
                yticklabels=[f[:40] for f in facts],
                xlabel="Cosine Similarity", title=f"Semantic Memory: '{query}'")
    axes[2].tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT / "memory_agents.png", dpi=100)
    plt.close()
    print("  Saved memory_agents.png")


if __name__ == "__main__":
    demo()
