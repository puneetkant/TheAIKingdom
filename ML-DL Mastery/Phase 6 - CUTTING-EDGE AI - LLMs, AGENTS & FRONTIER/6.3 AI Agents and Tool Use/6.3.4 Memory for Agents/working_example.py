"""
Working Example: Memory for AI Agents
Covers short-term, long-term, episodic, and semantic memory types,
plus vector memory, memory management, and retrieval patterns.
"""
import numpy as np
import os, json
from collections import deque

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_agent_memory")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Memory taxonomy ────────────────────────────────────────────────────────
def memory_taxonomy():
    print("=== Memory for AI Agents ===")
    print()
    print("  Human memory analogy:")
    mem_types = [
        ("In-context (working)",  "Current conversation; context window; fast; limited"),
        ("External (semantic)",   "Vector DB of facts; unlimited; slow retrieval"),
        ("External (episodic)",   "Past experiences; interaction logs; time-stamped"),
        ("Procedural",            "How to do things; baked into model weights"),
        ("Cache (working+)",      "Prefix caching; KV cache; intermediate"),
    ]
    for m, d in mem_types:
        print(f"  {m:<26} {d}")
    print()
    print("  When each type is needed:")
    use_cases = [
        ("Short task",       "In-context only; everything fits in window"),
        ("Long conversation","Summarise old turns → re-inject key facts"),
        ("Repeat user",      "User profile in semantic memory; retrieve on query"),
        ("Multi-session",    "Episodic: remember 'last time user asked about...'"),
        ("Factual tasks",    "RAG from semantic memory; grounded answers"),
    ]
    for uc, d in use_cases:
        print(f"  {uc:<18} {d}")


# ── 2. Sliding window memory (in-context) ─────────────────────────────────────
class ConversationMemory:
    """Sliding window with summarisation."""
    def __init__(self, max_turns=10):
        self.turns    = deque(maxlen=max_turns)
        self.summary  = ""

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def get_messages(self):
        msgs = []
        if self.summary:
            msgs.append({"role": "system",
                         "content": f"Previous conversation summary: {self.summary}"})
        msgs.extend(self.turns)
        return msgs

    def compress(self, max_kept=5):
        """Summarise old turns to free window space."""
        if len(self.turns) < max_kept:
            return
        old = list(self.turns)[:-max_kept]
        self.summary += " | ".join(f"{m['role']}: {m['content'][:30]}" for m in old)
        for _ in range(len(old)):
            self.turns.popleft()


def demo_conversation_memory():
    print("\n=== Conversation Memory Demo ===")
    mem = ConversationMemory(max_turns=6)
    conversation = [
        ("user",      "My name is Alice."),
        ("assistant", "Nice to meet you, Alice!"),
        ("user",      "I'm a software engineer."),
        ("assistant", "Great! What do you work on?"),
        ("user",      "I build ML pipelines."),
        ("assistant", "Fascinating! What frameworks do you use?"),
        ("user",      "Mostly PyTorch and MLflow."),
        ("assistant", "Good choices for production ML!"),
    ]
    for role, content in conversation:
        mem.add(role, content)

    msgs = mem.get_messages()
    print(f"  After {len(conversation)} turns, {len(msgs)} messages in context")
    print(f"  Visible messages:")
    for m in msgs:
        print(f"    [{m['role']}]: {m['content'][:60]}")

    mem.compress(max_kept=3)
    print(f"\n  After compression: {len(mem.turns)} messages + summary")
    print(f"  Summary: {mem.summary[:100]}")


# ── 3. Vector memory ──────────────────────────────────────────────────────────
class VectorMemory:
    """Simple vector store for agent memory."""
    def __init__(self, dim=8):
        self.embeddings = []
        self.memories   = []
        self.dim        = dim

    def _embed(self, text: str):
        """Deterministic fake embedding (real: sentence-transformers)."""
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.normal(0, 1, self.dim)

    def add(self, text: str, metadata: dict = None):
        emb = self._embed(text)
        emb /= np.linalg.norm(emb)
        self.embeddings.append(emb)
        self.memories.append({"text": text, "metadata": metadata or {}})

    def search(self, query: str, k: int = 3):
        if not self.embeddings:
            return []
        q = self._embed(query)
        q /= np.linalg.norm(q)
        sims = np.array(self.embeddings) @ q
        top_k = np.argsort(-sims)[:k]
        return [(self.memories[i]["text"], float(sims[i])) for i in top_k]


def demo_vector_memory():
    print("\n=== Vector Memory Demo ===")
    mem = VectorMemory(dim=16)
    facts = [
        "Alice prefers Python over JavaScript for backend tasks.",
        "Alice has 5 years of machine learning experience.",
        "The user's favourite framework is PyTorch.",
        "The user lives in London and works remotely.",
        "Alice is allergic to peanuts.",
        "The user has a dog named Pixel.",
        "Alice's birthday is March 15th.",
    ]
    for f in facts:
        mem.add(f)

    query = "What programming languages does Alice prefer?"
    results = mem.search(query, k=3)
    print(f"  Query: '{query}'")
    print(f"  Top retrieved memories:")
    for text, sim in results:
        print(f"    [sim={sim:.3f}] {text}")


# ── 4. Memory management ──────────────────────────────────────────────────────
def memory_management():
    print("\n=== Memory Management Strategies ===")
    print()
    strategies = [
        ("Recency weighting",    "More recent memories get higher weight"),
        ("Importance scoring",   "LLM scores surprise/importance at storage time"),
        ("Reflection",           "Periodically synthesise memories into higher abstractions"),
        ("Forgetting",           "LRU eviction or decay by access frequency"),
        ("Hierarchical",         "Hot/warm/cold tiers; recent in-context, old in DB"),
        ("Episodic + semantic",  "Store raw episodes; extract semantic facts separately"),
    ]
    for s, d in strategies:
        print(f"  {s:<24} {d}")
    print()
    print("  Memory frameworks:")
    frameworks = [
        ("MemGPT",    "OS-style virtual context management"),
        ("Zep",       "Memory layer for AI apps; graph + temporal"),
        ("Mem0",      "Memory for AI agents and products; managed"),
        ("LangMem",   "LangChain memory; summarisation + storage"),
    ]
    for f, d in frameworks:
        print(f"  {f:<12} {d}")


if __name__ == "__main__":
    memory_taxonomy()
    demo_conversation_memory()
    demo_vector_memory()
    memory_management()
