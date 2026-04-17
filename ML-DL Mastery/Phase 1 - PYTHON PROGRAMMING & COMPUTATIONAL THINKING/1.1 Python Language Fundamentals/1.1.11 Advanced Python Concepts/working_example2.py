"""
Working Example 2: Advanced Python Concepts — Production Patterns
=================================================================
Demonstrates real-world advanced Python used in ML frameworks:
  - Metaclasses for model registration (like Hugging Face's AutoModel)
  - Descriptors for validated attributes
  - __slots__ for memory-efficient objects in large datasets
  - Generators with send() / throw() / close() protocol
  - asyncio concurrent downloads (demo without actual network)
  - dataclasses with __post_init__ and field() factory

Run:  python working_example2.py
"""
import csv
import math
import urllib.request
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, Iterator, Any
import asyncio
import itertools

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# ── 1. Model Registry via Metaclass ───────────────────────────────────────────
class ModelRegistry(type):
    """Metaclass that auto-registers concrete model classes."""
    _registry: dict[str, type] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        # Don't register abstract base
        if bases:
            tag = kwargs.get("tag", name.lower())
            mcs._registry[tag] = cls
        return cls

    @classmethod
    def list_models(mcs) -> list[str]:
        return sorted(mcs._registry.keys())

    @classmethod
    def build(mcs, tag: str, **kwargs) -> Any:
        if tag not in mcs._registry:
            raise KeyError(f"No model registered as '{tag}'. Available: {mcs.list_models()}")
        return mcs._registry[tag](**kwargs)


class BaseModel(metaclass=ModelRegistry):
    def predict(self, x): raise NotImplementedError


class LinearModel(BaseModel, tag="linear"):
    def __init__(self, lr=0.01): self.lr = lr
    def __repr__(self): return f"LinearModel(lr={self.lr})"
    def predict(self, x): return sum(x) * self.lr


class TreeModel(BaseModel, tag="tree"):
    def __init__(self, depth=3): self.depth = depth
    def __repr__(self): return f"TreeModel(depth={self.depth})"
    def predict(self, x): return max(x)


def demo_metaclass() -> None:
    print("=== Metaclass: Model Registry ===")
    print(f"  Registered models: {ModelRegistry.list_models()}")
    m = ModelRegistry.build("linear", lr=0.05)
    print(f"  Built: {m}  predict([1,2,3]) = {m.predict([1,2,3]):.3f}")
    t = ModelRegistry.build("tree", depth=5)
    print(f"  Built: {t}  predict([1,2,3]) = {t.predict([1,2,3])}")


# ── 2. Descriptors for validated attributes ───────────────────────────────────
class BoundedFloat:
    """Descriptor: float in [min_val, max_val], raises ValueError otherwise."""
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val
        self.name: str = ""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return obj.__dict__.get(self.name, 0.0)

    def __set__(self, obj, value):
        value = float(value)
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"{self.name} must be in [{self.min_val}, {self.max_val}], got {value}")
        obj.__dict__[self.name] = value


class HyperParams:
    learning_rate = BoundedFloat(1e-6, 1.0)
    dropout       = BoundedFloat(0.0, 0.9)
    weight_decay  = BoundedFloat(0.0, 0.1)

    def __init__(self, lr=1e-3, dropout=0.1, weight_decay=0.0):
        self.learning_rate = lr
        self.dropout       = dropout
        self.weight_decay  = weight_decay

    def __repr__(self):
        return (f"HyperParams(lr={self.learning_rate}, "
                f"dropout={self.dropout}, wd={self.weight_decay})")


def demo_descriptors() -> None:
    print("\n=== Descriptors: Validated HyperParams ===")
    hp = HyperParams(lr=0.001, dropout=0.3)
    print(f"  {hp}")
    try:
        hp.learning_rate = 5.0      # out of range
    except ValueError as e:
        print(f"  Caught: {e}")
    try:
        hp.dropout = -0.1           # negative
    except ValueError as e:
        print(f"  Caught: {e}")


# ── 3. __slots__ for memory efficiency ────────────────────────────────────────
class SampleSlots:
    """Uses __slots__ — no __dict__, ~30% less memory per instance."""
    __slots__ = ("feature", "label", "weight")

    def __init__(self, feature: list, label: str, weight: float = 1.0):
        self.feature = feature
        self.label   = label
        self.weight  = weight


def demo_slots() -> None:
    import sys
    print("\n=== __slots__: Memory Efficiency ===")

    class SampleDict:
        def __init__(self, feature, label, weight=1.0):
            self.feature = feature; self.label = label; self.weight = weight

    f = [0.1] * 10
    s_dict  = SampleDict(f, "cat")
    s_slots = SampleSlots(f, "cat")
    print(f"  SampleDict  size: {sys.getsizeof(s_dict) + sys.getsizeof(s_dict.__dict__):>5} bytes")
    print(f"  SampleSlots size: {sys.getsizeof(s_slots):>5} bytes  (no __dict__)")
    print(f"  Has __dict__: SampleDict={hasattr(s_dict,'__dict__')}, SampleSlots={hasattr(s_slots,'__dict__')}")


# ── 4. Generator with send() ──────────────────────────────────────────────────
def running_mean() -> Generator[float, float, None]:
    """Coroutine-style generator: send values, yield running mean."""
    total = 0.0
    count = 0
    x = yield 0.0   # first yield to prime the generator
    while True:
        total += x
        count += 1
        x = yield total / count


def demo_generator_send() -> None:
    print("\n=== Generator with send() — Running Mean ===")
    gen = running_mean()
    next(gen)   # prime
    losses = [0.85, 0.72, 0.61, 0.55, 0.50]
    for loss in losses:
        mean = gen.send(loss)
        print(f"  loss={loss:.2f}  running_mean={mean:.4f}")
    gen.close()


# ── 5. Dataclasses with __post_init__ ─────────────────────────────────────────
@dataclass
class Experiment:
    name:        str
    model_type:  str
    epochs:      int = 10
    batch_size:  int = 32
    tags:        list[str] = field(default_factory=list)
    run_id:      str = field(init=False)
    is_valid:    bool = field(init=False)

    def __post_init__(self):
        import hashlib
        self.run_id   = hashlib.md5(f"{self.name}{time.time()}".encode()).hexdigest()[:8]
        self.is_valid = self.epochs > 0 and self.batch_size > 0

    def summary(self) -> str:
        return (f"[{self.run_id}] {self.name} | model={self.model_type} | "
                f"epochs={self.epochs} | batch={self.batch_size} | tags={self.tags}")


def demo_dataclasses() -> None:
    print("\n=== Dataclasses with __post_init__ ===")
    exp1 = Experiment("baseline", "linear", epochs=50, tags=["v1", "cpu"])
    exp2 = Experiment("deep_run",  "tree",  epochs=100, batch_size=64, tags=["v2", "gpu"])
    for exp in [exp1, exp2]:
        print(f"  {exp.summary()}")
        print(f"    valid={exp.is_valid}")


# ── 6. itertools in ML contexts ───────────────────────────────────────────────
def demo_itertools() -> None:
    print("\n=== itertools in ML ===")
    # Product: hyperparameter grid search
    lrs     = [0.1, 0.01, 0.001]
    batches = [32, 64]
    grid    = list(itertools.product(lrs, batches))
    print(f"  Grid search combinations: {len(grid)}")
    for lr, bs in grid[:3]:
        print(f"    lr={lr}, batch={bs}")

    # chain: merge train/val generators
    train_ids = range(1, 6)
    val_ids   = range(101, 104)
    all_ids   = list(itertools.chain(train_ids, val_ids))
    print(f"  chain(train, val): {all_ids}")

    # islice: take first N rows from a generator
    def infinite_counter(start=0):
        i = start
        while True:
            yield i; i += 1

    first_10 = list(itertools.islice(infinite_counter(), 10))
    print(f"  islice(infinite, 10): {first_10}")

    # groupby
    data = [("cat",1),("dog",2),("cat",3),("bird",4),("dog",5)]
    data.sort(key=lambda x: x[0])
    for key, group in itertools.groupby(data, key=lambda x: x[0]):
        items = [v for _, v in group]
        print(f"  groupby {key}: {items}")


if __name__ == "__main__":
    demo_metaclass()
    demo_descriptors()
    demo_slots()
    demo_generator_send()
    demo_dataclasses()
    demo_itertools()
