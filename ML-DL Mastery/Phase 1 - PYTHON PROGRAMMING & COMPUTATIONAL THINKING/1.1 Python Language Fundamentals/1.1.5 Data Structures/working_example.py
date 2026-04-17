"""
Working Example: Data Structures
Covers lists, tuples, dicts, sets, comprehensions,
nested structures, defaultdict, Counter, and choosing structures.
"""
from collections import defaultdict, Counter, namedtuple, OrderedDict
import copy


def lists_demo():
    print("=== Lists ===")
    fruits = ["banana", "apple", "cherry", "date"]

    # Methods
    fruits.append("elderberry")
    fruits.insert(0, "avocado")
    fruits.remove("date")
    popped = fruits.pop()
    fruits.sort()
    print(f"  sorted   : {fruits}")
    print(f"  popped   : {popped}")
    print(f"  reversed : {list(reversed(fruits))}")
    print(f"  slice [1:3]: {fruits[1:3]}")

    # Comprehension
    squares = [x**2 for x in range(1, 8)]
    evens   = [x for x in range(20) if x % 2 == 0]
    print(f"  squares  : {squares}")
    print(f"  evens    : {evens}")

    # Nested list (matrix)
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"  matrix[1][2] = {matrix[1][2]}")
    transposed = [[row[i] for row in matrix] for i in range(3)]
    print(f"  transposed: {transposed}")

    # Shallow vs deep copy
    original = [[1, 2], [3, 4]]
    shallow  = original.copy()
    deep     = copy.deepcopy(original)
    original[0][0] = 99
    print(f"  after mutate original[0][0]=99:")
    print(f"    shallow[0][0]={shallow[0][0]}  (affected)")
    print(f"    deep[0][0]   ={deep[0][0]}     (independent)")


def tuples_demo():
    print("\n=== Tuples ===")
    point  = (3, 4)
    triple = (1, 2, 3)

    # Unpacking
    x, y = point
    a, *rest, z = (1, 2, 3, 4, 5)
    print(f"  point      : {point}")
    print(f"  x={x}, y={y}")
    print(f"  star unpack: a={a}, rest={rest}, z={z}")

    # Named tuple
    Person = namedtuple("Person", ["name", "age", "job"])
    p = Person("Alice", 30, "Engineer")
    print(f"  namedtuple : {p}")
    print(f"    name={p.name}, age={p.age}")


def dicts_demo():
    print("\n=== Dictionaries ===")
    inventory = {"apples": 10, "bananas": 5, "cherries": 20}

    # Access
    print(f"  get('apples')     = {inventory.get('apples')}")
    print(f"  get('grapes', 0)  = {inventory.get('grapes', 0)}")
    inventory.setdefault("dates", 0)
    inventory.update({"bananas": 8, "figs": 3})
    print(f"  after update      = {inventory}")

    # Comprehension
    doubled = {k: v * 2 for k, v in inventory.items()}
    print(f"  doubled values    = {doubled}")

    # OrderedDict
    od = OrderedDict(a=1, b=2, c=3)
    od.move_to_end("a")
    print(f"  OrderedDict       = {dict(od)}")

    # defaultdict
    word_groups = defaultdict(list)
    for word in ["apple", "art", "banana", "avocado", "blueberry"]:
        word_groups[word[0]].append(word)
    print(f"  defaultdict       = {dict(word_groups)}")


def sets_demo():
    print("\n=== Sets ===")
    a = {1, 2, 3, 4, 5}
    b = {3, 4, 5, 6, 7}
    print(f"  a ∪ b (union)        = {a | b}")
    print(f"  a ∩ b (intersection) = {a & b}")
    print(f"  a - b (difference)   = {a - b}")
    print(f"  a △ b (symmetric ∆)  = {a ^ b}")
    print(f"  a ⊆ {1,2,3} (subset) = {a >= {1,2,3}}")

    # Remove duplicates from list preserving order
    dupes = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    seen = set()
    unique = [x for x in dupes if not (x in seen or seen.add(x))]
    print(f"  deduped list = {unique}")

    # frozenset
    fs = frozenset([1, 2, 3])
    d  = {fs: "immutable key"}
    print(f"  frozenset as dict key: {d}")


def counter_demo():
    print("\n=== Counter ===")
    text = "the quick brown fox jumps over the lazy dog"
    word_counts = Counter(text.split())
    print(f"  most common 5: {word_counts.most_common(5)}")

    char_counts = Counter(text.replace(" ", ""))
    print(f"  top 5 chars  : {char_counts.most_common(5)}")


def choosing_structures():
    print("\n=== Choosing the Right Data Structure ===")
    guide = {
        "list"       : "Ordered, mutable, duplicates allowed → sequences, stacks, queues",
        "tuple"      : "Ordered, immutable → fixed records, dict keys, namedtuples",
        "dict"       : "Key→value lookup, O(1) average → counts, caches, config",
        "set"        : "Unordered, unique → dedup, fast membership O(1), set math",
        "defaultdict": "dict with default factory → grouping, graph adjacency",
        "Counter"    : "dict subclass for counting → word frequency, histogram",
        "deque"      : "Double-ended queue → fast O(1) append/pop both ends",
    }
    for struct, desc in guide.items():
        print(f"  {struct:<14} : {desc}")


if __name__ == "__main__":
    lists_demo()
    tuples_demo()
    dicts_demo()
    sets_demo()
    counter_demo()
    choosing_structures()
