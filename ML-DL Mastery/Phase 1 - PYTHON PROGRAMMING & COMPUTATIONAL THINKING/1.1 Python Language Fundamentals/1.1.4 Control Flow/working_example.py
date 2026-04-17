"""
Working Example: Control Flow
Covers if/elif/else, ternary, for loops, while loops,
break/continue/pass, nested loops, enumerate, zip, and match/case.
"""


def if_elif_else():
    print("=== if / elif / else ===")
    scores = [95, 75, 55, 35]
    for score in scores:
        if score >= 90:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "F"
        print(f"  score={score} → grade={grade}")


def ternary():
    print("\n=== Ternary (Conditional Expression) ===")
    for n in [-3, 0, 7]:
        label = "positive" if n > 0 else ("zero" if n == 0 else "negative")
        print(f"  {n:3d} → {label}")


def for_loops():
    print("\n=== for Loops ===")
    # Basic range
    print("  squares:", [x**2 for x in range(1, 6)])

    # Enumerate
    print("\n  enumerate:")
    fruits = ["apple", "banana", "cherry"]
    for i, fruit in enumerate(fruits, start=1):
        print(f"    {i}. {fruit}")

    # zip
    print("\n  zip:")
    names  = ["Alice", "Bob", "Carol"]
    scores = [88, 94, 71]
    for name, score in zip(names, scores):
        print(f"    {name}: {score}")

    # Iterating dict
    print("\n  dict iteration:")
    data = {"a": 1, "b": 2, "c": 3}
    for key, val in data.items():
        print(f"    {key} → {val}")


def while_loops():
    print("\n=== while Loop ===")
    # Collatz conjecture
    n = 27
    steps = 0
    print(f"  Collatz from {n}: ", end="")
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    print(f"reached 1 in {steps} steps")

    # while with user-style counter
    print("\n  Countdown:")
    count = 5
    while count > 0:
        print(f"    {count}...", end=" ")
        count -= 1
    print("Go!")


def break_continue_pass():
    print("\n=== break / continue / pass ===")
    print("  First prime > 10:")
    for n in range(11, 50):
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                break
        else:
            print(f"    {n}")
            break

    print("  Odd numbers 1-10 (skip evens with continue):")
    result = []
    for n in range(1, 11):
        if n % 2 == 0:
            continue
        result.append(n)
    print(f"    {result}")

    # pass as a placeholder
    class PlaceholderModel:
        pass  # will implement later

    print("  PlaceholderModel defined with pass:", PlaceholderModel())


def nested_loops():
    print("\n=== Nested Loops — Multiplication Table ===")
    for i in range(1, 6):
        row = "  " + "  ".join(f"{i*j:3d}" for j in range(1, 6))
        print(row)


def match_case():
    print("\n=== match / case (Python 3.10+) ===")
    import sys
    if sys.version_info >= (3, 10):
        commands = ["quit", "go north", "go south", "look", "unknown"]
        for cmd in commands:
            match cmd.split():
                case ["quit"]:
                    action = "Quitting the game."
                case ["go", direction]:
                    action = f"Moving {direction}."
                case ["look"]:
                    action = "Looking around."
                case _:
                    action = f"Unknown command: '{cmd}'"
            print(f"  '{cmd}' → {action}")
    else:
        print("  match/case requires Python 3.10+")


if __name__ == "__main__":
    if_elif_else()
    ternary()
    for_loops()
    while_loops()
    break_continue_pass()
    nested_loops()
    match_case()
