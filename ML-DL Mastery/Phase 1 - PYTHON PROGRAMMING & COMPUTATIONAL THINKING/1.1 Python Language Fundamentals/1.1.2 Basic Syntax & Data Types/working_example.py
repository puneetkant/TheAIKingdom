"""
Working Example: Basic Syntax & Data Types
Covers variables, naming, all built-in types, type conversion,
f-strings, string methods, and type introspection.
"""


def variables_and_naming():
    print("=== Variables & Naming Conventions ===")
    # snake_case for variables and functions
    first_name = "Ada"
    last_name = "Lovelace"
    birth_year = 1815
    is_mathematician = True

    print(f"Name      : {first_name} {last_name}")
    print(f"Born      : {birth_year}")
    print(f"Mathematician: {is_mathematician}")

    # Constants use ALL_CAPS (by convention)
    MAX_RETRIES = 3
    PI = 3.14159
    print(f"MAX_RETRIES={MAX_RETRIES}, PI={PI}")


def numeric_types():
    print("\n=== Numeric Types ===")
    i = 42               # int
    f = 3.14             # float
    c = 2 + 3j           # complex
    big = 10 ** 50       # Python ints have unlimited precision

    print(f"int    : {i}  type={type(i).__name__}")
    print(f"float  : {f}  type={type(f).__name__}")
    print(f"complex: {c}  real={c.real} imag={c.imag}")
    print(f"big int: {big}")

    # Float precision gotcha
    print(f"0.1 + 0.2 = {0.1 + 0.2}  (floating point imprecision)")
    from decimal import Decimal
    print(f"Decimal: {Decimal('0.1') + Decimal('0.2')}  (precise)")


def string_types():
    print("\n=== String Type ===")
    s = "Hello, World!"
    raw = r"C:\Users\admin\new_folder"   # raw string
    multi = """Line 1
Line 2
Line 3"""

    print(f"String  : {s}")
    print(f"Length  : {len(s)}")
    print(f"Upper   : {s.upper()}")
    print(f"Lower   : {s.lower()}")
    print(f"Replace : {s.replace('World', 'Python')}")
    print(f"Split   : {s.split(', ')}")
    print(f"Strip   : '  hello  '.strip() = {'  hello  '.strip()!r}")
    print(f"Find    : {s.find('World')}")
    print(f"Starts  : {s.startswith('Hello')}")
    print(f"Raw str : {raw}")
    print(f"Multiline:\n{multi}")

    # f-strings
    name = "Learner"
    score = 98.5
    print(f"\nf-string: Hello {name}, your score is {score:.1f}%")
    print(f"f-string math: 2**10 = {2**10}")

    # String slicing
    print(f"\nSlicing 'Python':")
    word = "Python"
    print(f"  [0]   = {word[0]}")
    print(f"  [-1]  = {word[-1]}")
    print(f"  [1:4] = {word[1:4]}")
    print(f"  [::-1]= {word[::-1]} (reversed)")


def boolean_and_none():
    print("\n=== Boolean & None ===")
    t = True
    f_ = False
    n = None

    print(f"True and False = {t and f_}")
    print(f"True or False  = {t or f_}")
    print(f"not True       = {not t}")
    print(f"None is None   = {n is None}")

    # Truthy / falsy
    falsy_values = [0, 0.0, "", [], {}, set(), None, False]
    print("\nFalsy values:", [bool(v) for v in falsy_values])


def type_conversion():
    print("\n=== Type Conversion ===")
    pairs = [
        ("int('42')",    int('42')),
        ("float('3.14')", float('3.14')),
        ("str(100)",     str(100)),
        ("bool(0)",      bool(0)),
        ("bool(42)",     bool(42)),
        ("list('abc')",  list('abc')),
        ("int(3.9)",     int(3.9)),   # truncates, not rounds
    ]
    for expr, result in pairs:
        print(f"  {expr:<22} → {result!r:<15} type={type(result).__name__}")


def type_introspection():
    print("\n=== Type Introspection ===")
    values = [42, 3.14, "hello", True, None, [1, 2], {"a": 1}]
    for v in values:
        print(f"  value={v!r:<15} type={type(v).__name__:<10} isinstance(int)={isinstance(v, int)}")


if __name__ == "__main__":
    variables_and_naming()
    numeric_types()
    string_types()
    boolean_and_none()
    type_conversion()
    type_introspection()
