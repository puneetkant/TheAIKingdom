"""
Working Example: Operators
Covers arithmetic, comparison, logical, bitwise, assignment,
identity, membership operators, and precedence.
"""


def arithmetic_operators():
    print("=== Arithmetic Operators ===")
    a, b = 17, 5
    ops = [
        ('+',  a + b),
        ('-',  a - b),
        ('*',  a * b),
        ('/',  a / b),    # true division -> float
        ('//', a // b),   # floor division
        ('%',  a % b),    # modulo
        ('**', a ** b),   # exponentiation
    ]
    for op, result in ops:
        print(f"  {a} {op} {b} = {result}")


def comparison_operators():
    print("\n=== Comparison Operators ===")
    pairs = [(3, 5), (5, 5), (7, 5)]
    for x, y in pairs:
        print(f"  {x} vs {y}: "
              f"=={x==y}  !={x!=y}  <{x<y}  >{x>y}  <={x<=y}  >={x>=y}")


def logical_operators():
    print("\n=== Logical Operators ===")
    cases = [(True, True), (True, False), (False, False)]
    for a, b in cases:
        print(f"  {a} and {b} = {a and b} | {a} or {b} = {a or b} | not {a} = {not a}")

    # Short-circuit evaluation
    print("\n  Short-circuit:")
    print(f"  0 and (1/0) -> no ZeroDivisionError = {0 and (1/0) if False else 0 and 'skipped'}")
    print(f"  1 or  (1/0) -> no ZeroDivisionError = {1 or 'skipped'}")


def bitwise_operators():
    print("\n=== Bitwise Operators ===")
    a, b = 0b1010, 0b1100   # 10, 12
    print(f"  a  = {a:04b} ({a})")
    print(f"  b  = {b:04b} ({b})")
    print(f"  a & b  = {a & b:04b} ({a & b})   AND")
    print(f"  a | b  = {a | b:04b} ({a | b})   OR")
    print(f"  a ^ b  = {a ^ b:04b} ({a ^ b})   XOR")
    print(f"  ~a     = {~a}   NOT (bitwise complement)")
    print(f"  a << 1 = {a << 1:04b} ({a << 1})   left shift")
    print(f"  a >> 1 = {a >> 1:04b} ({a >> 1})   right shift")
    print("\n  Practical: check if number is even")
    for n in [0, 1, 2, 3, 8, 9]:
        print(f"    {n} & 1 = {n & 1}  -> {'odd' if n & 1 else 'even'}")


def assignment_operators():
    print("\n=== Assignment Operators ===")
    x = 10
    steps = [('+=', 5), ('-=', 2), ('*=', 3), ('//=', 4), ('%=', 3), ('**=', 2)]
    print(f"  start x = {x}")
    for op, val in steps:
        exec(f"x {op} {val}")
        print(f"  x {op} {val}  -> x = {eval('x', {'x': x})}", end='')
        if op == '+=':   x += val
        elif op == '-=': x -= val
        elif op == '*=': x *= val
        elif op == '//=':x //= val
        elif op == '%=': x %= val
        elif op == '**=':x **= val
        print(f" = {x}")


def identity_and_membership():
    print("\n=== Identity (is / is not) ===")
    a = [1, 2, 3]
    b = a        # same object
    c = [1, 2, 3]  # equal but different object
    print(f"  a is b     = {a is b}   (same object)")
    print(f"  a is c     = {a is c}   (different object)")
    print(f"  a == c     = {a == c}   (equal value)")
    print(f"  None is None = {None is None}")

    print("\n=== Membership (in / not in) ===")
    fruits = ["apple", "banana", "cherry"]
    for fruit in ["banana", "grape"]:
        print(f"  '{fruit}' in fruits     = {fruit in fruits}")
        print(f"  '{fruit}' not in fruits = {fruit not in fruits}")


def operator_precedence():
    print("\n=== Operator Precedence ===")
    exprs = [
        "2 + 3 * 4",       # * before +
        "(2 + 3) * 4",     # parens override
        "2 ** 3 ** 2",     # ** is right-associative -> 2**(3**2)
        "not True or True",# not binds tighter than or
        "10 - 3 - 2",      # left-associative
    ]
    for expr in exprs:
        print(f"  {expr:<22} = {eval(expr)}")


if __name__ == "__main__":
    arithmetic_operators()
    comparison_operators()
    logical_operators()
    bitwise_operators()
    assignment_operators()
    identity_and_membership()
    operator_precedence()
