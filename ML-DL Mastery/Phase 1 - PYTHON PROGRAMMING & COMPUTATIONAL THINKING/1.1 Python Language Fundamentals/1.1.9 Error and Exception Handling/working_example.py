"""
Working Example: Error and Exception Handling
Covers try/except/else/finally, exception hierarchy, raising,
custom exceptions, context managers, logging, and common patterns.
"""
import logging
import contextlib
import traceback


# -- Basic try/except ----------------------------------------------------------
def basic_try_except():
    print("=== Basic try / except ===")
    test_cases = ["42", "3.14", "abc", "0"]
    for s in test_cases:
        try:
            result = 100 / int(s)
            print(f"  100 / int({s!r:5}) = {result:.2f}")
        except ValueError:
            print(f"  int({s!r}) -> ValueError: cannot convert")
        except ZeroDivisionError:
            print(f"  100 / 0    -> ZeroDivisionError: division by zero")


# -- else / finally ------------------------------------------------------------
def else_and_finally():
    print("\n=== else / finally ===")
    def safe_divide(a, b):
        try:
            result = a / b
        except ZeroDivisionError as e:
            print(f"  Error: {e}")
            result = None
        else:
            # Runs only if no exception was raised
            print(f"  {a} / {b} = {result}")
        finally:
            # Always runs
            print(f"  [finally block executed for {a}/{b}]")
        return result

    safe_divide(10, 2)
    safe_divide(10, 0)


# -- Catching multiple exceptions ----------------------------------------------
def multiple_exceptions():
    print("\n=== Catching Multiple Exceptions ===")
    risky = [
        lambda: int("oops"),
        lambda: [][5],
        lambda: {"a": 1}["z"],
        lambda: 1 / 0,
        lambda: None.attr,          # AttributeError
    ]
    for fn in risky:
        try:
            fn()
        except (ValueError, KeyError) as e:
            print(f"  ValueError/KeyError: {e}")
        except (IndexError, ZeroDivisionError, AttributeError) as e:
            print(f"  {type(e).__name__}: {e}")


# -- Exception hierarchy -------------------------------------------------------
def exception_hierarchy():
    print("\n=== Exception Hierarchy ===")
    print("  BaseException")
    print("  ├- SystemExit, KeyboardInterrupt, GeneratorExit")
    print("  +- Exception")
    print("     ├- ArithmeticError -> ZeroDivisionError, OverflowError")
    print("     ├- LookupError    -> IndexError, KeyError")
    print("     ├- TypeError, ValueError, AttributeError")
    print("     ├- OSError        -> FileNotFoundError, PermissionError")
    print("     +- RuntimeError   -> RecursionError, NotImplementedError")

    # Catching broad base class
    try:
        raise OverflowError("number too large")
    except ArithmeticError as e:
        print(f"\n  Caught via ArithmeticError base: {e}")


# -- Raising exceptions --------------------------------------------------------
def raising_exceptions():
    print("\n=== Raising Exceptions ===")
    def set_age(age):
        if not isinstance(age, int):
            raise TypeError(f"age must be int, got {type(age).__name__}")
        if age < 0 or age > 150:
            raise ValueError(f"age {age} is out of range [0, 150]")
        return age

    for val in [25, -1, 200, "old"]:
        try:
            a = set_age(val)
            print(f"  set_age({val!r:5}) -> {a}")
        except (TypeError, ValueError) as e:
            print(f"  set_age({val!r:5}) -> {type(e).__name__}: {e}")

    # Re-raise
    try:
        try:
            x = int("bad")
        except ValueError:
            raise RuntimeError("Conversion failed") from ValueError
    except RuntimeError as e:
        print(f"\n  Chained exception: {e}")


# -- Custom exceptions ---------------------------------------------------------
class InsufficientFundsError(Exception):
    """Raised when a bank account has insufficient funds."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount  = amount
        super().__init__(
            f"Cannot withdraw {amount:.2f}: balance is {balance:.2f}"
        )


class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner   = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount


def custom_exceptions():
    print("\n=== Custom Exceptions ===")
    acct = BankAccount("Alice", 100.0)
    ops = [("deposit", 50), ("withdraw", 30), ("withdraw", 200)]
    for action, amount in ops:
        try:
            if action == "deposit":
                acct.deposit(amount)
            else:
                acct.withdraw(amount)
            print(f"  {action}({amount}) -> balance={acct.balance:.2f}")
        except InsufficientFundsError as e:
            print(f"  {action}({amount}) -> InsufficientFundsError: {e}")
        except ValueError as e:
            print(f"  {action}({amount}) -> ValueError: {e}")


# -- contextlib.suppress -------------------------------------------------------
def suppress_demo():
    print("\n=== contextlib.suppress ===")
    d = {"a": 1}
    with contextlib.suppress(KeyError):
        _ = d["z"]           # silently swallowed
    print("  KeyError suppressed — code continues")

    import os
    with contextlib.suppress(FileNotFoundError):
        os.remove("nonexistent_file.txt")
    print("  FileNotFoundError suppressed")


# -- Logging ------------------------------------------------------------------
def logging_demo():
    print("\n=== Logging ===")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s | %(message)s"
    )
    log = logging.getLogger("example")
    log.debug("Debug detail — not shown in production typically")
    log.info("Application started")
    log.warning("Config file missing, using defaults")
    try:
        int("bad")
    except ValueError:
        log.exception("Caught exception (includes traceback):")


if __name__ == "__main__":
    basic_try_except()
    else_and_finally()
    multiple_exceptions()
    exception_hierarchy()
    raising_exceptions()
    custom_exceptions()
    suppress_demo()
    logging_demo()
