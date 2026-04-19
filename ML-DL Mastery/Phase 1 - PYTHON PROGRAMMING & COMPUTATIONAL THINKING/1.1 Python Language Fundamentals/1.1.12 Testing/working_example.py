"""
Working Example: Testing
Covers unittest, assert statements, test structure,
mocking, parametrize patterns, fixtures, and TDD mindset.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import math


# -- Code under test -----------------------------------------------------------
def add(a, b):
    return a + b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner   = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount


def send_email(to, subject, body, smtp_client):
    """Function that uses an external dependency (smtp_client)."""
    smtp_client.send(to, subject, body)
    return True


# -- TestCase classes -----------------------------------------------------------
class TestArithmetic(unittest.TestCase):
    """Tests for arithmetic functions."""

    def test_add_integers(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_floats(self):
        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=10)

    def test_add_negatives(self):
        self.assertEqual(add(-4, -6), -10)

    def test_divide_normal(self):
        self.assertEqual(divide(10, 2), 5.0)

    def test_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            divide(5, 0)

    def test_divide_returns_float(self):
        result = divide(7, 2)
        self.assertIsInstance(result, float)


class TestIsPrime(unittest.TestCase):
    """Tests for is_prime with parametrised-style loops."""

    def test_primes(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 97]
        for n in primes:
            with self.subTest(n=n):
                self.assertTrue(is_prime(n), f"{n} should be prime")

    def test_non_primes(self):
        non_primes = [0, 1, 4, 6, 8, 9, 15, 25, 100]
        for n in non_primes:
            with self.subTest(n=n):
                self.assertFalse(is_prime(n), f"{n} should not be prime")


class TestBankAccount(unittest.TestCase):
    """Tests for BankAccount — setUp/tearDown lifecycle."""

    def setUp(self):
        self.account = BankAccount("Alice", 100.0)

    def tearDown(self):
        del self.account   # cleanup

    def test_initial_balance(self):
        self.assertEqual(self.account.balance, 100.0)

    def test_deposit_increases_balance(self):
        self.account.deposit(50.0)
        self.assertEqual(self.account.balance, 150.0)

    def test_deposit_invalid_amount(self):
        with self.assertRaises(ValueError):
            self.account.deposit(-10)
        with self.assertRaises(ValueError):
            self.account.deposit(0)

    def test_withdraw_decreases_balance(self):
        self.account.withdraw(30.0)
        self.assertAlmostEqual(self.account.balance, 70.0)

    def test_withdraw_insufficient_funds(self):
        with self.assertRaises(ValueError) as ctx:
            self.account.withdraw(200.0)
        self.assertIn("Insufficient", str(ctx.exception))

    def test_multiple_operations(self):
        self.account.deposit(200)
        self.account.withdraw(50)
        self.account.withdraw(25)
        self.assertEqual(self.account.balance, 225.0)


class TestMocking(unittest.TestCase):
    """Demonstrates unittest.mock."""

    def test_send_email_calls_smtp(self):
        mock_smtp = MagicMock()
        result = send_email("bob@example.com", "Hello", "Body", mock_smtp)
        self.assertTrue(result)
        mock_smtp.send.assert_called_once_with("bob@example.com", "Hello", "Body")

    def test_send_email_with_patch(self):
        """Patch a built-in inside the module under test."""
        with patch("builtins.print") as mock_print:
            print("test")
            mock_print.assert_called_once_with("test")

    def test_mock_return_value(self):
        mock_db = MagicMock()
        mock_db.query.return_value = [{"id": 1, "name": "Alice"}]
        rows = mock_db.query("SELECT * FROM users")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Alice")

    def test_mock_side_effect(self):
        mock_fn = MagicMock(side_effect=[10, 20, ValueError("out")])
        self.assertEqual(mock_fn(), 10)
        self.assertEqual(mock_fn(), 20)
        with self.assertRaises(ValueError):
            mock_fn()


# -- Assert statement usage outside unittest ------------------------------------
def demo_assert():
    print("=== assert statement (development-time checks) ===")
    x = 42
    assert x > 0, f"Expected positive, got {x}"
    print(f"  assert x > 0 passed (x={x})")

    try:
        y = -1
        assert y >= 0, f"y must be non-negative, got {y}"
    except AssertionError as e:
        print(f"  AssertionError caught: {e}")


# -- Run tests programmatically -------------------------------------------------
def run_all_tests():
    print("\n=== Running unittest suites ===")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestArithmetic, TestIsPrime, TestBankAccount, TestMocking]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n  Ran {result.testsRun} tests — "
          f"failures={len(result.failures)}, errors={len(result.errors)}")


if __name__ == "__main__":
    demo_assert()
    run_all_tests()
