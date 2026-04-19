"""
Working Example 2: Testing — Production ML Testing Patterns
============================================================
Demonstrates testing patterns used in real ML codebases:
  - unittest.TestCase with setUp/tearDown
  - Property-based intuition (invariant checks)
  - Data validation tests (schema, dtype, range)
  - Model behavioural tests (monotonicity, equivariance)
  - Parametrized tests via subTest
  - Mocking with unittest.mock for external dependencies

Run:  python working_example2.py
Also:  python -m pytest working_example2.py -v
"""
import csv
import math
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch, MagicMock

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# -- Module under test (inline for self-contained demo) ------------------------
def normalize(data: list[float]) -> list[float]:
    lo, hi = min(data), max(data)
    span = hi - lo or 1e-8
    return [(x - lo) / span for x in data]


def standardize(data: list[float]) -> list[float]:
    n = len(data)
    mean = sum(data) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in data) / n) or 1e-8
    return [(x - mean) / std for x in data]


def accuracy(y_true: list, y_pred: list) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch")
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def train_knn(X_tr, y_tr, X_te, k=3) -> list:
    """Minimal KNN predict (for testing)."""
    def dist(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    from collections import Counter
    preds = []
    for x in X_te:
        ranked = sorted(zip([dist(x, xi) for xi in X_tr], y_tr))
        preds.append(Counter(y for _, y in ranked[:k]).most_common(1)[0][0])
    return preds


# -- Test Cases ----------------------------------------------------------------
class TestNormalize(unittest.TestCase):
    """Tests for the normalize() preprocessing function."""

    def test_range_is_zero_to_one(self):
        result = normalize([1, 2, 3, 4, 5])
        self.assertAlmostEqual(min(result), 0.0, places=9)
        self.assertAlmostEqual(max(result), 1.0, places=9)

    def test_single_element(self):
        result = normalize([42.0])
        self.assertEqual(len(result), 1)

    def test_all_same_values(self):
        # Should not raise ZeroDivisionError
        result = normalize([7.0, 7.0, 7.0])
        self.assertEqual(len(result), 3)

    def test_output_length_matches_input(self):
        data = list(range(100))
        self.assertEqual(len(normalize(data)), 100)

    def test_monotonic_order_preserved(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        result = normalize(data)
        for i in range(len(data) - 1):
            if data[i] <= data[i + 1]:
                self.assertLessEqual(result[i], result[i + 1])

    def test_parametrized_inputs(self):
        cases = [
            ([0, 1],             0.0, 1.0),
            ([-10, 0, 10],       0.0, 1.0),
            ([100, 200, 300],    0.0, 1.0),
        ]
        for data, expected_min, expected_max in cases:
            with self.subTest(data=data):
                r = normalize(data)
                self.assertAlmostEqual(min(r), expected_min, places=9)
                self.assertAlmostEqual(max(r), expected_max, places=9)


class TestStandardize(unittest.TestCase):
    def test_mean_near_zero(self):
        result = standardize([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sum(result) / len(result)
        self.assertAlmostEqual(mean, 0.0, places=9)

    def test_std_near_one(self):
        result = standardize([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sum(result) / len(result)
        var  = sum((x - mean) ** 2 for x in result) / len(result)
        self.assertAlmostEqual(math.sqrt(var), 1.0, places=9)


class TestAccuracy(unittest.TestCase):
    def test_perfect_predictions(self):
        y = [1, 0, 1, 1, 0]
        self.assertEqual(accuracy(y, y), 1.0)

    def test_zero_accuracy(self):
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        self.assertEqual(accuracy(y_true, y_pred), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            accuracy([1, 0], [1])

    def test_partial_accuracy(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.5)


class TestKNN(unittest.TestCase):
    """Behavioural tests for KNN — not whitebox, but invariant-based."""

    def setUp(self):
        # Simple 2D, 2-class data
        self.X_tr = [[0,0],[0,1],[1,0],[1,1],[10,10],[10,11],[11,10],[11,11]]
        self.y_tr = ["A","A","A","A","B","B","B","B"]

    def test_obvious_class_A(self):
        preds = train_knn(self.X_tr, self.y_tr, [[0.5, 0.5]])
        self.assertEqual(preds[0], "A")

    def test_obvious_class_B(self):
        preds = train_knn(self.X_tr, self.y_tr, [[10.5, 10.5]])
        self.assertEqual(preds[0], "B")

    def test_output_length_matches_query(self):
        queries = [[0,0],[5,5],[10,10]]
        preds = train_knn(self.X_tr, self.y_tr, queries, k=3)
        self.assertEqual(len(preds), len(queries))


class TestDataDownload(unittest.TestCase):
    """Mock external network calls."""

    @patch("urllib.request.urlretrieve")
    def test_download_called_once(self, mock_retrieve):
        dest = DATA / "_test_mock.csv"
        dest.unlink(missing_ok=True)
        mock_retrieve.return_value = (str(dest), MagicMock())
        # Create the file manually (mock won't)
        dest.write_text("a,b\n1,2\n")
        urllib.request.urlretrieve("http://example.com/data.csv", dest)
        mock_retrieve.assert_called_once()
        dest.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
