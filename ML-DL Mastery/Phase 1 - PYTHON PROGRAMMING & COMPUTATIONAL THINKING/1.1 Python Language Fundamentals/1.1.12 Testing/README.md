# 1.1.12 Testing

Unit tests, parametrized subTests, mocking, and ML behavioural invariant testing with `unittest`.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | assert, simple unittest, parametrized tests |
| `working_example2.py` | TestNormalize, TestAccuracy, TestKNN (behavioural), TestDataDownload (mock) |
| `working_example.ipynb` | Interactive: define + test normalize and accuracy, mock urllib |

## Run

```bash
python working_example.py
python working_example2.py                   # unittest verbosity=2
python -m pytest working_example2.py -v     # with pytest
jupyter lab working_example.ipynb
```

## Testing Patterns

```python
import unittest

class TestNormalize(unittest.TestCase):
    def setUp(self): ...          # runs before each test
    def tearDown(self): ...       # runs after each test

    def test_range_zero_to_one(self):
        result = normalize([1,2,3,4,5])
        self.assertAlmostEqual(min(result), 0.0)
        self.assertAlmostEqual(max(result), 1.0)

    def test_parametrized(self):
        for data in [[0,1],[-10,10],[100,200]]:
            with self.subTest(data=data):    # reports each case separately
                r = normalize(data)
                self.assertAlmostEqual(min(r), 0.0)
```

## ML Behavioural Tests

```python
# Monotonicity invariant
def test_monotonic(self):
    data = sorted([...])
    result = transform(data)
    for i in range(len(result)-1):
        self.assertLessEqual(result[i], result[i+1])

# Mock external dependencies
from unittest.mock import patch
with patch('urllib.request.urlretrieve') as mock:
    mock.return_value = ('path', {})
    ...  # test without network
```

## Learning Resources
- [unittest docs](https://docs.python.org/3/library/unittest.html)
- [unittest.mock docs](https://docs.python.org/3/library/unittest.mock.html)
- [pytest docs](https://docs.pytest.org/)
- [Real Python: Python Testing](https://realpython.com/python-testing/)
- [Real Python: pytest tutorial](https://realpython.com/pytest-python-testing/)
- **Book:** *Python Testing with pytest* (Brian Okken)
- **Book:** *Python Cookbook* Ch. 14 (testing and debugging)

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
