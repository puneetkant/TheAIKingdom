# 1.1.13 Algorithms & Complexity

Sorting, binary search, heap top-K, Levenshtein edit distance — with Big-O benchmarks on real data.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Bubble/selection sort, binary search, Big-O examples |
| `working_example2.py` | Merge sort vs Timsort timing, binary search for lr, heap top-K, edit distance DP, MovieLens benchmark |
| `working_example.ipynb` | Interactive: sort timing, binary search, heap vs sort, edit distance, Big-O table |

## Run

```bash
python working_example.py
python working_example2.py    # downloads MovieLens ratings.csv
jupyter lab working_example.ipynb
```

## Big-O Reference

| Complexity | Example |
|------------|---------|
| O(1) | dict/set lookup, list index |
| O(log n) | binary search |
| O(n) | linear scan, sum(), min(), max() |
| O(n log n) | merge sort, Timsort, heapq.nlargest |
| O(n²) | insertion sort, naive matrix multiply |
| O(n·m) | edit distance DP |
| O(2ⁿ) | power set, naive subset |

## Key Patterns

```python
# Heap top-K: O(n log k) beats O(n log n) sort when k << n
import heapq
top_10 = heapq.nlargest(10, huge_list)

# Binary search on sorted list
import bisect
idx = bisect.bisect_left(sorted_list, target)

# Edit distance (Levenshtein) — core of spell checkers, NLP
def edit_dist(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1] if s1[i-1]==s2[j-1] \
                       else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[m][n]
```

## Dataset
- **MovieLens Small** — [Shahrukh0/MovieLens-Small](https://huggingface.co/datasets/Shahrukh0/MovieLens-Small) — used for timing benchmarks

## Learning Resources
- [Python sorting HowTo](https://docs.python.org/3/howto/sorting.html)
- [heapq docs](https://docs.python.org/3/library/heapq.html)
- [bisect docs](https://docs.python.org/3/library/bisect.html)
- [Real Python: Sorting](https://realpython.com/python-sort/)
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)
- **Book:** *Introduction to Algorithms* (CLRS) — Ch. 1-4
- **Book:** *Python Algorithms* (Magnus Lie Hetland)

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
