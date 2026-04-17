# 2.3.1 Combinatorics

Permutations, combinations, multinomial coefficients, birthday problem Monte Carlo, password entropy.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Pascal's triangle, recursion, pigeonhole |
| `working_example2.py` | P(n,k), C(n,k), birthday problem MC, MISSISSIPPI, entropy |
| `working_example.ipynb` | Interactive: counting → birthday → multinomial → entropy |

## Quick Reference

```python
import math

math.perm(n, k)      # P(n,k) = n!/(n-k)!  ordered
math.comb(n, k)      # C(n,k) = n!/(k!(n-k)!)  unordered
math.factorial(n)

# Multinomial: arrange 'MISSISSIPPI' -> 11! / (1!4!4!2!)
from collections import Counter
c = Counter(word)
n_ways = math.factorial(len(word)) // math.prod(math.factorial(v) for v in c.values())

# Shannon entropy of random password
bits = length * math.log2(alphabet_size)
```

## ML Connections
- **Cross-entropy loss** rooted in information-theoretic counting
- **Hyperparameter search** space = C(n, k) grid cells
- **Data augmentation** = controlled combinatorial expansions

## Learning Resources
- [Combinatorics — Art of Problem Solving](https://artofproblemsolving.com/wiki/index.php/Combinatorics)
- [Probability & Statistics (MIT OCW 18.650)](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/)

Explore this topic with a small practical project or coding exercise.

## What to build

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
