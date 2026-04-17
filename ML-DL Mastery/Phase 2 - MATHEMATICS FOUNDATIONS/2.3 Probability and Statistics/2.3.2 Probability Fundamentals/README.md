# 2.3.2 Probability Fundamentals

Probability axioms, Bayes' theorem, conditional independence, Monte Carlo simulation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Set operations, Venn diagram probability |
| `working_example2.py` | Die simulation, Bayes medical test, conditional independence demo, MC π |
| `working_example.ipynb` | Interactive: axioms → Bayes → conditional indep → MC π |

## Quick Reference

```python
# Bayes' theorem
# P(A|B) = P(B|A) * P(A) / P(B)
P_pos = P_pos_D * P_D + P_pos_nD * (1 - P_D)
P_D_given_pos = P_pos_D * P_D / P_pos

# Law of total probability
# P(B) = Σ P(B|A_i) P(A_i)

# Monte Carlo
inside = sum((random.random()**2 + random.random()**2) < 1 for _ in range(N))
pi_est = 4 * inside / N
```

## Key Concepts
- **Kolmogorov axioms**: P(Ω)=1, P(A)≥0, additive for disjoint events
- **Bayes**: posterior ∝ likelihood × prior
- **Conditional independence**: X ⊥ Y | Z (Naive Bayes assumption)

## ML Connections
- **Naive Bayes classifier** relies on conditional independence
- **Probabilistic graphical models** encode conditional independence structure

## Learning Resources
- [Khan Academy: Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)
- **Book:** *Probability and Statistics for ML* (Wainwright)

Compute probabilities, expectations, and distributions.

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
