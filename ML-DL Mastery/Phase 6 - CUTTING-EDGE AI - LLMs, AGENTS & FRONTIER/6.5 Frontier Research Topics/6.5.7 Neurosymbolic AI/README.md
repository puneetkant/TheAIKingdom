# 6.5.7 Neurosymbolic AI

Neurosymbolic AI combines the pattern-matching power of neural networks with the precision and compositionality of symbolic reasoning. Systems like AlphaGeometry, DreamCoder, and Neural Module Networks route sub-problems to appropriate solvers. This folder implements a hybrid system: a neural digit recogniser (numpy softmax) paired with a symbolic arithmetic engine.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Neural digit predictor + symbolic arithmetic solver, noise robustness comparison |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `neurosymbolic.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Neural module networks | Route queries to specialised sub-networks |
| Symbolic solver | Rule-based exact computation (arithmetic, logic) |
| Program synthesis | Generate executable programs from examples |
| DreamCoder | Learn a library of reusable programs via wake-sleep |
| Hybrid verification | Neural perception + symbolic proof checker |

## Learning Resources

- Andreas et al. *Neural Module Networks* (2016)
- Trinh et al. *AlphaGeometry* (2024)
- Ellis et al. *DreamCoder* (2021)
