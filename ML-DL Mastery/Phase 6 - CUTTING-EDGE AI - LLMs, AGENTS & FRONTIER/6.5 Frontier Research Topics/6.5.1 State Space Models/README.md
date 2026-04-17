# 6.5.1 State Space Models

State Space Models (SSMs) such as Mamba and S4 are linear recurrent systems that achieve transformer-level language modelling with linear (rather than quadratic) sequence complexity. The continuous SSM x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t) is discretised via Zero-Order Hold (ZOH) or bilinear methods. This folder implements SSM forward pass, impulse response, and eigenvalue stability analysis.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Linear SSM A/B/C/D matrices, bilinear discretisation (Tustin), impulse response, eigenvalue plot |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `state_space.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| State space | x'=Ax+Bu, y=Cx+Du; continuous linear system |
| HiPPO matrix | Structured A that projects input onto orthogonal polynomials |
| Selective SSM | Mamba's input-dependent Δt, B, C parameters |
| Bilinear (Tustin) | Discretisation method: (I+ΔA/2)^{-1}(I−ΔA/2) |
| Linear complexity | O(L) inference vs O(L²) for attention |

## Learning Resources

- Gu et al. *S4* (2021)
- Gu & Dao *Mamba* (2023)
- Gu et al. *HiPPO* (2020)
