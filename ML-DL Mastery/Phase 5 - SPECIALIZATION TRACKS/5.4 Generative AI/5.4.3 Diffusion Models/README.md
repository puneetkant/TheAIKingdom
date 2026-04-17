# 5.4.3 Diffusion Models

DDPM, DDIM, noise schedules, score matching, classifier-free guidance, Stable Diffusion.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | DDPM sampling pseudo-code with noise schedule |
| `working_example2.py` | Forward noising process visualised on digit images + SNR curve |
| `working_example.ipynb` | Interactive: 2D forward diffusion animation |

## Quick Reference

```python
# Forward process q(x_t | x_0)
def q_sample(x0, t, alphas_cumprod):
    sqrt_ab = alphas_cumprod[t] ** 0.5
    sqrt_1ab = (1 - alphas_cumprod[t]) ** 0.5
    eps = torch.randn_like(x0)
    return sqrt_ab * x0 + sqrt_1ab * eps, eps

# Training objective: predict noise eps
loss = F.mse_loss(model(x_t, t), eps)

# DDPM reverse step p(x_{t-1} | x_t)
def p_sample(model, x_t, t):
    eps_pred = model(x_t, t)
    x_pred   = (x_t - beta_t**0.5 * eps_pred) / alpha_t**0.5
    noise    = torch.randn_like(x_t) if t > 0 else 0
    return x_pred + sigma_t * noise
```

## Model Comparison

| Model | Steps | Speed | Quality |
|-------|-------|-------|---------|
| DDPM | 1000 | Slow | High |
| DDIM | 50 | Medium | High |
| LDM (Stable Diffusion) | 50 | Fast | Very high |
| Consistency Models | 1-4 | Very fast | Good |

## Learning Resources
- [DDPM paper](https://arxiv.org/abs/2006.11239)
- [Annotated diffusion (blog)](https://huggingface.co/blog/annotated-diffusion)

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
