# 4.7.1 PyTorch

PyTorch tensors, autograd, `nn.Module`, training loop, model persistence.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Tensor creation, basic operations, device awareness |
| `working_example2.py` | End-to-end MLP with autograd, Adam optimizer, Val MSE |
| `working_example.ipynb` | Interactive: autograd → MLP training loop |

## Quick Reference

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class MLP(nn.Module):
    def __init__(self, in_f, h, out_f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, h), nn.ReLU(), nn.Linear(h, out_f))
    def forward(self, x): return self.net(x)

model = MLP(8, 64, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Save / load
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| `Tensor` | n-d array with autograd support |
| `requires_grad` | Track gradient for that tensor |
| `backward()` | Compute all gradients via chain rule |
| `optimizer.zero_grad()` | Clear accumulated gradients before step |
| `model.eval()` + `torch.no_grad()` | Disable dropout/BN updates at inference |

## Learning Resources
- [PyTorch docs](https://pytorch.org/docs/stable/)
- [Deep Learning with PyTorch (book)](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

Write code using PyTorch tensors and models.

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
