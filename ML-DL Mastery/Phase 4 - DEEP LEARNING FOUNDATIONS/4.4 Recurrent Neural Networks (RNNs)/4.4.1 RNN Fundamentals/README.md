# 4.4.1 RNN Fundamentals

Vanilla RNN cell, hidden state dynamics, vanishing gradients, sine wave prediction.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | RNN cell step-by-step trace |
| `working_example2.py` | VanillaRNN class + sine prediction + hidden state heatmap |
| `working_example.ipynb` | Interactive: RNN step → prediction → hidden state plot |

## Quick Reference

```python
import torch.nn as nn

rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
# input: (batch, seq_len, input_size)
out, h_n = rnn(x)   # out: (B,T,H), h_n: (1,B,H)

# Single step
h = torch.zeros(1, 1, hidden_size)
for t in range(seq_len):
    out_t, h = rnn(x[:, t:t+1, :], h)
```

## RNN Equations

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

## Vanishing Gradient Problem

- Gradients ∝ $\prod_t \frac{\partial h_t}{\partial h_{t-1}}$
- tanh derivative ≤ 1 → product shrinks exponentially with T
- Fix: LSTM/GRU gates (4.4.2)

## Learning Resources
- [The Unreasonable Effectiveness of RNNs (Karpathy)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch RNN docs](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

Work with sequence data and recurrent models.

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
