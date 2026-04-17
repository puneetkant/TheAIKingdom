# 6.5.3 Reasoning and Test-Time Compute

Test-time compute scaling (OpenAI o1, DeepSeek-R1) trades inference FLOPs for accuracy by generating extended reasoning chains before answering. Strategies include majority voting (self-consistency), best-of-N sampling, and beam search over reasoning paths. This folder simulates accuracy vs N-samples and compute-budget trade-offs.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Majority voting, best-of-N sampling, accuracy vs N, compute budget curve |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `reasoning_compute.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Self-consistency | Sample multiple CoT answers; majority vote |
| Best-of-N | Generate N solutions; score with verifier; return best |
| Process Reward Model | Score intermediate reasoning steps |
| MCTS | Monte Carlo Tree Search over reasoning steps |
| Scaling curve | Accuracy ~ log(N) for independent samples |

## Learning Resources

- Wang et al. *Self-Consistency* (2022)
- Lightman et al. *Let’s Verify Step by Step* (2023)
- OpenAI *o1 system card* (2024)
