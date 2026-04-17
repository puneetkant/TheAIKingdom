"""
Working Example 2: Probability Fundamentals — Sample Spaces, Bayes, Conditional
=================================================================================
Axioms, set-based simulation, Bayes theorem (medical test), law of total probability,
conditional independence, Monte Carlo for π.

Run:  python working_example2.py
"""
import random, math
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_axioms():
    print("=== Probability Axioms ===")
    # Roll a fair die 100,000 times
    random.seed(99)
    N = 100_000
    rolls = [random.randint(1, 6) for _ in range(N)]
    for face in range(1, 7):
        p = rolls.count(face) / N
        print(f"  P(die={face}) ≈ {p:.4f}  (expected 1/6 ≈ {1/6:.4f})")
    print(f"  Sum of probabilities: {sum(rolls.count(f)/N for f in range(1,7)):.6f}")

def demo_bayes():
    print("\n=== Bayes' Theorem — Medical Test ===")
    # Rare disease: P(D)=0.001, test sensitivity=0.99, specificity=0.99
    P_D      = 0.001        # prior
    P_pos_D  = 0.99         # sensitivity (true positive rate)
    P_pos_nD = 0.01         # 1 - specificity (false positive rate)
    # P(pos) = P(pos|D)P(D) + P(pos|~D)P(~D)
    P_pos = P_pos_D * P_D + P_pos_nD * (1 - P_D)
    # Posterior
    P_D_pos = P_pos_D * P_D / P_pos
    print(f"  P(D) = {P_D}")
    print(f"  P(+|D) = {P_pos_D}  P(+|~D) = {P_pos_nD}")
    print(f"  P(+) = {P_pos:.6f}")
    print(f"  P(D|+) = {P_D_pos:.6f}  ← only ~{P_D_pos*100:.1f}% chance despite positive test!")

def demo_conditional_independence():
    print("\n=== Conditional Independence ===")
    # Simulate: X and Y independent given Z (Naive Bayes style)
    # Z ~ Bernoulli(0.5), X|Z ~ Bernoulli(0.7 if Z else 0.3), Y|Z same
    random.seed(0); N = 100_000
    Z = [random.random() < 0.5 for _ in range(N)]
    X = [random.random() < (0.7 if z else 0.3) for z in Z]
    Y = [random.random() < (0.7 if z else 0.3) for z in Z]
    # P(X=1,Y=1)
    joint_xy = sum(x and y for x,y in zip(X, Y)) / N
    px = sum(X) / N; py = sum(Y) / N
    # Marginal X and Y are NOT independent
    print(f"  P(X=1,Y=1) = {joint_xy:.4f}  vs  P(X=1)*P(Y=1) = {px*py:.4f}")
    print(f"  Marginal dependence: {abs(joint_xy - px*py):.4f}")
    # But P(X=1,Y=1|Z=1) ≈ 0.7*0.7 = 0.49
    subset = [(x,y) for x,y,z in zip(X,Y,Z) if z]
    joint_given_z = sum(x and y for x,y in subset) / len(subset)
    print(f"  P(X=1,Y=1|Z=1) = {joint_given_z:.4f}  (conditional indep ≈ 0.49)")

def demo_monte_carlo_pi():
    print("\n=== Monte Carlo Estimation of π ===")
    import math
    random.seed(7)
    results = []
    for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
        inside = sum((random.random()**2 + random.random()**2) < 1 for _ in range(N))
        pi_est = 4 * inside / N
        results.append((N, pi_est))
        print(f"  N={N:>8,}: π ≈ {pi_est:.6f}  error={abs(pi_est - math.pi):.6f}")

if __name__ == "__main__":
    demo_axioms()
    demo_bayes()
    demo_conditional_independence()
    demo_monte_carlo_pi()
