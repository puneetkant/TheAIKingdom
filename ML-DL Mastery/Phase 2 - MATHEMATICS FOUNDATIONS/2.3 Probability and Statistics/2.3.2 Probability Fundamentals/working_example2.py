"""
Working Example 2: Probability Fundamentals — Sample Spaces, Bayes, Conditional
=================================================================================
Axioms, set-based simulation, Bayes theorem (medical test), law of total probability,
conditional independence, Monte Carlo for pi.

Run:  python working_example2.py
"""
import random, math
from pathlib import Path
try:
    import matplotlib
    matplotlib.use("Agg")
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
        print(f"  P(die={face}) ~= {p:.4f}  (expected 1/6 ~= {1/6:.4f})")
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
    print(f"  P(D|+) = {P_D_pos:.6f}  <- only ~{P_D_pos*100:.1f}% chance despite positive test!")

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
    # But P(X=1,Y=1|Z=1) ~= 0.7*0.7 = 0.49
    subset = [(x,y) for x,y,z in zip(X,Y,Z) if z]
    joint_given_z = sum(x and y for x,y in subset) / len(subset)
    print(f"  P(X=1,Y=1|Z=1) = {joint_given_z:.4f}  (conditional indep ~= 0.49)")

def demo_monte_carlo_pi():
    print("\n=== Monte Carlo Estimation of pi ===")
    import math
    random.seed(7)
    results = []
    for N in [100, 1_000, 10_000, 100_000, 1_000_000]:
        inside = sum((random.random()**2 + random.random()**2) < 1 for _ in range(N))
        pi_est = 4 * inside / N
        results.append((N, pi_est))
        print(f"  N={N:>8,}: pi ~= {pi_est:.6f}  error={abs(pi_est - math.pi):.6f}")

def demo_law_total_probability():
    print("\n=== Law of Total Probability ===")
    # P(B) = sum_i P(B|A_i) * P(A_i)
    # Partition: study groups A1 (<2h), A2 (2-5h), A3 (>5h)
    P_A = [0.30, 0.50, 0.20]
    P_B_given_A = [0.20, 0.60, 0.90]
    P_B = sum(pb * pa for pb, pa in zip(P_B_given_A, P_A))
    print(f"  Groups: A1(<2h) P={P_A[0]}, A2(2-5h) P={P_A[1]}, A3(>5h) P={P_A[2]}")
    print(f"  P(pass|A1)={P_B_given_A[0]}, P(pass|A2)={P_B_given_A[1]}, "
          f"P(pass|A3)={P_B_given_A[2]}")
    print(f"  P(pass) via total prob = {P_B:.4f}")
    # Monte Carlo verification
    random.seed(3)
    N = 200_000
    passes = 0
    for _ in range(N):
        r = random.random()
        if r < P_A[0]:
            prob_pass = P_B_given_A[0]
        elif r < P_A[0] + P_A[1]:
            prob_pass = P_B_given_A[1]
        else:
            prob_pass = P_B_given_A[2]
        if random.random() < prob_pass:
            passes += 1
    print(f"  MC estimate P(pass) ~= {passes/N:.4f}  (N={N:,})")


def demo_birthday_paradox():
    print("\n=== Birthday Paradox (Exact Formula) ===")
    # P(collision) = 1 - 365!/ ((365-n)! * 365^n)
    # Use log-sum for numerical stability
    import math
    print(f"  {'n':>4}  {'P(at least 1 collision)':>24}  {'> 50%?':>7}")
    for n in [10, 20, 23, 30, 40, 50, 57, 100, 366]:
        if n > 365:
            p_coll = 1.0
        else:
            log_p_none = sum(math.log((365 - i) / 365) for i in range(n))
            p_coll = 1.0 - math.exp(log_p_none)
        over_half = "YES" if p_coll > 0.5 else "no"
        print(f"  {n:>4}  {p_coll:>24.6f}  {over_half:>7}")
    print("  (n=23 is the first where P > 0.5 -- the famous threshold)")


def demo_conditional_prob_table():
    print("\n=== Joint, Marginal, and Conditional Probabilities ===")
    # Weather / umbrella joint distribution
    # P(Rain, Umbrella) as a 2x2 table
    #                Umbrella=Yes  Umbrella=No
    # Rain=Yes          0.30          0.05
    # Rain=No           0.10          0.55
    joint = np.array([[0.30, 0.05],
                      [0.10, 0.55]])
    labels_rain = ["Rain=Yes", "Rain=No"]
    labels_umb  = ["Umb=Yes", "Umb=No"]
    p_rain = joint.sum(axis=1)   # marginal over umbrella
    p_umb  = joint.sum(axis=0)   # marginal over rain
    print(f"  Joint P(Rain, Umb):")
    for i, lr in enumerate(labels_rain):
        for j, lu in enumerate(labels_umb):
            print(f"    P({lr}, {lu}) = {joint[i,j]:.2f}", end="  ")
        print()
    print(f"  Marginal P(Rain=Yes)={p_rain[0]:.2f}, P(Rain=No)={p_rain[1]:.2f}")
    print(f"  Marginal P(Umb=Yes)={p_umb[0]:.2f},  P(Umb=No)={p_umb[1]:.2f}")
    # Conditional P(Umb | Rain)
    cond = joint / p_rain[:, None]
    print(f"  Conditional P(Umb=Yes | Rain=Yes) = {cond[0,0]:.4f}")
    print(f"  Conditional P(Umb=Yes | Rain=No)  = {cond[1,0]:.4f}")
    # Independence test
    indep = np.outer(p_rain, p_umb)
    print(f"  Independent product P(Rain)*P(Umb):")
    for i in range(2):
        for j in range(2):
            print(f"    {indep[i,j]:.4f}", end="  ")
        print()
    print(f"  Max deviation from independence: {np.max(np.abs(joint - indep)):.4f}")
    print(f"  Are Rain and Umbrella independent? {np.allclose(joint, indep, atol=1e-9)}")


def demo_probability_inequalities():
    print("\n=== Probability Inequalities (Markov & Union Bound) ===")
    import math
    # Markov: P(X >= a) <= E[X] / a  for non-negative X
    np.random.seed(42)
    lam = 3.0
    samples = np.random.exponential(1.0 / lam, size=1_000_000)  # Exp(lambda=3)
    mu = float(np.mean(samples))
    print(f"  X ~ Exp(lambda={lam}), E[X]={mu:.4f}")
    print(f"  Markov: P(X >= a) <= E[X]/a")
    print(f"  {'a':>6}  {'Markov bound':>14}  {'Empirical':>10}")
    for a in [0.5, 1.0, 2.0, 5.0]:
        bound    = mu / a
        empirical = float(np.mean(samples >= a))
        ok = "OK" if empirical <= bound + 1e-6 else "FAIL"
        print(f"  {a:>6.2f}  {min(bound,1.0):>14.6f}  {empirical:>10.6f}  {ok}")
    # Union bound: P(A u B) <= P(A) + P(B)
    print(f"\n  Union Bound: P(A u B) <= P(A) + P(B)")
    P_A, P_B = 0.3, 0.4
    print(f"  P(A)={P_A}, P(B)={P_B}, union bound <= {P_A + P_B:.2f}")
    for P_AB in [0.10, 0.20, 0.30]:
        P_union = P_A + P_B - P_AB
        print(f"  P(AnB)={P_AB}: P(AuB)={P_union:.2f} <= {P_A+P_B:.2f}: "
              f"{P_union <= P_A + P_B}")


def demo_law_of_large_numbers():
    print("\n=== Law of Large Numbers ===")
    # Running mean of iid coin flips converges to p=0.5
    np.random.seed(11)
    flips = np.random.randint(0, 2, size=1_000_000)
    print(f"  Coin flip running mean (p=0.5):")
    print(f"  {'N':>9}  {'Mean':>10}  {'|err|':>8}")
    for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        mean = float(np.mean(flips[:n]))
        print(f"  {n:>9,}  {mean:>10.6f}  {abs(mean - 0.5):>8.6f}")
    print(f"  LLN: mean -> 0.5 as N -> inf (by strong LLN)")


if __name__ == "__main__":
    demo_axioms()
    demo_bayes()
    demo_conditional_independence()
    demo_monte_carlo_pi()
    demo_law_total_probability()
    demo_birthday_paradox()
    demo_conditional_prob_table()
    demo_probability_inequalities()
    demo_law_of_large_numbers()
