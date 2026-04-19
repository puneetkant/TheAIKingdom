"""
Working Example: Probability Fundamentals
Covers sample spaces, axioms, conditional probability, independence,
Bayes' theorem, and the law of total probability.
"""
import numpy as np
from fractions import Fraction
import itertools


# -- 1. Sample space and events ------------------------------------------------
def sample_spaces():
    print("=== Sample Spaces and Events ===")
    # Roll two fair dice
    Omega = list(itertools.product(range(1,7), range(1,7)))
    n     = len(Omega)
    print(f"  Omega (two dice): {n} outcomes")

    # Events
    A = {s for s in Omega if s[0]+s[1] == 7}   # sum = 7
    B = {s for s in Omega if s[0] == s[1]}      # doubles

    print(f"  A = {{sum=7}}: |A|={len(A)}  P(A)={Fraction(len(A), n)}")
    print(f"  B = {{doubles}}: |B|={len(B)}  P(B)={Fraction(len(B), n)}")
    print(f"  AnB: {A&B}")
    print(f"  AuB: |AuB|={len(A|B)}  P(AuB)={Fraction(len(A|B), n)}")
    print(f"  Aᶜ:  |Aᶜ|={n-len(A)}")


# -- 2. Kolmogorov axioms verification ----------------------------------------
def kolmogorov_axioms():
    print("\n=== Kolmogorov's Axioms ===")
    print("  1. P(A) >= 0  for any event A")
    print("  2. P(Omega) = 1")
    print("  3. If AnB = {} then P(AuB) = P(A) + P(B)  (sigma-additivity)")
    print()
    # Verify with uniform distribution on {1,...,6}
    probs = {i: Fraction(1,6) for i in range(1,7)}
    print(f"  Fair die probabilities: {probs}")
    print(f"  Sum = {sum(probs.values())} = 1  [OK]")

    A = {1, 2, 3}; B = {4, 5}   # disjoint
    pA = sum(probs[x] for x in A)
    pB = sum(probs[x] for x in B)
    pAuB = sum(probs[x] for x in A|B)
    print(f"  A={A} P(A)={pA}   B={B} P(B)={pB}")
    print(f"  AnB={A&B}  (disjoint)   P(AuB)={pAuB}  = P(A)+P(B)={pA+pB}  [OK]")


# -- 3. Conditional probability ------------------------------------------------
def conditional_probability():
    print("\n=== Conditional Probability P(A|B) = P(AnB)/P(B) ===")
    # Card deck: 52 cards
    suits = ['♠','♥','♦','♣']
    ranks = list(range(1,14))   # 1=Ace,11=Jack,...
    deck  = [(r,s) for r in ranks for s in suits]
    N     = len(deck)

    A = {c for c in deck if c[0] == 1}    # Aces
    B = {c for c in deck if c[1] == '♠'}  # Spades

    pA    = len(A) / N
    pB    = len(B) / N
    pAB   = len(A & B) / N   # Ace of Spades
    pA_gB = pAB / pB          # P(Ace | Spade)
    pB_gA = pAB / pA          # P(Spade | Ace)

    print(f"  Deck: {N} cards")
    print(f"  P(Ace)           = {pA:.4f} = {Fraction(4,52)}")
    print(f"  P(Spade)         = {pB:.4f} = {Fraction(13,52)}")
    print(f"  P(Ace n Spade)   = {pAB:.4f} = {Fraction(1,52)}")
    print(f"  P(Ace | Spade)   = {pA_gB:.4f} = {Fraction(1,13)}")
    print(f"  P(Spade | Ace)   = {pB_gA:.4f} = {Fraction(1,4)}")


# -- 4. Independence -----------------------------------------------------------
def independence():
    print("\n=== Independence: P(AnB) = P(A)·P(B) ===")
    # Two fair coins
    Omega  = list(itertools.product(['H','T'], repeat=2))
    N      = len(Omega)
    A      = {s for s in Omega if s[0]=='H'}   # first coin heads
    B      = {s for s in Omega if s[1]=='H'}   # second coin heads
    AB     = A & B

    pA, pB, pAB = len(A)/N, len(B)/N, len(AB)/N
    print(f"  P(A) = {pA}, P(B) = {pB}, P(AnB) = {pAB}")
    print(f"  P(A)·P(B) = {pA*pB}   independent? {np.isclose(pAB, pA*pB)}")

    # Conditional independence
    print("\n  Conditional independence: P(AnB|C) = P(A|C)·P(B|C)")
    print("  Example: two dice rolls are independent given nothing.")
    print("  But knowing sum=7 makes them dependent!")
    Omega2 = list(itertools.product(range(1,7), repeat=2))
    A2 = {s for s in Omega2 if s[0]==4}   # first die = 4
    C  = {s for s in Omega2 if s[0]+s[1]==7}
    pA2_gC   = len(A2&C)/len(C)
    print(f"  P(die1=4 | sum=7) = {pA2_gC:.4f}  (not 1/6 = {1/6:.4f})")


# -- 5. Bayes' theorem --------------------------------------------------------
def bayes_theorem():
    print("\n=== Bayes' Theorem P(H|E) = P(E|H)·P(H) / P(E) ===")
    # Medical test example
    prev    = 0.01     # prior P(disease)
    sens    = 0.95     # sensitivity P(test+ | disease)
    spec    = 0.95     # specificity P(test- | no disease)
    fp_rate = 1-spec   # P(test+ | no disease)

    # P(test+) = sens·prev + fp_rate·(1-prev)
    p_pos   = sens*prev + fp_rate*(1-prev)
    # Posterior P(disease | test+)
    posterior = sens * prev / p_pos

    print(f"  Disease prevalence  P(D)  = {prev}")
    print(f"  Sensitivity P(+|D)        = {sens}")
    print(f"  Specificity P(-|D)        = {spec}")
    print(f"  P(test+)                  = {p_pos:.4f}")
    print(f"  P(D | test+)              = {posterior:.4f}  (posterior)")
    print(f"\n  Despite 95% accurate test, positive result only ~{posterior:.0%} likely disease")
    print(f"  (due to low prevalence — base-rate fallacy)")


# -- 6. Law of total probability -----------------------------------------------
def total_probability():
    print("\n=== Law of Total Probability ===")
    # P(B) = Sigma P(B|Ai)·P(Ai)  where {Ai} partition Omega
    # Factory example: 3 machines producing defective parts
    machines = [
        {"name": "M1", "output_frac": 0.5, "defect_rate": 0.02},
        {"name": "M2", "output_frac": 0.3, "defect_rate": 0.05},
        {"name": "M3", "output_frac": 0.2, "defect_rate": 0.03},
    ]
    print("  Factory defect example:")
    for m in machines:
        print(f"    {m['name']}: {m['output_frac']*100:.0f}% of output, {m['defect_rate']*100:.0f}% defect rate")

    p_defect = sum(m['output_frac'] * m['defect_rate'] for m in machines)
    print(f"\n  P(defective) = {p_defect:.4f}")

    # Bayes: given defective, which machine?
    print("\n  P(machine | defective):")
    for m in machines:
        p_mach_given_defect = (m['defect_rate'] * m['output_frac']) / p_defect
        print(f"    P({m['name']} | defective) = {p_mach_given_defect:.4f}")


# -- 7. Monte Carlo probability estimation -------------------------------------
def monte_carlo():
    print("\n=== Monte Carlo Probability Estimation ===")
    rng = np.random.default_rng(0)
    N   = 1_000_000

    # Estimate pi via circle inscribed in unit square
    xs, ys = rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)
    inside = (xs**2 + ys**2 <= 1).sum()
    pi_est = 4 * inside / N
    print(f"  pi estimate (N={N:,}): {pi_est:.5f}  (true: {np.pi:.5f})")

    # P(two dice sum > 8)
    d1 = rng.integers(1, 7, N)
    d2 = rng.integers(1, 7, N)
    p_est  = (d1 + d2 > 8).mean()
    p_true = sum(1 for a in range(1,7) for b in range(1,7) if a+b>8) / 36
    print(f"  P(sum > 8): MC={p_est:.4f}  exact={p_true:.4f}")


if __name__ == "__main__":
    sample_spaces()
    kolmogorov_axioms()
    conditional_probability()
    independence()
    bayes_theorem()
    total_probability()
    monte_carlo()
