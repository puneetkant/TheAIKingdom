"""
Working Example: Association Rule Learning
Covers frequent itemset mining (Apriori), association rules (support/confidence/lift),
FP-Growth concepts, and market basket analysis.
"""
import numpy as np
from itertools import combinations
from collections import defaultdict
import os


# ── 1. Market basket dataset ─────────────────────────────────────────────────
TRANSACTIONS = [
    {"milk", "bread", "butter"},
    {"beer", "bread"},
    {"milk", "bread", "butter", "cheese"},
    {"bread", "butter", "cheese"},
    {"milk", "cheese"},
    {"milk", "bread"},
    {"bread", "butter", "beer"},
    {"milk", "bread", "cheese"},
    {"butter", "cheese"},
    {"milk", "bread", "butter", "beer"},
]


# ── 2. Apriori from scratch ───────────────────────────────────────────────────
def apriori_scratch(transactions, min_support=0.3, min_confidence=0.6):
    print("=== Apriori Algorithm (from scratch) ===")
    print(f"  min_support={min_support}  min_confidence={min_confidence}")
    print(f"  Transactions: {len(transactions)}")
    print()

    n = len(transactions)

    def support(itemset):
        count = sum(1 for t in transactions if itemset.issubset(t))
        return count / n

    # 1-itemsets
    items = set(item for t in transactions for item in t)
    freq1 = {frozenset([i]): support(frozenset([i])) for i in items}
    freq1 = {k: v for k, v in freq1.items() if v >= min_support}

    print(f"  Frequent 1-itemsets ({len(freq1)}):")
    for k, v in sorted(freq1.items(), key=lambda x: -x[1]):
        print(f"    {set(k)}: support={v:.3f}")

    # Generate candidate k+1 itemsets from frequent k-itemsets
    def generate_candidates(freq_k):
        items_list = [list(fs) for fs in freq_k]
        candidates = set()
        for i, j in combinations(range(len(items_list)), 2):
            union = frozenset(items_list[i]) | frozenset(items_list[j])
            if len(union) == len(items_list[i]) + 1:
                candidates.add(union)
        return candidates

    all_frequent = dict(freq1)
    current_freq = freq1

    for k in range(2, len(items)+1):
        candidates = generate_candidates(current_freq)
        new_freq   = {c: support(c) for c in candidates if support(c) >= min_support}
        if not new_freq:
            break
        all_frequent.update(new_freq)
        current_freq = new_freq
        print(f"\n  Frequent {k}-itemsets ({len(new_freq)}):")
        for itemset, sup in sorted(new_freq.items(), key=lambda x: -x[1]):
            print(f"    {set(itemset)}: support={sup:.3f}")

    return all_frequent


# ── 3. Association rule generation ───────────────────────────────────────────
def generate_rules(all_frequent, transactions, min_confidence=0.6, min_lift=1.0):
    print(f"\n=== Association Rules (min_confidence={min_confidence}, min_lift={min_lift}) ===")
    n = len(transactions)

    def support(itemset):
        return sum(1 for t in transactions if itemset.issubset(t)) / n

    rules = []
    for itemset in all_frequent:
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                sup_both = support(itemset)
                sup_ant  = support(antecedent)
                sup_con  = support(consequent)
                if sup_ant == 0 or sup_con == 0:
                    continue
                conf = sup_both / sup_ant
                lift = conf / sup_con
                conv = (1 - sup_con) / (1 - conf + 1e-12) if conf < 1 else np.inf
                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": set(antecedent),
                        "consequent": set(consequent),
                        "support":    round(sup_both, 4),
                        "confidence": round(conf, 4),
                        "lift":       round(lift, 4),
                        "conviction": round(conv, 4),
                    })

    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    print(f"  Found {len(rules)} rules\n")
    print(f"  {'Antecedent':<25} → {'Consequent':<15} {'Supp':>7} {'Conf':>7} {'Lift':>7} {'Conv':>8}")
    for r in rules[:15]:
        print(f"  {str(r['antecedent']):<25} → {str(r['consequent']):<15} "
              f"{r['support']:>7.3f} {r['confidence']:>7.3f} {r['lift']:>7.3f} {r['conviction']:>8.3f}")
    return rules


# ── 4. Metrics explained ─────────────────────────────────────────────────────
def metrics_explained():
    print("\n=== Association Rule Metrics ===")
    print("  Support(A→B)    = P(A∪B)        — how often A and B occur together")
    print("  Confidence(A→B) = P(B|A)         — how often B when A occurs")
    print("  Lift(A→B)       = P(B|A)/P(B)    — >1: positive association")
    print("  Conviction      = (1-P(B))/(1-Conf) — directional, ∞ if conf=1")
    print("  Leverage        = P(A∪B)-P(A)·P(B)  — deviation from independence")
    print()
    print("  Lift interpretation:")
    print("    Lift > 1: A and B appear together more than expected (positive rule)")
    print("    Lift = 1: A and B are independent (useless rule)")
    print("    Lift < 1: A and B appear together less than expected (negative rule)")


# ── 5. FP-Growth concept ────────────────────────────────────────────────────
def fp_growth_concept():
    print("\n=== FP-Growth (Frequent Pattern Growth) ===")
    print("  Apriori: generates many candidate itemsets → expensive")
    print("  FP-Growth: compresses database into FP-tree → mine without candidate gen.")
    print()
    print("  Steps:")
    print("  1. Find all frequent items (min_support), sort by frequency")
    print("  2. Build FP-tree: insert transactions (filtered, sorted) into prefix tree")
    print("  3. Mine conditional pattern bases and conditional FP-trees recursively")
    print()
    print("  Advantage: typically 10–100× faster than Apriori on large sparse data")
    print()
    # Use mlxtend if available, else show concept only
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        import pandas as pd

        te = TransactionEncoder()
        te_arr = te.fit([list(t) for t in TRANSACTIONS]).transform([list(t) for t in TRANSACTIONS])
        df = pd.DataFrame(te_arr, columns=te.columns_)
        freq_items = fpgrowth(df, min_support=0.3, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
        print(f"  FP-Growth (mlxtend): {len(freq_items)} frequent itemsets  {len(rules)} rules")
        print(f"  Top rules by lift:")
        top = rules.nlargest(5, "lift")[["antecedents","consequents","support","confidence","lift"]]
        print(top.to_string(index=False))
    except ImportError:
        print("  (mlxtend not installed; install with: pip install mlxtend)")


# ── 6. Practical analysis ────────────────────────────────────────────────────
def practical_analysis():
    print("\n=== Practical Recommendations ===")
    print("  1. Start with min_support = 1-5% for large datasets")
    print("  2. Use min_confidence > 60% to avoid noisy rules")
    print("  3. Filter by lift > 1 to keep positively associated rules")
    print("  4. Use FP-Growth over Apriori for datasets > 1000 transactions")
    print("  5. Watch out for spurious correlations in large rule sets")
    print("  6. Conviction is more directional than lift — prefer for recommendations")
    print()
    print("  Common applications:")
    print("    - Market basket analysis (supermarket transactions)")
    print("    - Web click stream analysis (page co-visits)")
    print("    - Medical symptom/drug co-occurrence")
    print("    - Recommender systems (collaborative filtering fallback)")


if __name__ == "__main__":
    all_freq = apriori_scratch(TRANSACTIONS, min_support=0.3, min_confidence=0.6)
    rules = generate_rules(all_freq, TRANSACTIONS, min_confidence=0.6, min_lift=1.0)
    metrics_explained()
    fp_growth_concept()
    practical_analysis()
