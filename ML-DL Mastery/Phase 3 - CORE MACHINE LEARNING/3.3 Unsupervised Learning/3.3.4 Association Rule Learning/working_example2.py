"""
Working Example 2: Association Rule Learning — Apriori, Frequent Itemsets
==========================================================================
Market basket analysis with mlxtend (or manual implementation).

Run:  python working_example2.py
"""
import itertools
from collections import defaultdict
from pathlib import Path

def load_transactions():
    """Sample supermarket transactions."""
    return [
        ["milk", "bread", "butter"],
        ["beer", "bread"],
        ["milk", "bread"],
        ["beer", "diapers"],
        ["milk", "diapers", "bread", "butter"],
        ["beer", "bread", "diapers"],
        ["milk", "diapers", "beer"],
        ["milk", "bread", "diapers"],
        ["beer", "diapers"],
        ["milk", "bread", "butter", "diapers"],
    ]

def get_itemsets(transactions, min_support):
    n = len(transactions)
    txn_sets = [set(t) for t in transactions]

    # All items
    all_items = sorted({item for txn in transactions for item in txn})
    freq = {}

    # k=1
    for item in all_items:
        sup = sum(1 for t in txn_sets if item in t) / n
        if sup >= min_support:
            freq[frozenset([item])] = sup

    # k=2,3,...
    k = 2
    current = list(freq.keys())
    while current:
        candidates = []
        for a, b in itertools.combinations(current, 2):
            union = a | b
            if len(union) == k:
                candidates.append(union)
        current = []
        for cand in set(map(frozenset, [list(c) for c in candidates])):
            sup = sum(1 for t in txn_sets if cand <= t) / n
            if sup >= min_support:
                freq[cand] = sup
                current.append(cand)
        k += 1
    return freq

def generate_rules(freq, min_confidence):
    rules = []
    for itemset in freq:
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in map(frozenset, itertools.combinations(itemset, r)):
                consequent = itemset - antecedent
                conf = freq[itemset] / freq[antecedent]
                lift = conf / freq[consequent]
                if conf >= min_confidence:
                    rules.append((antecedent, consequent, freq[itemset], conf, lift))
    return sorted(rules, key=lambda x: x[4], reverse=True)

def demo_association_rules():
    print("=== Association Rule Mining ===")
    transactions = load_transactions()
    min_support    = 0.3
    min_confidence = 0.5

    freq = get_itemsets(transactions, min_support)
    print(f"\nFrequent itemsets (support >= {min_support}):")
    for itemset, sup in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        print(f"  {set(itemset)}: support={sup:.2f}")

    rules = generate_rules(freq, min_confidence)
    print(f"\nAssociation rules (conf >= {min_confidence}):")
    print(f"  {'Antecedent':20s} -> {'Consequent':15s} | sup    conf   lift")
    for ant, con, sup, conf, lift in rules:
        print(f"  {str(set(ant)):20s} -> {str(set(con)):15s} | {sup:.2f}  {conf:.2f}  {lift:.2f}")

def demo_mlxtend():
    print("\n=== mlxtend (if available) ===")
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import apriori, association_rules
        transactions = [
            {"milk":1,"bread":1,"butter":1,"beer":0,"diapers":0},
            {"milk":0,"bread":1,"butter":0,"beer":1,"diapers":0},
            {"milk":1,"bread":1,"butter":0,"beer":0,"diapers":0},
            {"milk":0,"bread":0,"butter":0,"beer":1,"diapers":1},
            {"milk":1,"bread":1,"butter":1,"beer":0,"diapers":1},
        ]
        df = pd.DataFrame(transactions).astype(bool)
        freq = apriori(df, min_support=0.4, use_colnames=True)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        print(rules[["antecedents","consequents","support","confidence","lift"]].to_string(index=False))
    except ImportError:
        print("  mlxtend not installed (pip install mlxtend)")

if __name__ == "__main__":
    demo_association_rules()
    demo_mlxtend()
