"""
Working Example 2: Control Flow — Real-World Applications
==========================================================
Control flow applied to ML training loops, data validation pipelines,
and processing a real Titanic dataset.

Run:  python working_example2.py
"""
import urllib.request
import csv
from pathlib import Path
from collections import defaultdict
from itertools import islice

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# -- 1. Download data ----------------------------------------------------------
def get_titanic() -> list[dict]:
    dest = DATA_DIR / "titanic.csv"
    if not dest.exists():
        url = "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv"
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            dest.write_text(
                "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
                "1,0,3,Braund Mr. Owen Harris,male,22,1,0,A/5 21171,7.25,,S\n"
                "2,1,1,Cumings Mrs. John Bradley,female,38,1,0,PC 17599,71.2833,C85,C\n"
                "3,1,3,Heikkinen Miss. Laina,female,26,0,0,STON/O2.,7.925,,S\n"
                "4,1,1,Futrelle Mrs. Jacques Heath,female,35,1,0,113803,53.1,C123,S\n"
                "5,0,3,Allen Mr. William Henry,male,35,0,0,373450,8.05,,S\n"
            )
    rows = []
    with open(dest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# -- 2. Data validation pipeline (if/elif chains) ------------------------------
def validate_passengers(rows: list[dict]) -> tuple[list, list]:
    print("=== Data Validation Pipeline ===")
    valid, invalid = [], []
    reasons: dict[str, int] = defaultdict(int)

    for row in rows:
        errors = []
        # Age check
        age_str = row.get("Age", "").strip()
        if not age_str:
            errors.append("missing age")
        elif not (0 < float(age_str) < 120):
            errors.append(f"invalid age={age_str}")

        # Fare check
        fare_str = row.get("Fare", "").strip()
        if not fare_str or float(fare_str) < 0:
            errors.append("invalid fare")

        # Class check
        pclass = row.get("Pclass", "")
        if pclass not in ("1", "2", "3"):
            errors.append("invalid class")

        if errors:
            for e in errors:
                reasons[e] += 1
            invalid.append(row)
        else:
            valid.append(row)

    print(f"  Valid   : {len(valid)}")
    print(f"  Invalid : {len(invalid)}")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count}")
    return valid, invalid


# -- 3. Simulate ML training loop ----------------------------------------------
def simulate_training_loop():
    print("\n=== Simulated Training Loop ===")
    import math, random
    random.seed(42)

    n_epochs  = 20
    patience  = 4
    best_loss = float("inf")
    no_improve = 0

    print(f"  {'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Status'}")
    for epoch in range(1, n_epochs + 1):
        # Simulate decaying loss with noise
        train_loss = 1.0 * math.exp(-epoch * 0.15) + random.gauss(0, 0.02)
        val_loss   = 1.0 * math.exp(-epoch * 0.13) + random.gauss(0, 0.03)

        if val_loss < best_loss - 0.001:
            best_loss  = val_loss
            no_improve = 0
            status = "[OK] improved"
        else:
            no_improve += 1
            status = f"no improve ({no_improve}/{patience})"

        print(f"  {epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} {status}")

        if no_improve >= patience:
            print(f"  -> Early stopping at epoch {epoch}")
            break
    print(f"  Best val loss: {best_loss:.4f}")


# -- 4. for/while/break/continue on real data ----------------------------------
def analyze_passengers(valid: list[dict]) -> None:
    print("\n=== Control Flow on Passenger Data ===")

    # enumerate + for
    print("  First 5 survivors (using enumerate + continue):")
    count = 0
    for i, p in enumerate(valid):
        if p.get("Survived") != "1":
            continue
        print(f"    [{i}] {p['Name'][:35]}, Age={p['Age']}, Fare={float(p['Fare']):.2f}")
        count += 1
        if count >= 5:
            break

    # while loop: find first 1st class female survivor
    print("\n  while: find first 1st-class female survivor:")
    idx = 0
    while idx < len(valid):
        p = valid[idx]
        if p["Pclass"] == "1" and p["Sex"] == "female" and p["Survived"] == "1":
            print(f"    Found at index {idx}: {p['Name'][:40]}, Age={p['Age']}")
            break
        idx += 1
    else:
        print("    Not found!")

    # Nested loop: survival matrix
    print("\n  Nested loop — survival rate matrix (Sex × Pclass):")
    print(f"  {'':8}", end="")
    for cls in ("1", "2", "3"):
        print(f"  Class {cls}", end="")
    print()
    for sex in ("male", "female"):
        print(f"  {sex:<8}", end="")
        for cls in ("1", "2", "3"):
            subset = [p for p in valid if p["Sex"] == sex and p["Pclass"] == cls]
            if subset:
                rate = sum(1 for p in subset if p["Survived"] == "1") / len(subset)
                print(f"  {rate:>6.1%} ", end="")
            else:
                print(f"  {'N/A':>6}  ", end="")
        print()

    # zip: pair consecutive passengers
    print("\n  zip: age difference between consecutive passenger pairs (first 5):")
    ages = [float(p["Age"]) for p in valid]
    for a, b in islice(zip(ages, ages[1:]), 5):
        print(f"    |{a:.0f} - {b:.0f}| = {abs(a - b):.0f}")


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    rows = get_titanic()
    valid, invalid = validate_passengers(rows)
    simulate_training_loop()
    analyze_passengers(valid)
