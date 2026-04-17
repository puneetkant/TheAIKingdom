"""
Working Example 2: Basic Syntax & Data Types — Real-World Applications
========================================================================
Real-world data parsing scenarios: reading a CSV from Hugging Face,
cleaning mixed types, using Python's type system for data validation,
and building a simple typed data record.

Run:  python working_example2.py
"""
import sys
import os
import urllib.request
import csv
import io
from pathlib import Path
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── 1. Download and parse a real CSV (Titanic from HF) ───────────────────────
TITANIC_URL = (
    "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv"
)


def download_titanic() -> Path:
    dest = DATA_DIR / "titanic_train.csv"
    if dest.exists():
        print(f"Using cached: {dest}")
        return dest
    print(f"Downloading Titanic dataset …")
    try:
        urllib.request.urlretrieve(TITANIC_URL, dest)
        print(f"✓ Saved {dest.stat().st_size // 1024} KB to {dest}")
    except Exception as e:
        print(f"✗ Download failed ({e}). Creating synthetic fallback …")
        synthetic = (
            "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n"
            "1,0,3,Braund Mr. Owen Harris,male,22,1,0,A/5 21171,7.25,,S\n"
            "2,1,1,Cumings Mrs. John Bradley,female,38,1,0,PC 17599,71.2833,C85,C\n"
            "3,1,3,Heikkinen Miss. Laina,female,26,0,0,STON/O2. 3101282,7.925,,S\n"
            "4,1,1,Futrelle Mrs. Jacques Heath,female,35,1,0,113803,53.1,C123,S\n"
            "5,0,3,Allen Mr. William Henry,male,35,0,0,373450,8.05,,S\n"
        )
        dest.write_text(synthetic)
    return dest


def demo_type_coercion_on_real_csv(path: Path) -> list[dict]:
    """Parse CSV rows, demonstrating Python type coercions on real data."""
    print("\n=== Type Coercion on Titanic CSV ===")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader):
            if i >= 10:
                break
            # Demonstrate type conversions
            survived  = bool(int(raw.get("Survived", 0)))
            pclass    = int(raw.get("Pclass", 0))
            age_str   = raw.get("Age", "").strip()
            age: Optional[float] = float(age_str) if age_str else None
            fare_str  = raw.get("Fare", "").strip()
            try:
                fare = Decimal(fare_str) if fare_str else Decimal("0")
            except InvalidOperation:
                fare = Decimal("0")
            rows.append({
                "name":     raw.get("Name", ""),
                "survived": survived,
                "pclass":   pclass,
                "age":      age,
                "fare":     fare,
                "sex":      raw.get("Sex", ""),
            })
    print(f"  {'Name':<35} {'Sur':<5} {'Cls':<4} {'Age':<6} {'Fare'}")
    for r in rows:
        age_s = f"{r['age']:.0f}" if r["age"] is not None else "N/A"
        print(f"  {r['name'][:34]:<35} {str(r['survived']):<5} {r['pclass']:<4} {age_s:<6} £{r['fare']:.2f}")
    return rows


# ── 2. Typed dataclass (Python 3.10+ pattern) ─────────────────────────────────
@dataclass
class Passenger:
    passenger_id: int
    name: str
    survived: bool
    pclass: int
    sex: str
    age: Optional[float]
    fare: Decimal

    @property
    def title(self) -> str:
        """Extract title from name string."""
        parts = self.name.split(",")
        if len(parts) > 1:
            return parts[1].strip().split(".")[0]
        return "Unknown"

    @classmethod
    def from_dict(cls, d: dict) -> "Passenger":
        age_str = d.get("Age", "").strip()
        fare_str = d.get("Fare", "").strip()
        return cls(
            passenger_id=int(d.get("PassengerId", 0)),
            name=d.get("Name", ""),
            survived=bool(int(d.get("Survived", 0))),
            pclass=int(d.get("Pclass", 0)),
            sex=d.get("Sex", ""),
            age=float(age_str) if age_str else None,
            fare=Decimal(fare_str) if fare_str else Decimal("0"),
        )

    def __str__(self) -> str:
        age_s = f"{self.age:.0f}" if self.age is not None else "??"
        return (f"Passenger({self.name[:25]!r}, "
                f"title={self.title!r}, age={age_s}, "
                f"survived={self.survived})")


def demo_typed_dataclass(path: Path) -> None:
    print("\n=== Typed Dataclass (Python 3.10+) ===")
    passengers: list[Passenger] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            passengers.append(Passenger.from_dict(row))
    for p in passengers:
        print(f"  {p}")


# ── 3. String formatting methods compared ─────────────────────────────────────
def string_formatting_comparison() -> None:
    print("\n=== String Formatting Comparison ===")
    name  = "Titanic"
    count = 891
    rate  = 0.3838

    print("  %-style    :", "Dataset: %s, rows: %d, survival: %.1f%%" % (name, count, rate * 100))
    print("  .format()  :", "Dataset: {}, rows: {:,}, survival: {:.1%}".format(name, count, rate))
    print("  f-string   :", f"Dataset: {name}, rows: {count:,}, survival: {rate:.1%}")
    print("  Template   :", end=" ")
    from string import Template
    t = Template("Dataset: $n, rows: $c")
    print(t.substitute(n=name, c=count))

    # f-string advanced: alignment, fill, expressions
    print("\n  f-string alignment:")
    headers = ["Name", "Count", "Rate"]
    print("  " + "  ".join(f"{h:>10}" for h in headers))
    print("  " + "  ".join(f"{v:>10}" for v in [name, count, f"{rate:.2%}"]))


# ── 4. Python walrus operator and type matching (3.10+) ───────────────────────
def modern_python_syntax() -> None:
    print("\n=== Modern Python Syntax (3.8-3.12) ===")

    # Walrus operator (3.8+)
    data = [1, None, 3, None, 5, 6, None]
    non_null = [x for item in data if (x := item) is not None]
    print(f"  Walrus filter (non-null): {non_null}")

    # Structural pattern matching (3.10+)
    def classify(value):
        match value:
            case int() if value < 0:
                return "negative int"
            case int():
                return "non-negative int"
            case float():
                return "float"
            case str():
                return "string"
            case None:
                return "None"
            case _:
                return "other"

    test_values = [-5, 0, 42, 3.14, "hello", None, [1, 2]]
    for v in test_values:
        print(f"  classify({v!r:<10}) → {classify(v)}")

    # PEP 695 type aliases (3.12+)
    print("\n  Type annotations (PEP 604 union syntax, 3.10+):")
    def greet(name: str | None = None) -> str:
        return f"Hello, {name}!" if name else "Hello, stranger!"
    print(f"  {greet('Ada')}")
    print(f"  {greet()}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = download_titanic()
    demo_type_coercion_on_real_csv(path)
    demo_typed_dataclass(path)
    string_formatting_comparison()
    modern_python_syntax()
