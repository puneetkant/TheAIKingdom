"""
Working Example 2: Operators — Real-World Use Cases
=====================================================
Demonstrates Python operators in data science and ML contexts:
  - Bitwise ops for flags/masks on real genomics-style data
  - Operator overloading for a custom Vector class
  - Numpy array operators on downloaded housing data
  - Chained comparisons for data validation

Run:  python working_example2.py
"""
import urllib.request
import csv
from pathlib import Path
from math import sqrt, isfinite
from operator import itemgetter

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# -- 1. Custom Vector with operator overloading --------------------------------
class Vector:
    """2D vector with full operator overloading."""
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __repr__(self):             return f"Vector({self.x}, {self.y})"
    def __add__(self, other):       return Vector(self.x + other.x, self.y + other.y)
    def __sub__(self, other):       return Vector(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar):      return Vector(self.x * scalar, self.y * scalar)
    def __rmul__(self, scalar):     return self.__mul__(scalar)
    def __truediv__(self, scalar):  return Vector(self.x / scalar, self.y / scalar)
    def __neg__(self):              return Vector(-self.x, -self.y)
    def __abs__(self):              return sqrt(self.x**2 + self.y**2)
    def __eq__(self, other):        return (self.x, self.y) == (other.x, other.y)
    def __lt__(self, other):        return abs(self) < abs(other)
    def dot(self, other):           return self.x * other.x + self.y * other.y


def demo_vector_operators():
    print("=== Operator Overloading — 2D Vector ===")
    u = Vector(3, 4)
    v = Vector(1, -2)
    print(f"  u = {u}, |u| = {abs(u)}")
    print(f"  v = {v}, |v| = {abs(v):.4f}")
    print(f"  u + v = {u + v}")
    print(f"  u - v = {u - v}")
    print(f"  3 * u = {3 * u}")
    print(f"  u / 2 = {u / 2}")
    print(f"  u.dot(v) = {u.dot(v)}")
    print(f"  u == Vector(3,4): {u == Vector(3, 4)}")
    print(f"  u < v (by magnitude): {u < v}")

    # Sort vectors by magnitude
    vectors = [Vector(5, 0), Vector(1, 1), Vector(3, 4), Vector(2, 2)]
    sorted_v = sorted(vectors)
    print(f"  Sorted by |v|: {sorted_v}")


# -- 2. Bitwise operators — feature flags -------------------------------------
def demo_bitwise_flags():
    print("\n=== Bitwise Operators — Feature Flags ===")
    # Imagine ML model features encoded as bitmask
    NUMERIC   = 0b0001   # 1
    TEXT      = 0b0010   # 2
    IMAGE     = 0b0100   # 4
    TEMPORAL  = 0b1000   # 8

    model_a = NUMERIC | TEXT             # 0b0011 = 3  (uses numeric + text)
    model_b = NUMERIC | IMAGE | TEMPORAL  # 0b1101 = 13

    print(f"  NUMERIC={NUMERIC:04b}  TEXT={TEXT:04b}  IMAGE={IMAGE:04b}  TEMPORAL={TEMPORAL:04b}")
    print(f"  model_a features ({model_a:04b}={model_a}): ", end="")
    features = {"NUMERIC": NUMERIC, "TEXT": TEXT, "IMAGE": IMAGE, "TEMPORAL": TEMPORAL}
    print(", ".join(k for k, v in features.items() if model_a & v))

    print(f"  model_b features ({model_b:04b}={model_b}): ", end="")
    print(", ".join(k for k, v in features.items() if model_b & v))

    both = model_a & model_b
    print(f"  Shared features (AND={both:04b}): {', '.join(k for k, v in features.items() if both & v)}")
    print(f"  Left shift  : NUMERIC << 2 = {NUMERIC << 2:04b} = {NUMERIC << 2}")
    print(f"  Right shift : model_b >> 1 = {model_b >> 1:04b} = {model_b >> 1}")


# -- 3. Download Boston housing -> operator-heavy data analysis -----------------
HOUSING_URL = (
    "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/"
    "cal_housing.csv"
)


def download_housing() -> Path:
    dest = DATA_DIR / "cal_housing.csv"
    if dest.exists():
        return dest
    print("\nDownloading California Housing dataset …")
    try:
        urllib.request.urlretrieve(HOUSING_URL, dest)
        print(f"  [OK] {dest.stat().st_size // 1024} KB saved")
    except Exception as e:
        print(f"  [X] Download failed ({e}). Using synthetic data.")
        import random
        random.seed(42)
        lines = ["MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedHouseVal"]
        for _ in range(200):
            lines.append(",".join([
                str(round(random.uniform(1, 15), 4)),
                str(random.randint(1, 52)),
                str(round(random.uniform(3, 10), 4)),
                str(round(random.uniform(0.8, 3), 4)),
                str(random.randint(100, 5000)),
                str(round(random.uniform(1, 10), 4)),
                str(round(random.uniform(32, 42), 4)),
                str(round(random.uniform(-124, -114), 4)),
                str(round(random.uniform(0.5, 5), 4)),
            ]))
        dest.write_text("\n".join(lines))
    return dest


def demo_operator_data_analysis(path: Path) -> None:
    print("\n=== Operators on Housing Data ===")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = {k: float(v) for k, v in row.items() if isfinite(float(v))}
                rows.append(r)
            except (ValueError, KeyError):
                pass

    print(f"  Loaded {len(rows)} rows")
    # Chained comparisons for validation
    med_inc = [r["MedInc"] for r in rows]
    valid   = [x for x in med_inc if 0.5 <= x <= 15.0]
    print(f"  Chained comparison (0.5 <= MedInc <= 15): {len(valid)}/{len(rows)} valid")

    # Arithmetic operators for feature engineering
    for r in rows:
        r["bedroom_ratio"]    = r["AveBedrms"] / r["AveRooms"]          # /
        r["rooms_per_person"] = r["AveRooms"] / r["AveOccup"]           # /
        r["is_expensive"]     = r["MedHouseVal"] > 4.0                  # comparison

    # Augmented assignment
    total_expensive = 0
    for r in rows:
        total_expensive += int(r["is_expensive"])   # +=
    print(f"  Expensive houses (>$400k): {total_expensive} ({total_expensive/len(rows):.1%})")

    # Sorted with key function (operator.itemgetter)
    top5 = sorted(rows, key=itemgetter("MedHouseVal"), reverse=True)[:5]
    print(f"  Top 5 most expensive (MedHouseVal):")
    for r in top5:
        print(f"    MedInc={r['MedInc']:.2f}  Value={r['MedHouseVal']:.3f}  "
              f"Rooms={r['AveRooms']:.1f}")


# -- 4. Operator precedence demo ------------------------------------------------
def demo_precedence():
    print("\n=== Operator Precedence ===")
    examples = [
        ("2 + 3 * 4",           2 + 3 * 4,          "* before +"),
        ("(2 + 3) * 4",         (2 + 3) * 4,        "() overrides"),
        ("2 ** 3 ** 2",         2 ** 3 ** 2,        "** is right-assoc"),
        ("-2 ** 2",             -2 ** 2,            "** before unary -"),
        ("3 > 2 and 5 > 4",     3 > 2 and 5 > 4,   "comparison before and"),
        ("True or False and False", True or False and False, "and before or"),
        ("not True or True",    not True or True,   "not before or"),
        ("4 & 5 | 2",           4 & 5 | 2,          "& before |"),
    ]
    for expr, result, note in examples:
        print(f"  {expr:<30} = {result!r:<8}  ({note})")


if __name__ == "__main__":
    demo_vector_operators()
    demo_bitwise_flags()
    path = download_housing()
    demo_operator_data_analysis(path)
    demo_precedence()
