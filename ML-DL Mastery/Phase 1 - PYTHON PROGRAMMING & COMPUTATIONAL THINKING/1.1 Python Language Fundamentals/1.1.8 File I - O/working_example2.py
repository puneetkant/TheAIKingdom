"""
Working Example 2: File I/O — Real-World Data Pipeline
=======================================================
Demonstrates production file-handling patterns:
  - Download Titanic CSV → parse with csv module
  - Write cleaned data back to CSV
  - Serialize/deserialize run artifacts as JSON
  - Read/write binary files (pickle)
  - pathlib-first file management
  - Context managers and safe file writes

Run:  python working_example2.py
"""
import csv
import json
import pickle
import hashlib
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT  = BASE / "output"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


# ── 1. Download & parse CSV ────────────────────────────────────────────────────
def download_titanic() -> Path:
    dest = DATA / "titanic.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv",
                dest
            )
            print(f"Downloaded {dest.name}")
        except Exception as e:
            # synthetic fallback
            dest.write_text(
                "PassengerId,Survived,Pclass,Name,Sex,Age,Fare\n"
                + "\n".join(f"{i},1,1,Person {i},male,{20+i},{50+i}" for i in range(1, 30))
            )
    return dest


def load_and_parse_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return list(reader.fieldnames or []), rows  # fieldnames captured inside


def demo_csv(path: Path) -> list[dict]:
    print("=== CSV: Load and Parse ===")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Rows loaded: {len(rows)}")
    print(f"  Columns    : {list(rows[0].keys())}")

    # Filter and clean
    cleaned = []
    for r in rows:
        try:
            age  = float(r.get("Age") or 0)
            fare = float(r.get("Fare") or 0)
            cleaned.append({
                "id":       int(r.get("PassengerId", 0)),
                "survived": int(r.get("Survived", 0)),
                "class":    int(r.get("Pclass", 0)),
                "sex":      r.get("Sex", "").strip(),
                "age":      age,
                "fare":     fare,
            })
        except ValueError:
            continue

    print(f"  After clean: {len(cleaned)} rows")

    # Write cleaned CSV
    out_csv = OUT / "titanic_cleaned.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cleaned[0].keys())
        writer.writeheader()
        writer.writerows(cleaned)
    print(f"  Written     : {out_csv}")
    return cleaned


# ── 2. JSON: save run artifact ────────────────────────────────────────────────
def demo_json(rows: list[dict]) -> None:
    print("\n=== JSON: Run Artifact ===")

    # Compute summary stats
    survived = [r for r in rows if r["survived"] == 1]
    artifact = {
        "run_id":       datetime.now(timezone.utc).isoformat(),
        "total_rows":   len(rows),
        "survived":     len(survived),
        "survival_rate": round(len(survived) / len(rows), 4),
        "avg_fare":     round(sum(r["fare"] for r in rows) / len(rows), 2),
        "checksum":     hashlib.sha256(str(rows).encode()).hexdigest()[:16],
    }

    path = OUT / "run_artifact.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Written: {path}")

    # Round-trip
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"  Loaded back: {loaded}")


# ── 3. Pickle: binary serialisation ──────────────────────────────────────────
def demo_pickle(rows: list[dict]) -> None:
    print("\n=== Pickle: Binary Serialisation ===")

    path = OUT / "titanic_rows.pkl"
    with open(path, "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = path.stat().st_size / 1024
    print(f"  Pickled {len(rows)} rows → {size_kb:.1f} KB")

    with open(path, "rb") as f:
        loaded = pickle.load(f)
    print(f"  Loaded  {len(loaded)} rows from pickle")
    print(f"  First row: {loaded[0]}")


# ── 4. pathlib operations ─────────────────────────────────────────────────────
def demo_pathlib() -> None:
    print("\n=== pathlib: File Management ===")
    p = OUT
    print(f"  Output dir: {p.resolve()}")
    print(f"  Files     :")
    for f in sorted(p.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name:<35}  {size:>8,} bytes")

    # Safe atomic write pattern (write to .tmp then rename)
    tmp  = OUT / "report.txt.tmp"
    final = OUT / "report.txt"
    tmp.write_text("Run complete.\n")
    tmp.replace(final)
    print(f"\n  Atomic write: {final.name} ({final.stat().st_size} bytes)")

    # glob all JSON files
    jsons = list(OUT.glob("*.json"))
    print(f"  JSON files  : {[f.name for f in jsons]}")


if __name__ == "__main__":
    csv_path = download_titanic()
    rows = demo_csv(csv_path)
    demo_json(rows)
    demo_pickle(rows)
    demo_pathlib()
