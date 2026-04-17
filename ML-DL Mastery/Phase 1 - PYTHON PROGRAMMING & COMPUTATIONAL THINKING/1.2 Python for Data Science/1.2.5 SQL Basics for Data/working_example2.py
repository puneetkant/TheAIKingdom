"""
Working Example 2: SQL Basics — Real Data in SQLite
====================================================
Downloads Titanic CSV, loads into SQLite3, then demonstrates:
  - CREATE TABLE, INSERT with executemany
  - SELECT with WHERE, ORDER BY, LIMIT
  - GROUP BY + HAVING for aggregations
  - JOIN (passengers + class_info lookup table)
  - Subqueries and CTEs (WITH)
  - UPDATE and parameterised queries (safe from SQL injection)
  - Exporting query results back to CSV

Run:  python working_example2.py
"""
import csv
import sqlite3
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
DB   = DATA / "titanic.db"


# ── 1. Download + load into SQLite ────────────────────────────────────────────
def download_titanic() -> Path:
    dest = DATA / "titanic.csv"
    if dest.exists(): return dest
    try:
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv",
            dest
        )
        print(f"Downloaded {dest.name}")
    except Exception:
        rows = ["PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked"]
        for i in range(1, 60):
            rows.append(
                f"{i},{i%2},{(i%3)+1},Person_{i},{'male' if i%2 else 'female'},"
                f"{20+i%40},{i%3},{i%2},{7.5+i*2:.2f},,{'S' if i%3==0 else 'C'}"
            )
        dest.write_text("\n".join(rows))
    return dest


def create_db(csv_path: Path) -> sqlite3.Connection:
    DB.unlink(missing_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")

    # ── passengers table ──
    conn.execute("""
        CREATE TABLE passengers (
            passenger_id INTEGER PRIMARY KEY,
            survived     INTEGER NOT NULL,
            pclass       INTEGER NOT NULL,
            name         TEXT    NOT NULL,
            sex          TEXT    NOT NULL,
            age          REAL,
            sib_sp       INTEGER DEFAULT 0,
            parch        INTEGER DEFAULT 0,
            fare         REAL,
            embarked     TEXT
        )
    """)

    # ── class_info lookup table ──
    conn.execute("""
        CREATE TABLE class_info (
            pclass      INTEGER PRIMARY KEY,
            class_name  TEXT NOT NULL,
            deck        TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO class_info VALUES (?, ?, ?)",
        [(1, "First",  "A-E"),
         (2, "Second", "D-F"),
         (3, "Third",  "G-T")]
    )

    # ── load passengers ──
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = []
        for r in reader:
            try:
                records.append((
                    int(r["PassengerId"]),
                    int(r["Survived"]),
                    int(r["Pclass"]),
                    r["Name"].strip(),
                    r["Sex"].strip(),
                    float(r["Age"]) if r["Age"].strip() else None,
                    int(r["SibSp"]) if r["SibSp"].strip() else 0,
                    int(r["Parch"]) if r["Parch"].strip() else 0,
                    float(r["Fare"]) if r["Fare"].strip() else None,
                    r.get("Embarked", "").strip() or None,
                ))
            except (ValueError, KeyError):
                continue
    conn.executemany(
        "INSERT INTO passengers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        records
    )
    conn.commit()
    print(f"  Loaded {len(records)} rows into {DB.name}")
    return conn


# ── 2. Demo queries ────────────────────────────────────────────────────────────
def demo_basic_queries(conn: sqlite3.Connection) -> None:
    print("\n=== Basic Queries ===")

    # Total + survived
    total, survived = conn.execute(
        "SELECT COUNT(*), SUM(survived) FROM passengers"
    ).fetchone()
    print(f"  Total: {total}  Survived: {survived}  Rate: {survived/total:.3f}")

    # WHERE filter (parameterised)
    pclass = 1
    first_class = conn.execute(
        "SELECT COUNT(*), AVG(fare) FROM passengers WHERE pclass = ?",
        (pclass,)
    ).fetchone()
    print(f"  Class {pclass}: {first_class[0]} passengers, avg fare={first_class[1]:.2f}")

    # ORDER BY + LIMIT
    top5 = conn.execute(
        "SELECT name, fare FROM passengers ORDER BY fare DESC LIMIT 5"
    ).fetchall()
    print("  Top 5 fares:")
    for name, fare in top5:
        print(f"    {name[:30]:<30}  £{fare:.2f}")


def demo_groupby(conn: sqlite3.Connection) -> None:
    print("\n=== GROUP BY + HAVING ===")
    rows = conn.execute("""
        SELECT pclass,
               sex,
               COUNT(*) AS n,
               ROUND(AVG(survived), 3) AS survival_rate,
               ROUND(AVG(fare), 2) AS avg_fare
        FROM   passengers
        GROUP  BY pclass, sex
        HAVING COUNT(*) > 5
        ORDER  BY pclass, sex
    """).fetchall()
    print(f"  {'Class':>6} {'Sex':<8} {'n':>5} {'Rate':>8} {'AvgFare':>10}")
    print("  " + "-" * 45)
    for row in rows:
        print(f"  {row[0]:>6} {row[1]:<8} {row[2]:>5} {row[3]:>8} {row[4]:>10}")


def demo_join(conn: sqlite3.Connection) -> None:
    print("\n=== JOIN ===")
    rows = conn.execute("""
        SELECT ci.class_name,
               ci.deck,
               COUNT(p.passenger_id) AS n,
               ROUND(AVG(p.survived), 3) AS rate
        FROM   passengers  p
        JOIN   class_info  ci ON p.pclass = ci.pclass
        GROUP  BY ci.class_name, ci.deck
        ORDER  BY ci.pclass
    """).fetchall()
    print(f"  {'Class':<10} {'Deck':<8} {'n':>5} {'SurvRate':>10}")
    print("  " + "-" * 40)
    for row in rows:
        print(f"  {row[0]:<10} {row[1]:<8} {row[2]:>5} {row[3]:>10}")


def demo_cte(conn: sqlite3.Connection) -> None:
    print("\n=== CTE (WITH clause) ===")
    rows = conn.execute("""
        WITH class_stats AS (
            SELECT pclass,
                   AVG(fare)     AS avg_fare,
                   AVG(survived) AS avg_surv
            FROM   passengers
            GROUP  BY pclass
        )
        SELECT p.name,
               p.fare,
               ROUND(p.fare - cs.avg_fare, 2) AS fare_diff_from_class_avg
        FROM   passengers    p
        JOIN   class_stats   cs ON p.pclass = cs.pclass
        WHERE  p.survived = 1
        ORDER  BY fare_diff_from_class_avg DESC
        LIMIT  5
    """).fetchall()
    print("  Top 5 survivors with fare above their class average:")
    for row in rows:
        print(f"    {row[0][:30]:<30}  fare={row[1]:.2f}  diff={row[2]:+.2f}")


def demo_update_and_safe_query(conn: sqlite3.Connection) -> None:
    print("\n=== Safe Parameterised UPDATE ===")
    # Add a derived column (simulate an UPDATE)
    conn.execute("ALTER TABLE passengers ADD COLUMN family_size INTEGER")
    conn.execute("UPDATE passengers SET family_size = sib_sp + parch + 1")
    conn.commit()

    large_fam = conn.execute(
        "SELECT COUNT(*) FROM passengers WHERE family_size >= ?", (4,)
    ).fetchone()[0]
    print(f"  Passengers in families ≥4: {large_fam}")


def export_to_csv(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT * FROM passengers LIMIT 20"
    ).fetchall()
    col_names = [d[0] for d in conn.execute("SELECT * FROM passengers LIMIT 1").description]
    out = DATA / "titanic_sql_export.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(col_names)
        w.writerows(rows)
    print(f"\n  Exported 20 rows to {out.name}")


if __name__ == "__main__":
    csv_path = download_titanic()
    print("=== SQLite: Titanic Database ===")
    conn = create_db(csv_path)

    demo_basic_queries(conn)
    demo_groupby(conn)
    demo_join(conn)
    demo_cte(conn)
    demo_update_and_safe_query(conn)
    export_to_csv(conn)
    conn.close()
