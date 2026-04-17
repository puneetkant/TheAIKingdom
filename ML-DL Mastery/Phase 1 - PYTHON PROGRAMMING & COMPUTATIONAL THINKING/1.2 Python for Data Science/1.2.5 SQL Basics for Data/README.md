# 1.2.5 SQL Basics for Data

SQLite3 in Python: CREATE TABLE, INSERT, SELECT, GROUP BY, JOIN, CTE, parameterised queries.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Basic SQLite CRUD with synthetic data |
| `working_example2.py` | Titanic CSV → SQLite: GROUP BY survival rates, JOIN, CTE, safe parameterised UPDATE |
| `working_example.ipynb` | Interactive: load CSV → SQL queries → aggregations |

## Run

```bash
python working_example.py
python working_example2.py    # creates data/titanic.db
jupyter lab working_example.ipynb
```

## SQL Quick Reference

```sql
-- CREATE
CREATE TABLE passengers (
    id INTEGER PRIMARY KEY,
    survived INTEGER, pclass INTEGER,
    age REAL, fare REAL
);

-- INSERT (safe with parameters from Python)
conn.executemany("INSERT INTO t VALUES (?,?,?,?,?)", records)

-- SELECT with filter + sort
SELECT name, fare FROM passengers
WHERE pclass = 1
ORDER BY fare DESC
LIMIT 10;

-- Aggregation
SELECT pclass, sex,
       COUNT(*) AS n,
       AVG(survived) AS surv_rate
FROM passengers
GROUP BY pclass, sex
HAVING COUNT(*) > 5;

-- JOIN
SELECT p.name, ci.class_name
FROM passengers p
JOIN class_info ci ON p.pclass = ci.pclass;

-- CTE
WITH stats AS (
    SELECT pclass, AVG(fare) AS avg_fare
    FROM passengers GROUP BY pclass
)
SELECT p.name, p.fare, p.fare - s.avg_fare AS diff
FROM passengers p JOIN stats s ON p.pclass = s.pclass;
```

## Security Note
Always use parameterised queries (`?` placeholders) — **never** f-string/concatenate user input into SQL (SQL injection risk).

## Dataset
- **Titanic** — [phihung/titanic on HuggingFace](https://huggingface.co/datasets/phihung/titanic)

## Learning Resources
- [SQLite docs](https://www.sqlite.org/lang.html)
- [Python sqlite3 docs](https://docs.python.org/3/library/sqlite3.html)
- [SQLBolt — interactive SQL tutorial](https://sqlbolt.com/)
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)
- **Book:** *Learning SQL* (Alan Beaulieu)
- **Cheatsheet:** [SQL cheatsheet (sqltutorial.org)](https://www.sqltutorial.org/sql-cheat-sheet/)

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
