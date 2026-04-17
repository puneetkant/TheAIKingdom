"""
Working Example: SQL Basics for Data
Covers SQLite (stdlib), DML, DDL, joins, aggregation,
subqueries, window functions, and pandas SQL integration.
"""
import sqlite3
import pandas as pd
from contextlib import closing


# ── Setup: create an in-memory SQLite database ────────────────────────────────
def get_connection():
    return sqlite3.connect(":memory:")


def setup_schema(conn):
    with closing(conn.cursor()) as cur:
        cur.executescript("""
            CREATE TABLE departments (
                dept_id   INTEGER PRIMARY KEY,
                dept_name TEXT NOT NULL
            );
            CREATE TABLE employees (
                emp_id    INTEGER PRIMARY KEY,
                name      TEXT NOT NULL,
                age       INTEGER,
                salary    REAL,
                dept_id   INTEGER REFERENCES departments(dept_id)
            );
            CREATE TABLE projects (
                proj_id   INTEGER PRIMARY KEY,
                title     TEXT,
                emp_id    INTEGER REFERENCES employees(emp_id)
            );

            INSERT INTO departments VALUES
                (10, 'Engineering'), (20, 'Marketing'),
                (30, 'Sales'),       (40, 'HR');

            INSERT INTO employees VALUES
                (1, 'Alice',   30, 95000, 10),
                (2, 'Bob',     25, 62000, 20),
                (3, 'Charlie', 35, 110000, 10),
                (4, 'Diana',   28, 71000, 30),
                (5, 'Eve',     32, 68000, 20),
                (6, 'Frank',   45, 88000, 10),
                (7, 'Grace',   29, NULL,   40);

            INSERT INTO projects VALUES
                (101, 'AI Platform',      1),
                (102, 'Ad Campaign',      2),
                (103, 'ML Pipeline',      3),
                (104, 'CRM Integration',  4),
                (105, 'Data Dashboard',   1);
        """)
        conn.commit()


def run_query(conn, sql, label=""):
    df = pd.read_sql_query(sql, conn)
    print(f"  {label}:")
    print(df.to_string(index=False))
    print()
    return df


# ── DDL / DML ─────────────────────────────────────────────────────────────────
def dml_demo(conn):
    print("=== DML — INSERT / UPDATE / DELETE ===")
    with closing(conn.cursor()) as cur:
        # INSERT
        cur.execute("INSERT INTO employees VALUES (8, 'Heidi', 31, 75000, 30)")
        print("  Inserted Heidi into employees")

        # UPDATE
        cur.execute("UPDATE employees SET salary = salary * 1.10 WHERE dept_id = 10")
        print("  10% raise for Engineering")

        # DELETE
        cur.execute("DELETE FROM employees WHERE emp_id = 7")
        print("  Deleted Grace (NULL salary)")
        conn.commit()

    run_query(conn, "SELECT * FROM employees ORDER BY emp_id", "employees table")


# ── Basic SELECT ──────────────────────────────────────────────────────────────
def select_demo(conn):
    print("=== SELECT ===")
    run_query(conn, "SELECT name, age, salary FROM employees ORDER BY salary DESC LIMIT 5",
              "top 5 by salary")

    run_query(conn, "SELECT * FROM employees WHERE age BETWEEN 28 AND 35 AND salary > 70000",
              "age 28-35 AND salary > 70k")

    run_query(conn, "SELECT DISTINCT dept_id FROM employees ORDER BY dept_id",
              "DISTINCT dept_ids")


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregation_demo(conn):
    print("=== Aggregation & GROUP BY ===")
    run_query(conn, """
        SELECT dept_id,
               COUNT(*)          AS headcount,
               ROUND(AVG(salary),2) AS avg_salary,
               MIN(salary)       AS min_sal,
               MAX(salary)       AS max_sal
        FROM employees
        WHERE salary IS NOT NULL
        GROUP BY dept_id
        HAVING COUNT(*) >= 2
        ORDER BY avg_salary DESC
    """, "dept stats (headcount≥2)")


# ── JOINs ────────────────────────────────────────────────────────────────────
def joins_demo(conn):
    print("=== JOINs ===")
    run_query(conn, """
        SELECT e.name, e.salary, d.dept_name
        FROM employees e
        INNER JOIN departments d ON e.dept_id = d.dept_id
        ORDER BY e.salary DESC
    """, "INNER JOIN employees × departments")

    run_query(conn, """
        SELECT d.dept_name, e.name, p.title
        FROM departments d
        LEFT JOIN employees e ON d.dept_id = e.dept_id
        LEFT JOIN projects  p ON e.emp_id  = p.emp_id
        ORDER BY d.dept_name, e.name
    """, "LEFT JOIN departments → employees → projects")


# ── Subqueries ────────────────────────────────────────────────────────────────
def subquery_demo(conn):
    print("=== Subqueries ===")
    run_query(conn, """
        SELECT name, salary
        FROM employees
        WHERE salary > (SELECT AVG(salary) FROM employees)
        ORDER BY salary DESC
    """, "employees earning above average salary")

    run_query(conn, """
        SELECT name FROM employees
        WHERE emp_id IN (SELECT emp_id FROM projects)
    """, "employees who have at least one project (IN subquery)")


# ── Window functions ──────────────────────────────────────────────────────────
def window_functions_demo(conn):
    print("=== Window Functions ===")
    run_query(conn, """
        SELECT
            name,
            dept_id,
            salary,
            RANK()    OVER (PARTITION BY dept_id ORDER BY salary DESC)  AS dept_rank,
            AVG(salary) OVER (PARTITION BY dept_id)                     AS dept_avg,
            salary - AVG(salary) OVER (PARTITION BY dept_id)            AS vs_dept_avg,
            ROW_NUMBER() OVER (ORDER BY salary DESC)                    AS overall_row
        FROM employees
        WHERE salary IS NOT NULL
        ORDER BY dept_id, dept_rank
    """, "window: RANK, AVG per dept, ROW_NUMBER")


# ── pandas + SQL ──────────────────────────────────────────────────────────────
def pandas_sql_integration(conn):
    print("=== pandas ↔ SQLite ===")
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    print(f"  pd.read_sql_query → DataFrame shape: {df.shape}")
    print(df.head(3).to_string(index=False))

    # Write a new DataFrame to SQL
    new_data = pd.DataFrame({
        "dept_id": [10, 20],
        "kpi"    : [0.92, 0.78],
    })
    new_data.to_sql("dept_kpis", conn, if_exists="replace", index=False)
    run_query(conn, "SELECT * FROM dept_kpis", "dept_kpis table from DataFrame")


if __name__ == "__main__":
    conn = get_connection()
    setup_schema(conn)
    dml_demo(conn)
    select_demo(conn)
    aggregation_demo(conn)
    joins_demo(conn)
    subquery_demo(conn)
    window_functions_demo(conn)
    pandas_sql_integration(conn)
    conn.close()
