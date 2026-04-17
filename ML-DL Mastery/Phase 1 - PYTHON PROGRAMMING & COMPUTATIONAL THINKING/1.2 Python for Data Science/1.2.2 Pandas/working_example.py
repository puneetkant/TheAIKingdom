"""
Working Example: Pandas
Covers Series, DataFrame creation, indexing, filtering,
groupby, merge/join, missing data, apply, and I/O.
"""
import io
import pandas as pd
import numpy as np


def series_demo():
    print("=== Pandas Series ===")
    s = pd.Series([10, 20, 30, 40, 50],
                  index=["a", "b", "c", "d", "e"],
                  name="values")
    print(s)
    print(f"\n  s['c']      = {s['c']}")
    print(f"  s[['a','e']]= {s[['a','e']].values}")
    print(f"  s > 25      = {s[s > 25].values}")
    print(f"  mean={s.mean()}, std={s.std():.2f}")
    print(f"  describe:\n{s.describe()}")


def dataframe_creation():
    print("\n=== DataFrame Creation ===")
    # From dict
    data = {
        "name"  : ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age"   : [30, 25, 35, 28, 32],
        "dept"  : ["Eng", "Marketing", "Eng", "HR", "Marketing"],
        "salary": [95000, 62000, 110000, 71000, 68000],
        "score" : [88.5, 74.0, 91.2, 79.3, np.nan],
    }
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    return df


def indexing_and_filtering(df):
    print("\n=== Indexing & Filtering ===")
    print(f"  df['name']:\n  {df['name'].values}")
    print(f"\n  df[['name','salary']]:\n{df[['name','salary']].to_string()}")
    print(f"\n  .loc (label): row 2\n{df.loc[2]}")
    print(f"\n  .iloc (pos): row 0:2\n{df.iloc[0:2].to_string()}")

    # Boolean filter
    eng = df[df['dept'] == "Eng"]
    print(f"\n  Engineering dept:\n{eng.to_string(index=False)}")

    high = df[(df['salary'] > 70000) & (df['age'] < 33)]
    print(f"\n  salary>70k AND age<33:\n{high.to_string(index=False)}")

    # query()
    mkt = df.query("dept == 'Marketing'")
    print(f"\n  query dept==Marketing:\n{mkt.to_string(index=False)}")


def missing_data(df):
    print("\n=== Missing Data ===")
    print(f"  isnull sum:\n{df.isnull().sum()}")

    df2 = df.copy()
    df2['score'].fillna(df2['score'].mean(), inplace=True)
    print(f"\n  after fillna(mean): score col = {df2['score'].values}")

    df3 = df.dropna()
    print(f"  after dropna: {len(df3)} rows remain (was {len(df)})")


def aggregation(df):
    print("\n=== Aggregation & GroupBy ===")
    print(f"  describe salary:\n{df['salary'].describe()}")

    grouped = df.groupby('dept')['salary'].agg(['mean', 'min', 'max', 'count'])
    print(f"\n  salary by dept:\n{grouped}")

    pivot = df.pivot_table(values='salary', index='dept',
                           aggfunc=['mean', 'count'])
    print(f"\n  pivot table:\n{pivot}")


def adding_columns(df):
    print("\n=== Adding / Transforming Columns ===")
    df2 = df.copy()
    df2['bonus']    = df2['salary'] * 0.10
    df2['level']    = df2['salary'].apply(lambda s: "Senior" if s > 80000 else "Junior")
    df2['score_ok'] = df2['score'].notna()
    print(df2[['name', 'salary', 'bonus', 'level', 'score_ok']].to_string(index=False))

    # np.where equivalent
    df2['age_group'] = np.where(df2['age'] >= 30, '30+', 'Under 30')
    print(f"\n  age_group: {df2['age_group'].values}")


def merge_and_join():
    print("\n=== Merge & Join ===")
    employees = pd.DataFrame({
        "emp_id": [1, 2, 3, 4],
        "name"  : ["Alice", "Bob", "Charlie", "Diana"],
        "dept_id": [10, 20, 10, 30],
    })
    departments = pd.DataFrame({
        "dept_id"  : [10, 20, 40],
        "dept_name": ["Engineering", "Marketing", "Finance"],
    })

    inner = pd.merge(employees, departments, on="dept_id", how="inner")
    left  = pd.merge(employees, departments, on="dept_id", how="left")
    print(f"  INNER merge:\n{inner.to_string(index=False)}")
    print(f"\n  LEFT  merge:\n{left.to_string(index=False)}")

    # concat
    df_a = pd.DataFrame({"x": [1,2], "y": [3,4]})
    df_b = pd.DataFrame({"x": [5,6], "y": [7,8]})
    stacked = pd.concat([df_a, df_b], ignore_index=True)
    print(f"\n  concat:\n{stacked}")


def sorting_and_ranking():
    print("\n=== Sorting & Ranking ===")
    data = {"name": ["Alice","Bob","Carol","Dave"],
            "score": [88, 95, 72, 95]}
    df = pd.DataFrame(data)
    print(f"  sort_values(score):\n{df.sort_values('score', ascending=False).to_string(index=False)}")
    df['rank'] = df['score'].rank(method='min', ascending=False).astype(int)
    print(f"  with rank:\n{df.to_string(index=False)}")


def io_demo():
    print("\n=== CSV I/O (in-memory) ===")
    csv_text = """name,age,score
Alice,30,88.5
Bob,25,74.0
Charlie,35,91.2
"""
    df = pd.read_csv(io.StringIO(csv_text))
    print(f"  read_csv:\n{df}")
    out = df.to_csv(index=False)
    print(f"\n  to_csv:\n{out}")

    print("  read_json / to_json:")
    json_str = df.to_json(orient="records", indent=2)
    df2 = pd.read_json(io.StringIO(json_str))
    print(f"  roundtrip rows: {len(df2)}")


if __name__ == "__main__":
    series_demo()
    df = dataframe_creation()
    indexing_and_filtering(df)
    missing_data(df)
    aggregation(df)
    adding_columns(df)
    merge_and_join()
    sorting_and_ranking()
    io_demo()
