"""
Working Example 2: Pandas — Real-World Data Pipeline
=====================================================
Downloads Titanic (HuggingFace) and demonstrates production Pandas patterns:
  - Loading, dtypes, info/describe
  - Missing data: isnull, fillna, dropna strategies
  - Feature engineering with .assign() chaining
  - GroupBy aggregations and pivot tables
  - Merging DataFrames (left join)
  - Method chaining + query API
  - pd.cut / pd.qcut for binning

Run:  python working_example2.py
"""
import urllib.request
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    print("Run: pip install pandas numpy")
    raise SystemExit(1)

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# -- Download -------------------------------------------------------------------
def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {dest.name}")
    except Exception as e:
        print(f"Download failed ({e}); fallback to synthetic")
        pd.DataFrame({
            "PassengerId": range(1, 51),
            "Survived":    [i % 2 for i in range(50)],
            "Pclass":      [(i % 3) + 1 for i in range(50)],
            "Name":        [f"Person_{i}" for i in range(50)],
            "Sex":         ["male" if i % 2 else "female" for i in range(50)],
            "Age":         [20 + i % 40 if i % 5 != 0 else None for i in range(50)],
            "SibSp":       [i % 3 for i in range(50)],
            "Parch":       [i % 2 for i in range(50)],
            "Fare":        [7.0 + i * 2.5 for i in range(50)],
            "Embarked":    ["S" if i%3==0 else ("C" if i%3==1 else "Q") for i in range(50)],
        }).to_csv(dest, index=False)
    return dest


def demo_load_inspect(path: Path) -> pd.DataFrame:
    print("=== Load & Inspect ===")
    df = pd.read_csv(path)
    print(f"  Shape    : {df.shape}")
    print(f"  Dtypes   :\n{df.dtypes.to_string()}")
    print(f"\n  Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print(f"\n  Head:\n{df.head(3).to_string()}")
    return df


def demo_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Missing Data Handling ===")
    df = df.copy()
    # Strategies
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.dropna(subset=["Fare"])
    print(f"  After fill/drop  — Nulls remaining: {df.isnull().sum().sum()}")
    return df


def demo_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Feature Engineering (method chaining) ===")
    df = (
        df.assign(
            Title       = df["Name"].str.extract(r",\s*([^\.]+)\.").iloc[:, 0].str.strip(),
            FamilySize  = df["SibSp"] + df["Parch"] + 1,
            IsAlone     = lambda x: (x["SibSp"] + x["Parch"] == 0).astype(int),
            LogFare     = np.log1p(df["Fare"]),
            AgeBin      = pd.cut(df["Age"], bins=[0,12,18,35,60,100],
                                 labels=["child","teen","adult","middle","senior"]),
        )
    )
    print(f"  Title value counts:\n{df['Title'].value_counts().head(6).to_string()}")
    print(f"\n  AgeBin distribution:\n{df['AgeBin'].value_counts().sort_index().to_string()}")
    return df


def demo_groupby(df: pd.DataFrame) -> None:
    print("\n=== GroupBy Aggregations ===")
    summary = (
        df.groupby(["Pclass", "Sex"])["Survived"]
          .agg(["count", "sum", "mean"])
          .rename(columns={"count": "total", "sum": "survived", "mean": "rate"})
          .round(3)
    )
    print(summary.to_string())

    pivot = df.pivot_table(
        values="Survived",
        index="Pclass",
        columns="Sex",
        aggfunc="mean"
    ).round(3)
    print(f"\n  Pivot — survival rate by class × sex:\n{pivot.to_string()}")


def demo_merge(df: pd.DataFrame) -> None:
    print("\n=== Merge DataFrames ===")
    # Build a class-info lookup table
    class_info = pd.DataFrame({
        "Pclass":      [1, 2, 3],
        "ClassName":   ["First", "Second", "Third"],
        "MedianFare":  [df[df["Pclass"]==c]["Fare"].median() for c in [1,2,3]],
    })
    merged = df.merge(class_info, on="Pclass", how="left")
    print(f"  Shape after merge : {merged.shape}")
    print(f"  Sample ClassName  : {merged['ClassName'].value_counts().to_dict()}")


def demo_query_api(df: pd.DataFrame) -> None:
    print("\n=== Query API & Filtering ===")
    high_fare_survivors = df.query("Fare > 100 and Survived == 1 and Pclass == 1")
    print(f"  First class survivors (Fare>100): {len(high_fare_survivors)}")

    # Method chain
    top_earners = (
        df[["Name", "Fare", "Survived"]]
        .sort_values("Fare", ascending=False)
        .head(5)
    )
    print(f"\n  Top 5 fares:\n{top_earners.to_string(index=False)}")


if __name__ == "__main__":
    url  = "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv"
    path = download(url, DATA / "titanic.csv")

    df = demo_load_inspect(path)
    df = demo_missing(df)
    df = demo_feature_engineering(df)
    demo_groupby(df)
    demo_merge(df)
    demo_query_api(df)
