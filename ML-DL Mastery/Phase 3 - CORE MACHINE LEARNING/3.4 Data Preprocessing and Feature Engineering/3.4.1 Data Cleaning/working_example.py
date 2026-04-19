"""
Working Example: Data Cleaning
Covers missing value detection, imputation strategies, outlier handling,
duplicate removal, data type fixes, and inconsistency detection.
"""
import numpy as np
import os

# Use only stdlib + numpy/scipy (pandas optional but used for demonstrations)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_cleaning")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Missing value detection ------------------------------------------------
def missing_value_detection():
    print("=== Missing Value Detection ===")
    rng = np.random.default_rng(0)
    n   = 100
    X   = rng.standard_normal((n, 5)).astype(object)

    # Inject NaNs
    for _ in range(30):
        i, j = rng.integers(0, n), rng.integers(0, 5)
        X[i, j] = np.nan

    X = X.astype(float)
    missing = np.isnan(X)
    print(f"  Data shape: {X.shape}")
    print(f"  Total missing: {missing.sum()} ({100*missing.mean():.1f}%)")
    for j in range(5):
        col_miss = missing[:, j].sum()
        pct = 100 * col_miss / n
        print(f"    col_{j}: {col_miss} missing ({pct:.1f}%)")

    # Missing patterns
    row_miss = missing.sum(axis=1)
    print(f"\n  Rows with 0 missing: {(row_miss==0).sum()}")
    print(f"  Rows with 1+ missing: {(row_miss>0).sum()}")
    print(f"  Rows with all missing: {(row_miss==5).sum()}")
    return X


# -- 2. Imputation strategies --------------------------------------------------
def imputation_strategies(X):
    print("\n=== Imputation Strategies ===")
    from sklearn.impute import SimpleImputer, KNNImputer

    strategies = {
        "Mean":    SimpleImputer(strategy="mean"),
        "Median":  SimpleImputer(strategy="median"),
        "Most frequent": SimpleImputer(strategy="most_frequent"),
        "Constant(0)": SimpleImputer(strategy="constant", fill_value=0),
        "KNN(k=5)":  KNNImputer(n_neighbors=5),
    }

    print(f"  {'Strategy':<20} {'Mean after imp.':<22} {'Std after imp.'}")
    for name, imp in strategies.items():
        X_imp = imp.fit_transform(X)
        print(f"  {name:<20} {X_imp.mean(0).round(3)}  {X_imp.std(0).round(3)}")

    # MCAR vs MNAR discussion
    print("\n  Missing data mechanisms:")
    print("    MCAR (Missing Completely At Random): imputation valid, any method OK")
    print("    MAR  (Missing At Random): depends on observed data, imputation valid")
    print("    MNAR (Missing Not At Random): systematic -> requires domain knowledge")

    return SimpleImputer(strategy="mean").fit_transform(X)


# -- 3. Outlier detection and treatment ---------------------------------------
def outlier_handling():
    print("\n=== Outlier Detection and Treatment ===")
    rng = np.random.default_rng(1)
    x   = rng.normal(50, 10, 200)
    # Inject outliers
    x   = np.concatenate([x, [-100, 200, 300, -200]])
    print(f"  Data: n={len(x)}  mean={x.mean():.2f}  std={x.std():.2f}  median={np.median(x):.2f}")

    # Z-score
    z        = np.abs((x - x.mean()) / x.std())
    z_mask   = z > 3
    print(f"\n  Z-score (|z|>3): {z_mask.sum()} outliers  -> {x[z_mask].round(1)}")

    # IQR
    Q1, Q3   = np.percentile(x, 25), np.percentile(x, 75)
    IQR      = Q3 - Q1
    iqr_mask = (x < Q1-1.5*IQR) | (x > Q3+1.5*IQR)
    print(f"  IQR method:      {iqr_mask.sum()} outliers  -> {x[iqr_mask].round(1)}")

    # Treatments
    x_clean      = x[~z_mask]                          # removal
    x_capped     = np.clip(x, Q1-1.5*IQR, Q3+1.5*IQR) # winsorization
    x_transformed = np.sign(x) * np.log1p(np.abs(x))  # log transform

    print(f"\n  Treatment effects on mean/std:")
    for name, arr in [("Original", x), ("After removal", x_clean),
                      ("After winsorize", x_capped), ("After log-transform", x_transformed)]:
        print(f"    {name:<22}: mean={arr.mean():.2f}  std={arr.std():.2f}")


# -- 4. Duplicate detection ----------------------------------------------------
def duplicate_handling():
    print("\n=== Duplicate Detection ===")
    # Simulate dataset with duplicates
    rng = np.random.default_rng(2)
    X   = rng.standard_normal((95, 4))
    # Add 5 exact duplicates
    dup_idx = rng.integers(0, 95, 5)
    X   = np.vstack([X, X[dup_idx]])
    print(f"  Total rows: {len(X)}")

    # Find exact duplicates
    seen   = set()
    is_dup = []
    for row in X:
        key = tuple(row.round(10))
        is_dup.append(key in seen)
        seen.add(key)
    is_dup = np.array(is_dup)
    print(f"  Exact duplicates: {is_dup.sum()}")

    # Near-duplicates (within tolerance)
    tol   = 0.01
    X_noisy = X + rng.normal(0, 0.001, X.shape)
    from scipy.spatial.distance import cdist
    dists = cdist(X_noisy, X_noisy)
    np.fill_diagonal(dists, np.inf)
    near_dup = (dists < tol).any(axis=1)
    print(f"  Near-duplicates (tol={tol}): {near_dup.sum()}")
    print(f"  After dedup: {(~is_dup).sum()} rows")


# -- 5. Data type fixes --------------------------------------------------------
def data_type_fixes():
    print("\n=== Data Type Issues ===")
    # Simulate mixed-type column
    raw_col = ["23", "45.5", " 67 ", "abc", "89", "-10", "nan", None, "100"]
    print(f"  Raw column: {raw_col}")

    cleaned = []
    errors  = []
    for val in raw_col:
        if val is None:
            errors.append(("None", "null"))
            cleaned.append(np.nan)
            continue
        val_stripped = str(val).strip()
        try:
            cleaned.append(float(val_stripped))
        except ValueError:
            errors.append((val_stripped, "non-numeric"))
            cleaned.append(np.nan)

    cleaned = np.array(cleaned, dtype=float)
    print(f"  Cleaned:    {cleaned}")
    print(f"  Parsing errors: {errors}")
    print(f"  NaN count: {np.isnan(cleaned).sum()}")
    print(f"  Valid stats: mean={np.nanmean(cleaned):.2f}  std={np.nanstd(cleaned):.2f}")


# -- 6. Inconsistency detection -----------------------------------------------
def inconsistency_detection():
    print("\n=== Inconsistency Detection ===")
    print("  Common inconsistencies in real datasets:")
    print()

    examples = {
        "Age out of range":       "age < 0 or age > 150",
        "Future dates":           "date > today",
        "Negative prices":        "price < 0",
        "Gender encoding mix":    "'M', 'male', 'Male', '0' — normalise to one form",
        "Contradictory fields":   "pregnant=True but gender=M",
        "Country-ZIP mismatch":   "zipcode does not match country",
        "Revenue < 0":            "transactions flagged as invalid",
    }
    for check, desc in examples.items():
        print(f"    {check:<30}: {desc}")

    print()
    # Numeric range check
    rng    = np.random.default_rng(3)
    ages   = rng.normal(35, 15, 200)
    ages   = np.concatenate([ages, [-5, 200, -20]])
    valid  = (ages >= 0) & (ages <= 120)
    print(f"  Age data: {(~valid).sum()} out-of-range values -> {ages[~valid].round(1)}")
    ages_clean = ages[valid]
    print(f"  After filter: n={len(ages_clean)}  mean={ages_clean.mean():.2f}")


# -- 7. Data cleaning pipeline ------------------------------------------------
def cleaning_pipeline():
    print("\n=== Data Cleaning Pipeline Summary ===")
    print("  1. Profile data: shape, dtypes, missing %, value counts")
    print("  2. Handle missing:  impute (mean/median/KNN) or drop rows/cols")
    print("  3. Fix dtypes:      parse strings, normalise categories")
    print("  4. Remove duplicates: exact + near-duplicates")
    print("  5. Handle outliers: Z-score, IQR, domain rules; clip/remove/transform")
    print("  6. Validate ranges: min/max, allowed values")
    print("  7. Resolve inconsistencies: cross-field rules")
    print("  8. Document changes: keep cleaning log for reproducibility")


if __name__ == "__main__":
    X = missing_value_detection()
    X_imputed = imputation_strategies(X)
    outlier_handling()
    duplicate_handling()
    data_type_fixes()
    inconsistency_detection()
    cleaning_pipeline()
