"""
Working Example: Feature Engineering
Covers polynomial features, interaction terms, domain-specific feature creation,
datetime features, binning, log/sqrt transforms, and feature crossing.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import os


# -- 1. Polynomial features ----------------------------------------------------
def polynomial_features():
    print("=== Polynomial Feature Expansion ===")
    print("  Adds x², x³, x1x2, ... to capture non-linear relationships")
    rng = np.random.default_rng(0)
    n   = 200
    x   = rng.uniform(-2, 2, (n, 2))
    y   = x[:,0]**2 + x[:,1]**2 + 0.5*x[:,0]*x[:,1] + rng.normal(0, 0.2, n)

    for degree in [1, 2, 3]:
        poly  = PolynomialFeatures(degree=degree, include_bias=False)
        X_p   = poly.fit_transform(x)
        model = LinearRegression().fit(X_p, y)
        rmse  = np.sqrt(np.mean((model.predict(X_p) - y)**2))
        print(f"  degree={degree}: {X_p.shape[1]} features  train RMSE={rmse:.4f}")
        if degree <= 2:
            print(f"    Feature names: {poly.get_feature_names_out(['x0','x1'])}")


# -- 2. Interaction features ---------------------------------------------------
def interaction_features():
    print("\n=== Interaction Features ===")
    print("  x1 x x2 captures synergistic effects between features")
    rng   = np.random.default_rng(1)
    n     = 300
    price = rng.uniform(10, 100, n)
    area  = rng.uniform(30, 200, n)
    # Price per square meter
    value = price / area + rng.normal(0, 0.05, n)

    # Without interaction
    X_base = np.column_stack([price, area])
    X_int  = np.column_stack([price, area, price*area, price/area])

    for name, X in [("price+area", X_base), ("+ price×area + price/area", X_int)]:
        cv = cross_val_score(LinearRegression(), X, value, cv=5,
                             scoring="neg_root_mean_squared_error")
        print(f"  {name:<35}: CV RMSE={-cv.mean():.4f}")


# -- 3. Domain feature engineering example (housing) --------------------------
def domain_features():
    print("\n=== Domain Feature Engineering (Housing) ===")
    rng = np.random.default_rng(2)
    n   = 500
    sqft  = rng.uniform(500, 4000, n)
    baths = rng.choice([1, 1.5, 2, 2.5, 3], n).astype(float)
    age   = rng.uniform(0, 80, n)
    floors= rng.choice([1,2,3], n).astype(float)
    price = 100*sqft + 30000*baths - 1000*age + 50000*floors + rng.normal(0, 25000, n)

    X_raw  = np.column_stack([sqft, baths, age, floors])
    # Engineered features
    sqft_per_bath = sqft / (baths + 0.1)
    age_squared   = age**2
    sqft_log      = np.log1p(sqft)
    bath_floor    = baths * floors
    X_eng = np.column_stack([sqft, baths, age, floors,
                              sqft_per_bath, age_squared, sqft_log, bath_floor])

    for name, X in [("Raw features", X_raw), ("Engineered features", X_eng)]:
        cv = cross_val_score(LinearRegression(), X, price, cv=5,
                             scoring="neg_root_mean_squared_error")
        print(f"  {name:<25}: CV RMSE={-cv.mean():.0f}")


# -- 4. Datetime feature extraction --------------------------------------------
def datetime_features():
    print("\n=== Datetime Feature Extraction ===")
    import datetime
    # Simulate timestamps
    base = datetime.datetime(2023, 1, 1)
    dates = [base + datetime.timedelta(hours=i*7+13) for i in range(200)]

    features = []
    for d in dates:
        features.append({
            "hour":         d.hour,
            "day_of_week":  d.weekday(),       # 0=Mon
            "month":        d.month,
            "quarter":      (d.month-1)//3 + 1,
            "is_weekend":   int(d.weekday() >= 5),
            "sin_hour":     np.sin(2*np.pi*d.hour/24),  # cyclic encoding
            "cos_hour":     np.cos(2*np.pi*d.hour/24),
            "sin_day":      np.sin(2*np.pi*d.weekday()/7),
            "cos_day":      np.cos(2*np.pi*d.weekday()/7),
        })

    print(f"  Extracted {len(features[0])} features from timestamps")
    for k, v in features[0].items():
        print(f"    {k:<15}: {v:.4f}" if isinstance(v, float) else f"    {k:<15}: {v}")

    print()
    print("  Cyclic encoding (sin/cos): avoids jump between 23->0 hour or Sun->Mon")


# -- 5. Binning (discretisation) -----------------------------------------------
def binning_demo():
    print("\n=== Binning (Discretisation) ===")
    print("  Converts continuous variable into discrete bins")
    rng  = np.random.default_rng(3)
    ages = rng.normal(40, 15, 300).clip(0, 100).reshape(-1, 1)

    for strategy in ["uniform", "quantile", "kmeans"]:
        kbin = KBinsDiscretizer(n_bins=5, strategy=strategy, encode="ordinal")
        X_b  = kbin.fit_transform(ages)
        print(f"  strategy={strategy}: bin_edges={kbin.bin_edges_[0].round(1)}")
        print(f"    Bin counts: {np.bincount(X_b[:,0].astype(int))}")

    # Manual age groups
    age_arr = ages.ravel()
    groups  = np.digitize(age_arr, bins=[0, 18, 35, 55, 75, 101]) - 1
    labels  = ["<18","18-35","35-55","55-75","75+"]
    print(f"\n  Manual age groups: {[labels[g] for g in groups[:10]]}")


# -- 6. Log and sqrt transforms -----------------------------------------------
def transform_skewed():
    print("\n=== Transforms for Skewed Features ===")
    rng   = np.random.default_rng(4)
    price = rng.exponential(50000, 500)   # right-skewed (house prices)

    from scipy.stats import skew
    print(f"  Original: skewness={skew(price):.4f}  range=[{price.min():.0f},{price.max():.0f}]")

    transforms = {
        "log(x+1)":      np.log1p(price),
        "sqrt(x)":       np.sqrt(price),
        "x^(1/3)":       price**(1/3),
        "1/x":           1/(price + 1),
        "Box-Cox (lambda=0)": np.log(price + 1),
    }
    print(f"  {'Transform':<18} {'Skewness':>12}  {'Range'}")
    for name, arr in transforms.items():
        print(f"  {name:<18} {skew(arr):>12.4f}  [{arr.min():.3f}, {arr.max():.3f}]")


# -- 7. Feature crossing -------------------------------------------------------
def feature_crossing():
    print("\n=== Feature Crossing (categorical × categorical) ===")
    print("  Combine two categorical features into a single interaction feature")
    print("  Common in wide-and-deep models, CTR prediction")

    rng     = np.random.default_rng(5)
    country = np.array(rng.choice(["US","UK","DE"], 100))
    browser = np.array(rng.choice(["Chrome","Firefox","Safari"], 100))

    # Cross feature
    crossed = np.array([f"{c}_{b}" for c, b in zip(country, browser)])
    unique_crossed = sorted(set(crossed))
    print(f"\n  Countries: {sorted(set(country))}")
    print(f"  Browsers:  {sorted(set(browser))}")
    print(f"  Crossed ({len(unique_crossed)} unique): {unique_crossed}")

    # Encode crossed feature
    cross_map = {v: i for i, v in enumerate(unique_crossed)}
    cross_enc = np.array([cross_map[c] for c in crossed])
    print(f"\n  First 10 encoded: {cross_enc[:10]}")
    print(f"  (These can be one-hot encoded or target-encoded for downstream models)")


# -- 8. Feature engineering impact --------------------------------------------
def engineering_impact():
    print("\n=== Feature Engineering Impact Summary ===")
    rng = np.random.default_rng(6)
    n   = 400
    # Non-linear dataset
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y  = (x1**2 + x2**2 < 1).astype(int)   # inside unit circle

    X_raw = np.column_stack([x1, x2])
    X_eng = np.column_stack([x1, x2, x1**2, x2**2, x1*x2])

    print(f"  Task: circle classifier  (LR can't solve without poly features)")
    for name, X in [("LR (raw x1,x2)",     X_raw),
                    ("LR (+ x1²,x2²,x1x2)",X_eng)]:
        cv  = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=5).mean()
        print(f"  {name:<35}: CV acc={cv:.4f}")

    cv_rf = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=0), X_raw, y, cv=5).mean()
    print(f"  {'RF (raw x1,x2) — no FE needed':<35}: CV acc={cv_rf:.4f}")


if __name__ == "__main__":
    polynomial_features()
    interaction_features()
    domain_features()
    datetime_features()
    binning_demo()
    transform_skewed()
    feature_crossing()
    engineering_impact()
