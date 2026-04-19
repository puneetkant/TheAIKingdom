"""
Working Example: Encoding Categorical Variables
Covers ordinal encoding, one-hot encoding, label encoding, target encoding,
binary encoding, frequency encoding, and handling high cardinality.
"""
import numpy as np
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder, LabelEncoder,
                                   TargetEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import os


# -- 1. Why encoding matters ---------------------------------------------------
def why_encoding():
    print("=== Why We Encode Categorical Variables ===")
    print("  ML models require numeric inputs.")
    print("  Encoding options depend on:")
    print("    - Is there a natural ordering? (ordinal vs nominal)")
    print("    - How many categories? (cardinality)")
    print("    - What model? (tree vs linear)")
    print("    - Label leakage risk? (target encoding needs careful cross-val)")
    print()
    categories_example = ["red", "green", "blue", "red", "blue"]
    print(f"  Example: {categories_example}")
    unique = sorted(set(categories_example))
    ohe = {v: [1 if v==u else 0 for u in unique] for v in unique}
    print(f"  One-hot encoding: {ohe}")


# -- 2. Label Encoding (ordinal/integer) --------------------------------------
def label_encoding():
    print("\n=== Label Encoding ===")
    print("  Assigns integer to each category: {'A':0, 'B':1, 'C':2}")
    print("  Risk: introduces artificial ordering for nominal features!")
    print("  OK for: target variable, truly ordinal features, tree-based models")

    sizes = np.array(["S", "M", "L", "XL", "M", "S", "L"])
    le = LabelEncoder()
    encoded = le.fit_transform(sizes)
    print(f"\n  Sizes:   {sizes}")
    print(f"  Encoded: {encoded}  (classes: {le.classes_})")

    # Ordinal with correct order
    size_order = {"S": 0, "M": 1, "L": 2, "XL": 3}
    ordinal    = np.array([size_order[s] for s in sizes])
    print(f"  Ordinal: {ordinal}  (correct order preserved)")


# -- 3. One-Hot Encoding -------------------------------------------------------
def one_hot_encoding():
    print("\n=== One-Hot Encoding ===")
    print("  Creates a binary column per category: no ordering implied")
    print("  Drawback: high dimensionality for high-cardinality features")

    colors = np.array(["red", "green", "blue", "red", "green"]).reshape(-1, 1)
    ohe    = OneHotEncoder(sparse_output=False, drop="first")  # drop=first avoids dummy trap
    X_ohe  = ohe.fit_transform(colors)

    print(f"\n  Categories: {ohe.categories_}")
    print(f"  Feature names: {ohe.get_feature_names_out()}")
    print(f"  Original:     {colors.ravel()}")
    print(f"  Encoded shape: {X_ohe.shape}")
    print(f"  First 5 rows:")
    for row, orig in zip(X_ohe[:5], colors[:5, 0]):
        print(f"    {orig:<8} -> {row}")

    print()
    print("  drop='first': removes one column to avoid perfect multicollinearity")
    print("  handle_unknown='ignore': set unseen categories to 0 at test time")


# -- 4. Ordinal Encoding -------------------------------------------------------
def ordinal_encoding():
    print("\n=== Ordinal Encoding ===")
    print("  Use when categories have a meaningful order")

    education = np.array([["high school"], ["bachelor"], ["master"], ["phd"],
                           ["bachelor"], ["high school"]])
    oe = OrdinalEncoder(categories=[["high school", "bachelor", "master", "phd"]])
    X_ord = oe.fit_transform(education)
    print(f"\n  {'Level':<15} {'Encoded'}")
    for orig, enc in zip(education[:,0], X_ord[:,0]):
        print(f"  {orig:<15} {int(enc)}")


# -- 5. Target Encoding --------------------------------------------------------
def target_encoding():
    print("\n=== Target Encoding ===")
    print("  Replace category with mean target value (+ smoothing to avoid leakage)")
    print("  Risk: leakage if not done with cross-validation!")

    rng = np.random.default_rng(0)
    n   = 500
    cities = rng.choice(["NY","LA","Chicago","NY","LA","Houston"], n)
    # Salary correlated with city
    salary = {"NY": 80000, "LA": 75000, "Chicago": 65000, "Houston": 60000}
    y      = np.array([salary[c] + rng.normal(0, 10000) for c in cities])

    # Simple target encoding (training set only — would leak in practice)
    X_city = cities.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X_city, y, test_size=0.3, random_state=0)

    # Use sklearn TargetEncoder
    te = TargetEncoder(random_state=0)
    X_tr_enc = te.fit_transform(X_tr, y_tr)
    X_te_enc  = te.transform(X_te)

    print(f"\n  Learned encodings: {dict(zip(te.categories_[0], te.encodings_[0].round(0)))}")
    print(f"  True means:        {salary}")

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_tr_enc, y_tr)
    rmse = np.sqrt(np.mean((model.predict(X_te_enc) - y_te)**2))
    print(f"  Linear Regression RMSE with target encoding: {rmse:.0f}")


# -- 6. Frequency / Count Encoding --------------------------------------------
def frequency_encoding():
    print("\n=== Frequency Encoding ===")
    print("  Replace category with its frequency in the dataset")
    print("  Preserves cardinality information without OHE dimensionality explosion")

    rng  = np.random.default_rng(1)
    data = rng.choice(["A","B","C","D","E"], 100,
                      p=[0.4, 0.3, 0.15, 0.10, 0.05])
    from collections import Counter
    counts = Counter(data)
    n      = len(data)
    freq_map = {k: v/n for k, v in counts.items()}
    encoded = np.array([freq_map[d] for d in data])

    print(f"\n  Category frequencies: {dict((k, round(v,3)) for k,v in freq_map.items())}")
    print(f"  First 10 encoded: {encoded[:10].round(3)}")


# -- 7. Binary Encoding (for high cardinality) --------------------------------
def binary_encoding():
    print("\n=== Binary Encoding (high cardinality) ===")
    print("  LabelEncode then convert integer to binary bits")
    print("  For n categories: log2(n) bits vs n bits for OHE")

    categories = [f"cat_{i}" for i in range(16)]  # 16 categories
    le = LabelEncoder()
    int_codes = le.fit_transform(categories)

    n_bits = int(np.ceil(np.log2(len(categories))))
    print(f"\n  {len(categories)} categories -> {n_bits} binary bits (OHE would need {len(categories)} cols)")
    print(f"  {'Category':<12} {'Int code':<12} {'Binary encoding'}")
    for cat, code in zip(categories[:8], int_codes[:8]):
        bits = [int(b) for b in format(code, f'0{n_bits}b')]
        print(f"  {cat:<12} {code:<12} {bits}")


# -- 8. ColumnTransformer pipeline --------------------------------------------
def column_transformer_pipeline():
    print("\n=== ColumnTransformer Pipeline ===")
    rng = np.random.default_rng(2)
    n   = 300
    # Simulate mixed dataset
    col_num  = rng.standard_normal((n, 2))      # 2 numeric
    col_cat1 = rng.choice(["A","B","C"], n)      # 3-cat nominal
    col_cat2 = rng.choice(["low","med","high"],n) # ordinal

    X_num = col_num
    y     = (col_num[:,0] + (col_cat1=="A").astype(float) > 0.5).astype(int)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer(transformers=[
        ("num",      StandardScaler(),
                     [0, 1]),  # numeric columns
        ("cat_nom",  OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                     [2]),     # nominal cat (imaginary column index)
        ("cat_ord",  OrdinalEncoder(categories=[["low","med","high"]]),
                     [3]),     # ordinal cat
    ], remainder="drop")

    # Build synthetic X with all 4 columns
    import numpy as np
    X_full = np.column_stack([col_num,
                               col_cat1,
                               col_cat2])

    pipeline = Pipeline([
        ("pre", ColumnTransformer(transformers=[
            ("num", StandardScaler(), [0,1]),
            ("nom", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [2]),
            ("ord", OrdinalEncoder(categories=[["low","med","high"]]), [3]),
        ])),
        ("clf", LogisticRegression(max_iter=500)),
    ])

    cv = cross_val_score(pipeline, X_full, y, cv=5).mean()
    print(f"  Mixed-type pipeline (LR) CV accuracy: {cv:.4f}")
    print(f"  (Demonstrates seamless categorical + numeric handling)")


if __name__ == "__main__":
    why_encoding()
    label_encoding()
    one_hot_encoding()
    ordinal_encoding()
    target_encoding()
    frequency_encoding()
    binary_encoding()
    column_transformer_pipeline()
