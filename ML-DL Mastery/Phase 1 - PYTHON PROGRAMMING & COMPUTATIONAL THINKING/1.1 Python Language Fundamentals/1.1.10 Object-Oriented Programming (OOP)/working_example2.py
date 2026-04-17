"""
Working Example 2: OOP — ML Model Class Hierarchy
==================================================
Builds a mini sklearn-style estimator hierarchy:
  - Abstract BaseEstimator with fit/predict/score interface
  - KNNClassifier and NaiveBayes (pure-Python implementations)
  - Pipeline class (composition over inheritance)
  - Mixin classes: ReprMixin, SerializableMixin
  - Dunder methods: __repr__, __len__, __call__, __iter__

Downloads Iris dataset from HuggingFace and trains/evaluates.

Run:  python working_example2.py
"""
import csv
import json
import math
import urllib.request
import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# ── Utility ───────────────────────────────────────────────────────────────────
def download_iris() -> Path:
    dest = DATA / "iris.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/scikit-learn/iris/resolve/main/Iris.csv",
                dest
            )
        except Exception:
            # synthetic 30-row fallback
            rows = ["SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species"]
            specs = [("Iris-setosa", 5.0, 3.5, 1.4, 0.2),
                     ("Iris-versicolor", 6.0, 2.9, 4.5, 1.5),
                     ("Iris-virginica", 6.5, 3.0, 5.5, 2.0)]
            for i in range(30):
                s, a, b, c, d = specs[i % 3]
                rows.append(f"{a+i*.1:.1f},{b:.1f},{c+i*.05:.1f},{d:.1f},{s}")
            dest.write_text("\n".join(rows))
    return dest


def load_iris(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    X, y = [], []
    for r in rows:
        try:
            X.append([float(r["SepalLengthCm"]), float(r["SepalWidthCm"]),
                      float(r["PetalLengthCm"]), float(r["PetalWidthCm"])])
            y.append(r["Species"].strip())
        except (ValueError, KeyError):
            pass
    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    import random; random.seed(seed)
    idx = list(range(len(X))); random.shuffle(idx)
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return [X[i] for i in tr], [X[i] for i in te], [y[i] for i in tr], [y[i] for i in te]


# ── Mixins ────────────────────────────────────────────────────────────────────
class ReprMixin:
    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def get_params(self) -> dict:
        return {}


class SerializableMixin:
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "SerializableMixin":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Abstract base ─────────────────────────────────────────────────────────────
class BaseEstimator(ABC, ReprMixin, SerializableMixin):
    """Abstract base — defines the fit/predict/score API."""
    _is_fitted: bool = False

    @abstractmethod
    def fit(self, X: list, y: list) -> "BaseEstimator": ...

    @abstractmethod
    def predict(self, X: list) -> list: ...

    def score(self, X: list, y: list) -> float:
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)

    def __call__(self, X: list) -> list:
        """Allow estimator(X) as shorthand for predict(X)."""
        return self.predict(X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} not fitted — call fit() first")


# ── KNN Classifier ────────────────────────────────────────────────────────────
class KNNClassifier(BaseEstimator):
    """k-Nearest Neighbours (pure Python, Euclidean distance)."""

    def __init__(self, k: int = 5):
        self.k = k
        self._X_train: list = []
        self._y_train: list = []

    def get_params(self) -> dict:
        return {"k": self.k}

    def fit(self, X: list, y: list) -> "KNNClassifier":
        self._X_train = X
        self._y_train = y
        self._is_fitted = True
        return self

    def _euclidean(self, a: list, b: list) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def predict(self, X: list) -> list:
        self._check_fitted()
        preds = []
        for x in X:
            dists = [(self._euclidean(x, xt), yt)
                     for xt, yt in zip(self._X_train, self._y_train)]
            dists.sort(key=lambda t: t[0])
            k_labels = [lab for _, lab in dists[: self.k]]
            preds.append(Counter(k_labels).most_common(1)[0][0])
        return preds


# ── Naive Bayes ───────────────────────────────────────────────────────────────
class GaussianNaiveBayes(BaseEstimator):
    """Gaussian Naive Bayes for continuous features (pure Python)."""

    def __init__(self):
        self._priors:   dict = {}
        self._means:    dict = {}
        self._vars:     dict = {}
        self._classes:  list = []

    def get_params(self) -> dict:
        return {}

    def fit(self, X: list, y: list) -> "GaussianNaiveBayes":
        from collections import defaultdict
        class_data: dict[str, list] = defaultdict(list)
        for xi, yi in zip(X, y):
            class_data[yi].append(xi)
        n = len(y)
        for cls, rows in class_data.items():
            self._priors[cls] = len(rows) / n
            cols = list(zip(*rows))
            self._means[cls] = [sum(c) / len(c) for c in cols]
            self._vars[cls]  = [sum((v - self._means[cls][i]) ** 2 for v in c) / len(c) + 1e-9
                                 for i, c in enumerate(cols)]
        self._classes = list(class_data.keys())
        self._is_fitted = True
        return self

    def _log_likelihood(self, x: list, cls: str) -> float:
        ll = math.log(self._priors[cls])
        for xi, mu, var in zip(x, self._means[cls], self._vars[cls]):
            ll -= 0.5 * (math.log(2 * math.pi * var) + (xi - mu) ** 2 / var)
        return ll

    def predict(self, X: list) -> list:
        self._check_fitted()
        return [max(self._classes, key=lambda c: self._log_likelihood(x, c)) for x in X]


# ── Pipeline (composition) ────────────────────────────────────────────────────
class Pipeline:
    """Chain of (name, estimator) steps — sklearn-inspired."""

    def __init__(self, steps: list[tuple[str, BaseEstimator]]):
        self.steps = steps

    def __repr__(self) -> str:
        names = " → ".join(name for name, _ in self.steps)
        return f"Pipeline([{names}])"

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def final_estimator(self) -> BaseEstimator:
        return self.steps[-1][1]

    def fit(self, X, y):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y) if hasattr(step, "fit_transform") else Xt
        self.final_estimator.fit(Xt, y)
        return self

    def predict(self, X):
        return self.final_estimator.predict(X)

    def score(self, X, y):
        return self.final_estimator.score(X, y)


# ── Demo ──────────────────────────────────────────────────────────────────────
def demo():
    iris_path = download_iris()
    X, y = load_iris(iris_path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    print("=== OOP: ML Estimator Hierarchy ===")
    print(f"  Train: {len(X_tr)} samples  |  Test: {len(X_te)} samples\n")

    models: list[BaseEstimator] = [
        KNNClassifier(k=3),
        KNNClassifier(k=7),
        GaussianNaiveBayes(),
    ]

    for model in models:
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        print(f"  {repr(model):<30}  test_acc={acc:.4f}")

    # Dunder __call__
    knn = KNNClassifier(k=5).fit(X_tr, y_tr)
    sample_preds = knn(X_te[:3])          # __call__ delegates to predict
    print(f"\n  knn(X_te[:3]) via __call__: {sample_preds}")
    print(f"  True labels              : {y_te[:3]}")

    # Pipeline
    pipeline = Pipeline([("knn", KNNClassifier(k=5))])
    pipeline.fit(X_tr, y_tr)
    print(f"\n  {pipeline}  len={len(pipeline)}  score={pipeline.score(X_te, y_te):.4f}")

    # Serialization via SerializableMixin
    save_path = DATA / "knn_model.pkl"
    knn.save(save_path)
    loaded = KNNClassifier.load(save_path)
    print(f"\n  Saved and reloaded — score: {loaded.score(X_te, y_te):.4f}")


if __name__ == "__main__":
    demo()
