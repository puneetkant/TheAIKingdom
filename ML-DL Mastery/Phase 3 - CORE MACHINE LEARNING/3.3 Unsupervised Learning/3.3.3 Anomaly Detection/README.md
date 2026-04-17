# 3.3.3 Anomaly Detection

Isolation Forest, Local Outlier Factor (LOF), One-Class SVM for unsupervised anomaly detection.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Statistical (z-score/IQR) outlier detection |
| `working_example2.py` | Isolation Forest, LOF, One-Class SVM with precision/recall |
| `working_example.ipynb` | Interactive: IsoForest contamination → LOF k → OC-SVM |

## Quick Reference

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest (fast, scalable)
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)   # -1 = anomaly, 1 = normal
anomaly_scores = iso.decision_function(X)  # lower = more anomalous

# LOF (local density comparison)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X)
```

## ML Connections
- **Fraud detection** — IsoForest on transaction features
- **Network intrusion** — LOF on packet statistics
- **Manufacturing QC** — One-Class SVM trained on good units only

## Learning Resources
- [sklearn Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [IsolationForest paper (Liu et al. 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

Explore this topic with a small practical project or coding exercise.

## What to build

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
