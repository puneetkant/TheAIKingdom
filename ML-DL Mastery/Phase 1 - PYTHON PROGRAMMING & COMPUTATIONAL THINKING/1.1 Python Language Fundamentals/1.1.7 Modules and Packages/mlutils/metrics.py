"""Classification and regression metrics."""
def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

def f1(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 == b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if b == 1 != a)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 != b)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)
