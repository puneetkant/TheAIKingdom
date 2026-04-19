"""Data preprocessing utilities."""
def normalize(data):
    lo, hi = min(data), max(data)
    return [(x - lo) / (hi - lo + 1e-8) for x in data]

def standardize(data):
    mean = sum(data) / len(data)
    std  = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / (std + 1e-8) for x in data]
