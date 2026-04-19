"""mlutils - a demo ML utilities package."""
from .metrics import accuracy, f1
from .preprocessing import normalize, standardize
__all__ = ['accuracy', 'f1', 'normalize', 'standardize']
__version__ = '0.1.0'
