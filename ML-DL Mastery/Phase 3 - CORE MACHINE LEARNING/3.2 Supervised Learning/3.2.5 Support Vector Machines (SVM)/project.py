"""Starter code for Phase 3 - CORE MACHINE LEARNING\3.2 Supervised Learning\3.2.5 Support Vector Machines (SVM).

Project: Learning Project
"""

try:
    import numpy as np
except ImportError:
    np = None

def example():
    if np is None:
        print('Install numpy to run this example.')
        return
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print('a + b =', a + b)

if __name__ == '__main__':
    example()
