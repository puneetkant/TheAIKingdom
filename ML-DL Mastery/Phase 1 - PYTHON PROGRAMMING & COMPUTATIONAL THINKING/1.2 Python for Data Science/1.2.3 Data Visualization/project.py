"""Starter code for Phase 1 - PYTHON PROGRAMMING & COMPUTATIONAL THINKING\1.2 Python for Data Science\1.2.3 Data Visualization.

Project: Visualization Project
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def example():
    if plt is None:
        print('Install matplotlib to run this example.')
        return
    x = [1, 2, 3, 4]
    y = [1, 4, 9, 16]
    plt.plot(x, y, marker='o')
    plt.title('Sample Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    example()
