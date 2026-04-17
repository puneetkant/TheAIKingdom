"""Starter code for Phase 1 - PYTHON PROGRAMMING & COMPUTATIONAL THINKING\1.2 Python for Data Science\1.2.2 Pandas.

Project: Pandas Data Project
"""

try:
    import pandas as pd
except ImportError:
    pd = None

def example():
    if pd is None:
        print('Install pandas to run this example.')
        return
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    print(df)

if __name__ == '__main__':
    example()
