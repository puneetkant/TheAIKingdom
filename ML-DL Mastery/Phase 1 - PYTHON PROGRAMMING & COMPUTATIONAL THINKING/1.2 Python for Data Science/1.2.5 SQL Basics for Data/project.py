"""Starter code for Phase 1 - PYTHON PROGRAMMING & COMPUTATIONAL THINKING\1.2 Python for Data Science\1.2.5 SQL Basics for Data.

Project: SQL Practice
"""

import sqlite3

def example():
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, value REAL)')
    c.execute('INSERT INTO items (name, value) VALUES (?, ?)', ('apple', 2.5))
    conn.commit()
    for row in c.execute('SELECT * FROM items'):
        print(row)
    conn.close()

if __name__ == '__main__':
    example()
