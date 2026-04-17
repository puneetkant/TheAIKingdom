"""Starter code for Phase 1 - PYTHON PROGRAMMING & COMPUTATIONAL THINKING\1.1 Python Language Fundamentals\1.1.10 Object-Oriented Programming (OOP).

Project: OOP Practice
"""

class ExampleClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f'Hello, {self.name}!')


def example():
    obj = ExampleClass('Learner')
    obj.greet()

if __name__ == '__main__':
    example()
