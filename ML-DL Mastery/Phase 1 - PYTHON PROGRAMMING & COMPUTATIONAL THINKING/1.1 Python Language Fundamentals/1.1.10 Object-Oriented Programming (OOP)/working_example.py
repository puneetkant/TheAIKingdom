"""
Working Example: Object-Oriented Programming (OOP)
Covers classes, __init__, methods, properties, class/static methods,
inheritance, MRO, dunder methods, dataclasses, and ABC.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math


# ── Basic class ───────────────────────────────────────────────────────────────
class Circle:
    """Represents a circle — demonstrates properties and dunder methods."""
    _instances_created = 0      # class variable

    def __init__(self, radius):
        if radius < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = radius   # name-mangled via single underscore → "private by convention"
        Circle._instances_created += 1

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be ≥ 0")
        self._radius = value

    @property
    def area(self):
        return math.pi * self._radius ** 2

    @property
    def circumference(self):
        return 2 * math.pi * self._radius

    # ── Class & static methods ────────────────────────────────────────────────
    @classmethod
    def from_diameter(cls, diameter):
        """Alternative constructor."""
        return cls(diameter / 2)

    @staticmethod
    def unit():
        """Returns a unit circle (radius=1)."""
        return Circle(1)

    @classmethod
    def instances_created(cls):
        return cls._instances_created

    # ── Dunder methods ────────────────────────────────────────────────────────
    def __repr__(self):
        return f"Circle(radius={self._radius})"

    def __str__(self):
        return f"Circle(r={self._radius:.2f}, area={self.area:.2f})"

    def __eq__(self, other):
        return isinstance(other, Circle) and math.isclose(self._radius, other._radius)

    def __lt__(self, other):
        return self._radius < other._radius

    def __add__(self, other):
        """Return a new circle whose area equals the sum of both areas."""
        combined_r = math.sqrt(self._radius**2 + other._radius**2)
        return Circle(combined_r)

    def __len__(self):
        """Perimeter rounded to int, just as a demo."""
        return round(self.circumference)


def demo_basic_class():
    print("=== Basic Class ===")
    c1 = Circle(5)
    c2 = Circle.from_diameter(14)
    c3 = Circle.unit()

    print(f"  repr: {c1!r}")
    print(f"  str : {c1}")
    print(f"  area: {c1.area:.4f}")
    print(f"  from_diameter(14): {c2}")
    print(f"  unit(): {c3}")
    print(f"  c1 == Circle(5): {c1 == Circle(5)}")
    print(f"  c1 < c2: {c1 < c2}")
    print(f"  c1 + c3 radius: {(c1 + c3).radius:.4f}")
    print(f"  len(c1) = {len(c1)}")
    print(f"  instances created: {Circle.instances_created()}")

    c1.radius = 10    # setter
    print(f"  after setter: {c1}")


# ── Inheritance ───────────────────────────────────────────────────────────────
class Animal:
    def __init__(self, name, sound):
        self.name  = name
        self.sound = sound

    def speak(self):
        return f"{self.name} says {self.sound}!"

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Woof")
        self.breed = breed

    def fetch(self, item):
        return f"{self.name} fetches the {item}!"

    def __repr__(self):
        return f"Dog(name={self.name!r}, breed={self.breed!r})"


class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Meow")
        self.indoor = indoor

    def purr(self):
        return f"{self.name} purrs..."


class ServiceDog(Dog):
    """Multiple roles: demonstrates MRO."""
    def __init__(self, name, breed, owner):
        super().__init__(name, breed)
        self.owner = owner

    def assist(self):
        return f"{self.name} assists {self.owner}."


def demo_inheritance():
    print("\n=== Inheritance & MRO ===")
    dog  = Dog("Rex", "Labrador")
    cat  = Cat("Whiskers")
    sd   = ServiceDog("Buddy", "Golden Retriever", "John")

    for a in [dog, cat, sd]:
        print(f"  {a!r}  → speak: {a.speak()}")

    print(f"  Dog.fetch: {dog.fetch('ball')}")
    print(f"  Cat.purr : {cat.purr()}")
    print(f"  ServiceDog.assist: {sd.assist()}")
    print(f"  MRO: {[c.__name__ for c in ServiceDog.__mro__]}")
    print(f"  isinstance(dog, Animal) = {isinstance(dog, Animal)}")
    print(f"  issubclass(Dog, Animal) = {issubclass(Dog, Animal)}")


# ── Abstract Base Class ───────────────────────────────────────────────────────
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

    def describe(self):
        return (f"{type(self).__name__}: "
                f"area={self.area():.2f}, perimeter={self.perimeter():.2f}")


class Rectangle(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h

    def area(self):      return self.w * self.h
    def perimeter(self): return 2 * (self.w + self.h)


class Triangle(Shape):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def area(self):
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s-self.a) * (s-self.b) * (s-self.c))

    def perimeter(self): return self.a + self.b + self.c


def demo_abc():
    print("\n=== Abstract Base Class ===")
    shapes = [Rectangle(4, 6), Triangle(3, 4, 5), Circle(7)]
    for s in shapes:
        # Circle also qualifies (has area property, but not Shape ABC here)
        try:
            print(f"  {s.describe()}")
        except AttributeError:
            print(f"  {s}: area={s.area:.2f}")   # Circle uses property


# ── Dataclass ────────────────────────────────────────────────────────────────
@dataclass(order=True)
class Point:
    x: float
    y: float
    label: str = field(default="", compare=False)

    def distance_to(self, other: "Point") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


def demo_dataclass():
    print("\n=== @dataclass ===")
    p1 = Point(0, 0, label="origin")
    p2 = Point(3, 4, label="A")
    p3 = Point(6, 8)
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  distance p1→p2 = {p1.distance_to(p2):.2f}")
    print(f"  p1 < p2 (by x then y) = {p1 < p2}")
    print(f"  sorted: {sorted([p3, p2, p1])}")


if __name__ == "__main__":
    demo_basic_class()
    demo_inheritance()
    demo_abc()
    demo_dataclass()
