# The Complete ML/DL Mastery Checklist
### Every Topic. Every Sub-Topic. Nothing Missing.

> **Instructions:** Check off each item as you complete it. Progress sequentially through Phases 1–4, then choose specialization tracks in Phase 5. Phases overlap — start the next phase when you're ~70% through the current one.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 1: PYTHON PROGRAMMING & COMPUTATIONAL THINKING
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1.1 Python Language Fundamentals

#### 1.1.1 Setup & Environment

📚 **Best Resources to Learn:**
- CS50P (Harvard, David Malan) — cs50.harvard.edu/python — **THE single best intro Python course**
- Python.org official setup — python.org/downloads
- VS Code Python setup — code.visualstudio.com/docs/python/python-tutorial
- Google Colab getting started — colab.research.google.com
- Real Python "Installing Python" — realpython.com/installing-python

- [ ] Installing Python (3.10+)
- [ ] Setting up a code editor / IDE (VS Code, PyCharm)
- [ ] Using the Python REPL / interactive interpreter
- [ ] Installing and using Jupyter Notebook
- [ ] Using Google Colab
- [ ] Understanding virtual environments (venv, conda)
- [ ] Using pip for package management
- [ ] Understanding `requirements.txt` and `pyproject.toml`


🏋️ **Exercises:**
1. Install Python 3.12+, verify with `python --version` in terminal
2. Set up VS Code with the Python extension, run a "Hello World" script
3. Create a virtual environment with `venv`, install `requests` and `numpy`, freeze requirements
4. Open Google Colab, run 5 cells demonstrating different Python expressions
5. Create a `requirements.txt` and a `.gitignore` for a Python project

#### 1.1.2 Basic Syntax & Data Types

📚 **Best Resources to Learn:**
- CS50P Weeks 0–1 — cs50.harvard.edu/python
- Python for Everybody Ch.1–2 — py4e.com
- Real Python "Basic Data Types" — realpython.com/python-data-types
- W3Schools Python — w3schools.com/python
- Automate the Boring Stuff Ch.1 — automatetheboringstuff.com

- [ ] Variables and assignment
- [ ] Naming conventions (snake_case, PEP 8)
- [ ] Comments and docstrings
- [ ] Numeric types: `int`, `float`, `complex`
- [ ] Boolean type: `bool`
- [ ] String type: `str`
  - [ ] String concatenation, repetition
  - [ ] String slicing and indexing
  - [ ] String methods (`split`, `join`, `strip`, `replace`, `find`, `format`)
  - [ ] f-strings (formatted string literals)
  - [ ] Raw strings, escape characters
  - [ ] String encoding (UTF-8, ASCII)
- [ ] `None` type
- [ ] Type conversion / casting (`int()`, `float()`, `str()`, `bool()`)
- [ ] Dynamic typing vs static typing
- [ ] `type()` and `isinstance()`


🏋️ **Exercises:**
1. Write a temperature converter (Celsius ↔ Fahrenheit ↔ Kelvin) using f-strings
2. Parse a full name string into first, middle, last using string methods
3. Write a program that demonstrates every string method (15+)
4. Create a type conversion cheat sheet — test what happens with every cast combination
5. CS50P Problem Set 0 (all problems)

#### 1.1.3 Operators

📚 **Best Resources to Learn:**
- CS50P Week 1 — cs50.harvard.edu/python
- Real Python "Operators and Expressions" — realpython.com/python-operators-expressions

- [ ] Arithmetic operators (`+`, `-`, `*`, `/`, `//`, `%`, `**`)
- [ ] Comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- [ ] Logical operators (`and`, `or`, `not`)
- [ ] Bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)
- [ ] Assignment operators (`=`, `+=`, `-=`, `*=`, etc.)
- [ ] Identity operators (`is`, `is not`)
- [ ] Membership operators (`in`, `not in`)
- [ ] Operator precedence


🏋️ **Exercises:**
1. Build a basic calculator that handles all arithmetic operators with error handling
2. Write a leap year checker using logical operators
3. Demonstrate `==` vs `is` with 5 different examples (strings, ints, lists)
4. Solve 10 operator precedence puzzles — predict output before running

#### 1.1.4 Control Flow

📚 **Best Resources to Learn:**
- CS50P Weeks 1–2 — cs50.harvard.edu/python
- Automate the Boring Stuff Ch.2 — automatetheboringstuff.com/2e/chapter2
- Python for Everybody Ch.3–5 — py4e.com

- [ ] `if` / `elif` / `else` statements
- [ ] Ternary conditional expressions
- [ ] `for` loops
  - [ ] Iterating over sequences
  - [ ] `range()` function
  - [ ] `enumerate()`
  - [ ] `zip()`
- [ ] `while` loops
- [ ] `break`, `continue`, `pass`
- [ ] Nested loops
- [ ] `match` / `case` (structural pattern matching, Python 3.10+)


🏋️ **Exercises:**
1. FizzBuzz (classic: 1–100, Fizz for 3, Buzz for 5, FizzBuzz for both)
2. Number guessing game with hints (higher/lower), limited attempts
3. Print multiplication tables (1–12) using nested for loops
4. Collatz conjecture: take any number, if even divide by 2, if odd multiply by 3 and add 1, count steps to reach 1
5. Rewrite a chain of 5+ if/elif statements using match/case
6. CS50P Problem Set 1

#### 1.1.5 Data Structures

📚 **Best Resources to Learn:**
- CS50P Week 3 — cs50.harvard.edu/python
- Python for Everybody Ch.8–10 — py4e.com
- Real Python "Lists and Tuples" — realpython.com/python-lists-tuples
- Real Python "Dictionaries" — realpython.com/python-dicts

- [ ] **Lists**
  - [ ] Creating, indexing, slicing
  - [ ] List methods (`append`, `extend`, `insert`, `remove`, `pop`, `sort`, `reverse`)
  - [ ] List comprehensions
  - [ ] Nested lists / matrices
  - [ ] Shallow vs deep copy
- [ ] **Tuples**
  - [ ] Creating and unpacking
  - [ ] Immutability
  - [ ] Named tuples (`collections.namedtuple`)
- [ ] **Dictionaries**
  - [ ] Creating, accessing, modifying
  - [ ] Dictionary methods (`keys`, `values`, `items`, `get`, `setdefault`, `update`)
  - [ ] Dictionary comprehensions
  - [ ] `defaultdict`, `OrderedDict`, `Counter` from `collections`
- [ ] **Sets**
  - [ ] Creating, adding, removing
  - [ ] Set operations (union, intersection, difference, symmetric difference)
  - [ ] Set comprehensions
  - [ ] `frozenset`
- [ ] **Strings as sequences**
- [ ] **Choosing the right data structure**


🏋️ **Exercises:**
1. Implement a contact book using nested dictionaries — add, search, delete, display all
2. Remove duplicates from a list using 3 different methods (set, dict, loop)
3. Word frequency counter using `Counter` on a paragraph of text
4. Implement a matrix as list of lists — write add, transpose, multiply functions
5. Compare performance: list vs set for membership testing with `timeit`
6. CS50P Problem Set 2

🛠️ **Mini-Project:** Build a **Grocery List Manager** — add items with quantities, remove items, check if exists, display sorted, show totals by category. Use lists, dicts, and sets.

#### 1.1.6 Functions

📚 **Best Resources to Learn:**
- CS50P Weeks 3–4 — cs50.harvard.edu/python
- Automate the Boring Stuff Ch.3 — automatetheboringstuff.com/2e/chapter3
- Real Python "Defining Functions" — realpython.com/defining-your-own-python-function
- Corey Schafer Functions playlist — youtube.com/@coreyms

- [ ] Defining functions with `def`
- [ ] Parameters and arguments
  - [ ] Positional arguments
  - [ ] Keyword arguments
  - [ ] Default parameter values
  - [ ] `*args` (variable positional arguments)
  - [ ] `**kwargs` (variable keyword arguments)
  - [ ] Keyword-only arguments
  - [ ] Positional-only arguments (Python 3.8+)
- [ ] Return values (single, multiple via tuple)
- [ ] Docstrings and function documentation
- [ ] Scope: local, enclosing, global, built-in (LEGB rule)
- [ ] `global` and `nonlocal` keywords
- [ ] Lambda (anonymous) functions
- [ ] Higher-order functions (`map`, `filter`, `reduce`)
- [ ] Closures
- [ ] Recursion
  - [ ] Base case and recursive case
  - [ ] Recursion depth and stack overflow
  - [ ] Tail recursion
- [ ] Function annotations / type hints
- [ ] `functools` module (`partial`, `lru_cache`, `wraps`)


🏋️ **Exercises:**
1. Write recursive functions for: factorial, fibonacci, binary search, tower of Hanoi, flatten nested list
2. Create a higher-order function that applies any transformation to every element in a list
3. Implement a memoized fibonacci using `@lru_cache`, compare speed with naive recursion
4. Write 5 functions with full type hints, validate with `mypy`
5. Create a function using `*args` and `**kwargs` that works as a universal logger
6. CS50P Problem Sets 3–4

🛠️ **Mini-Project:** Build a **Text Statistics Analyzer** — functions that compute word count, sentence count, avg word length, most common words (top 10), reading level estimate, character frequency distribution.

#### 1.1.7 Modules and Packages

📚 **Best Resources to Learn:**
- CS50P Week 4 — cs50.harvard.edu/python
- Real Python "Python Modules and Packages" — realpython.com/python-modules-packages
- Python docs — docs.python.org/3/tutorial/modules.html

- [ ] Importing modules (`import`, `from ... import`, `as`)
- [ ] Creating your own modules
- [ ] `__name__` and `__main__`
- [ ] Packages and `__init__.py`
- [ ] Standard library overview (`os`, `sys`, `math`, `random`, `datetime`, `json`, `re`, `collections`, `itertools`)
- [ ] Installing third-party packages with pip
- [ ] Understanding package managers (pip, conda)


🏋️ **Exercises:**
1. Create a custom `utils` module with 5+ helper functions, import in another script
2. Explore `os`, `sys`, `math`, `random`, `datetime` — write a script using each
3. Build a package with `__init__.py`, 3 sub-modules, and cross-module imports
4. Understand `__name__ == '__main__'` — demonstrate with a module that works both as import and script

#### 1.1.8 File I/O

📚 **Best Resources to Learn:**
- CS50P Week 6 — cs50.harvard.edu/python
- Automate the Boring Stuff Ch.9 — automatetheboringstuff.com/2e/chapter9
- Real Python "Reading and Writing Files" — realpython.com/read-write-files-python

- [ ] Reading files (`open`, `read`, `readline`, `readlines`)
- [ ] Writing files (`write`, `writelines`)
- [ ] Context managers (`with` statement)
- [ ] File modes (`r`, `w`, `a`, `rb`, `wb`)
- [ ] Working with CSV files (`csv` module)
- [ ] Working with JSON files (`json` module)
- [ ] Working with paths (`os.path`, `pathlib`)
- [ ] Reading/writing binary files
- [ ] Working with YAML, TOML, XML


🏋️ **Exercises:**
1. Read a CSV file, compute column statistics, write results to a new CSV
2. Build a JSON config reader/writer for application settings
3. Process a log file: count errors/warnings, extract timestamps, write summary
4. Recursively list all files in a directory tree using `pathlib`, compute total size
5. CS50P Problem Set 6

🛠️ **Mini-Project:** Build a **Personal Diary CLI** — write daily entries to dated files, search past entries by keyword, display by date range. Uses `pathlib`, `datetime`, `json`.

#### 1.1.9 Error and Exception Handling

📚 **Best Resources to Learn:**
- CS50P Week 3 (exceptions section) — cs50.harvard.edu/python
- Real Python "Python Exceptions" — realpython.com/python-exceptions
- Python docs — docs.python.org/3/tutorial/errors.html

- [ ] `try` / `except` / `else` / `finally`
- [ ] Common built-in exceptions (`ValueError`, `TypeError`, `IndexError`, `KeyError`, `FileNotFoundError`, `ZeroDivisionError`)
- [ ] Raising exceptions (`raise`)
- [ ] Custom exception classes
- [ ] Exception chaining
- [ ] Assertions (`assert`)
- [ ] Debugging with `pdb`
- [ ] Logging (`logging` module)


🏋️ **Exercises:**
1. Write a robust input validator that catches `ValueError`, `TypeError`, `IndexError` with specific messages
2. Create 3 custom exception classes for a banking app (`InsufficientFunds`, `InvalidAccount`, `DailyLimitExceeded`)
3. Add `logging` to an existing script — use DEBUG, INFO, WARNING, ERROR levels
4. Debug a deliberately broken 50-line script using `pdb` step-by-step

#### 1.1.10 Object-Oriented Programming (OOP)

📚 **Best Resources to Learn:**
- CS50P Weeks 8–9 — cs50.harvard.edu/python
- Corey Schafer OOP playlist (6 videos) — youtube.com/@coreyms
- Real Python "OOP in Python 3" — realpython.com/python3-object-oriented-programming
- Python docs — docs.python.org/3/tutorial/classes.html

- [ ] Classes and objects
- [ ] `__init__` constructor
- [ ] Instance variables and methods
- [ ] Class variables and class methods (`@classmethod`)
- [ ] Static methods (`@staticmethod`)
- [ ] Properties (`@property`, getters, setters)
- [ ] Encapsulation (public, protected `_`, private `__`)
- [ ] Inheritance
  - [ ] Single inheritance
  - [ ] Multiple inheritance
  - [ ] Method Resolution Order (MRO)
  - [ ] `super()` function
- [ ] Polymorphism
- [ ] Abstraction (`abc` module, abstract base classes)
- [ ] Dunder / magic methods (`__str__`, `__repr__`, `__len__`, `__getitem__`, `__eq__`, `__lt__`, `__add__`, `__iter__`, `__next__`, `__enter__`, `__exit__`, `__call__`)
- [ ] Data classes (`@dataclass`, Python 3.7+)
- [ ] Slots (`__slots__`)


🏋️ **Exercises:**
1. Build `BankAccount` with deposit, withdraw, transfer, transaction history, overdraft protection
2. Create `Shape` → `Circle`, `Rectangle`, `Triangle` hierarchy with area/perimeter/display methods
3. Implement `Vector` class with `__add__`, `__sub__`, `__mul__`, `__repr__`, `__eq__`, `__abs__`
4. Build a deck of cards: `Card` class with comparison, `Deck` class with shuffle/deal/sort
5. Implement an iterator class that yields Fibonacci numbers
6. CS50P Problem Sets 8–9

🛠️ **PROJECT: Library Management System** — Book, Member, Librarian classes with inheritance; check-out/return with dates; search by title/author/genre; late fees; save/load to JSON; full exception handling.

#### 1.1.11 Advanced Python Concepts

📚 **Best Resources to Learn:**
- CS50P Week 9 — cs50.harvard.edu/python
- Real Python "Generators" — realpython.com/introduction-to-python-generators
- Real Python "Decorators" — realpython.com/primer-on-python-decorators
- Automate the Boring Stuff Ch.7 (Regex) — automatetheboringstuff.com/2e/chapter7
- David Beazley "Generators" talk — youtube.com (search "David Beazley generators")

- [ ] Iterators and the iterator protocol
- [ ] Generators (`yield`, generator expressions)
- [ ] Decorators (function decorators, class decorators)
- [ ] Context managers (custom, `contextlib`)
- [ ] Comprehensions (list, dict, set, generator)
- [ ] Walrus operator (`:=`, Python 3.8+)
- [ ] `*` and `**` unpacking operators
- [ ] Regular expressions (`re` module)
  - [ ] Pattern matching, groups, quantifiers
  - [ ] `search`, `match`, `findall`, `sub`
- [ ] Multithreading (`threading` module)
- [ ] Multiprocessing (`multiprocessing` module)
- [ ] Async programming (`asyncio`, `async`/`await`)
- [ ] Type hints and `typing` module
- [ ] `collections` module deep dive
- [ ] `itertools` module deep dive


🏋️ **Exercises:**
1. Write a generator that yields prime numbers infinitely
2. Build `@timer`, `@retry(n=3)`, and `@cache` decorators from scratch
3. Create a custom context manager for database-like connections (acquire/release pattern)
4. Parse emails, phone numbers, and URLs from raw text using regex
5. Compare `threading` vs `multiprocessing` for CPU-bound vs I/O-bound tasks with benchmarks

#### 1.1.12 Testing

📚 **Best Resources to Learn:**
- Real Python "Getting Started with Testing" — realpython.com/python-testing
- pytest docs — docs.pytest.org
- Corey Schafer "unittest" tutorial — youtube.com/@coreyms

- [ ] Unit testing with `unittest`
- [ ] Testing with `pytest`
- [ ] Test-driven development (TDD) basics
- [ ] Mocking (`unittest.mock`)


🏋️ **Exercises:**
1. Write 15 pytest tests for a previous project (Library Management or Text Analyzer)
2. Practice TDD: write tests first, then implement a Stack and Queue data structure
3. Use `unittest.mock` to mock an API call in a test

---

### 1.2 Python for Data Science

#### 1.2.1 NumPy

📚 **Best Resources to Learn:**
- NumPy Quickstart — numpy.org/doc/stable/user/quickstart.html
- Python Data Science Handbook Ch.2 (FREE) — jakevdp.github.io/PythonDataScienceHandbook
- Kaggle Learn NumPy section
- Keith Galli NumPy tutorial — youtube.com (search "Keith Galli NumPy complete")

- [ ] Creating arrays (`np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.random`)
- [ ] Array attributes (`shape`, `dtype`, `ndim`, `size`)
- [ ] Array indexing and slicing (1D, 2D, 3D)
- [ ] Boolean indexing / masking
- [ ] Fancy indexing
- [ ] Array reshaping (`reshape`, `ravel`, `flatten`, `transpose`, `.T`)
- [ ] Broadcasting rules
- [ ] Element-wise operations
- [ ] Universal functions (ufuncs)
- [ ] Aggregation functions (`sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax`, `cumsum`)
- [ ] Linear algebra operations (`np.dot`, `np.matmul`, `@`, `np.linalg`)
- [ ] Random number generation (`np.random`)
- [ ] Stacking and splitting arrays (`np.concatenate`, `np.vstack`, `np.hstack`, `np.split`)
- [ ] `np.where`, `np.select`
- [ ] Sorting arrays (`np.sort`, `np.argsort`)
- [ ] Memory layout (C-order, Fortran-order)
- [ ] Structured arrays
- [ ] Saving and loading arrays (`np.save`, `np.load`, `np.savetxt`)


🏋️ **Exercises:**
1. Create a 10×10 multiplication table as a NumPy array using broadcasting
2. Implement matrix multiplication from scratch, verify with `np.matmul`
3. Normalize a 100×50 random matrix to zero mean and unit variance per column
4. Solve 3 systems of linear equations using `np.linalg.solve`
5. Load an image as array, convert to grayscale, flip horizontally, rotate 90°, apply blur

#### 1.2.2 Pandas

📚 **Best Resources to Learn:**
- Python Data Science Handbook Ch.3 (FREE) — jakevdp.github.io/PythonDataScienceHandbook
- Kaggle "Pandas" micro-course (FREE) — kaggle.com/learn/pandas
- pandas "10 Minutes to Pandas" — pandas.pydata.org/docs/user_guide/10min.html
- Corey Schafer Pandas playlist — youtube.com/@coreyms

- [ ] **Series**
  - [ ] Creating Series
  - [ ] Indexing and selection
  - [ ] Series methods and attributes
- [ ] **DataFrames**
  - [ ] Creating DataFrames (from dicts, lists, CSV, Excel, SQL)
  - [ ] Viewing data (`head`, `tail`, `info`, `describe`, `shape`, `dtypes`)
  - [ ] Selecting columns (single, multiple)
  - [ ] Selecting rows (`loc`, `iloc`, boolean indexing)
  - [ ] Adding and removing columns
  - [ ] Adding and removing rows
- [ ] **Data Cleaning**
  - [ ] Handling missing values (`isnull`, `dropna`, `fillna`, `interpolate`)
  - [ ] Handling duplicates (`duplicated`, `drop_duplicates`)
  - [ ] Data type conversion (`astype`)
  - [ ] String methods (`.str` accessor)
  - [ ] Renaming columns
  - [ ] Replacing values (`replace`, `map`)
- [ ] **Data Transformation**
  - [ ] `apply`, `map`, `applymap`/`map` (element-wise)
  - [ ] Sorting (`sort_values`, `sort_index`)
  - [ ] Ranking (`rank`)
  - [ ] Binning and discretization (`cut`, `qcut`)
  - [ ] One-hot encoding (`get_dummies`)
  - [ ] Pivot tables (`pivot_table`)
  - [ ] Melting / unpivoting (`melt`)
  - [ ] Cross-tabulations (`crosstab`)
- [ ] **Grouping and Aggregation**
  - [ ] `groupby` operations
  - [ ] Aggregation functions (`agg`, `transform`)
  - [ ] Multiple aggregations
  - [ ] Filtering groups (`filter`)
- [ ] **Merging, Joining, Concatenating**
  - [ ] `merge` (inner, outer, left, right joins)
  - [ ] `join`
  - [ ] `concat`
- [ ] **Time Series in Pandas**
  - [ ] DateTime index
  - [ ] `to_datetime`
  - [ ] Resampling (`resample`)
  - [ ] Rolling windows (`rolling`)
  - [ ] Time zone handling
  - [ ] Period and PeriodIndex
- [ ] **Input/Output**
  - [ ] Reading/writing CSV, Excel, JSON, SQL, Parquet, Feather, HDF5
  - [ ] Reading from APIs and web scraping
- [ ] **Performance optimization**
  - [ ] Vectorized operations vs loops
  - [ ] Memory optimization (`category` dtype)
  - [ ] `eval` and `query` methods


🏋️ **Exercises:**
1. Complete Kaggle Pandas micro-course (all exercises)
2. Load a messy CSV: fix 5 types of data quality issues (missing, wrong types, duplicates, inconsistent formats, outliers)
3. GroupBy analysis: compute 10 aggregate metrics on sales data grouped by region/product/quarter
4. Merge 3 DataFrames (customers + orders + products), compute customer lifetime value
5. Time series: resample daily data to weekly/monthly, compute 7-day and 30-day rolling averages

🛠️ **Mini-Project:** Download a real dataset (NYC taxi trips, Airbnb listings, or Spotify tracks), perform complete EDA answering 10 specific business questions, publish as a Kaggle notebook.

#### 1.2.3 Data Visualization

📚 **Best Resources to Learn:**
- Python Data Science Handbook Ch.4 (FREE) — jakevdp.github.io/PythonDataScienceHandbook
- Kaggle "Data Visualization" micro-course (FREE) — kaggle.com/learn/data-visualization
- Matplotlib tutorials — matplotlib.org/stable/tutorials
- Seaborn tutorial — seaborn.pydata.org/tutorial.html
- Plotly documentation — plotly.com/python

- [ ] **Matplotlib**
  - [ ] Figure and Axes objects
  - [ ] Line plots, scatter plots, bar charts, histograms
  - [ ] Pie charts, box plots, violin plots
  - [ ] Subplots and grids
  - [ ] Customizing plots (titles, labels, legends, colors, styles)
  - [ ] Saving figures (`savefig`)
  - [ ] Log scale, twin axes
  - [ ] Annotations and text
  - [ ] 3D plots (`mpl_toolkits.mplot3d`)
- [ ] **Seaborn**
  - [ ] Distribution plots (`histplot`, `kdeplot`, `ecdfplot`)
  - [ ] Categorical plots (`barplot`, `boxplot`, `violinplot`, `swarmplot`, `stripplot`)
  - [ ] Relational plots (`scatterplot`, `lineplot`)
  - [ ] Matrix plots (`heatmap`, `clustermap`)
  - [ ] Regression plots (`regplot`, `lmplot`)
  - [ ] Pair plots and joint plots
  - [ ] FacetGrid for multi-panel figures
  - [ ] Styling and themes
- [ ] **Plotly** (interactive visualization)
  - [ ] Interactive line, scatter, bar charts
  - [ ] 3D plots
  - [ ] Maps and geospatial visualization
  - [ ] Dash for web dashboards (overview)


🏋️ **Exercises:**
1. Recreate 5 classic charts: histogram, scatter with regression, box plot, heatmap, pair plot
2. Create a 2×3 subplot figure showing 6 views of the same dataset
3. Build an interactive Plotly dashboard with 3 linked charts and dropdown filters
4. Complete Kaggle Data Visualization micro-course

#### 1.2.4 Exploratory Data Analysis (EDA)

📚 **Best Resources to Learn:**
- Kaggle top-voted EDA notebooks — kaggle.com/code (search "EDA", sort by votes)
- StatQuest EDA overview — youtube.com/@statquest
- ydata-profiling docs — docs.profiling.ydata.ai

- [ ] Understanding data distributions
- [ ] Identifying outliers (IQR, Z-score)
- [ ] Correlation analysis (Pearson, Spearman, Kendall)
- [ ] Feature distributions and skewness
- [ ] Missing value patterns
- [ ] Data profiling (using `ydata-profiling` / `pandas-profiling`)
- [ ] Automated EDA tools


🏋️ **Exercises:**
1. Run `ydata-profiling` on 3 datasets, write up key findings from each report
2. Write a custom `eda_report()` function that produces summary stats, distributions, correlations, and missing value analysis for any DataFrame

#### 1.2.5 SQL Basics for Data

📚 **Best Resources to Learn:**
- CS50 SQL (Harvard, FREE) — cs50.harvard.edu/sql
- SQLBolt (interactive, FREE) — sqlbolt.com
- Mode Analytics SQL Tutorial (FREE) — mode.com/sql-tutorial
- Kaggle "Intro to SQL" + "Advanced SQL" micro-courses (FREE)

- [ ] SELECT, FROM, WHERE, ORDER BY
- [ ] GROUP BY, HAVING
- [ ] JOINs (INNER, LEFT, RIGHT, FULL, CROSS)
- [ ] Subqueries
- [ ] Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
- [ ] Window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD)
- [ ] CREATE TABLE, INSERT, UPDATE, DELETE
- [ ] Using SQLite with Python (`sqlite3`)
- [ ] SQLAlchemy basics
- [ ] Connecting to PostgreSQL / MySQL


🏋️ **Exercises:**
1. Complete all 20 SQLBolt lessons
2. Complete Kaggle SQL micro-courses (Intro + Advanced)
3. Write 20 queries on the Chinook database (increasing complexity)
4. Build a Python script that queries SQLite and loads results into a Pandas DataFrame
5. Write 5 window function queries (ROW_NUMBER, RANK, LAG, LEAD, running totals)

#### 1.2.6 Version Control with Git

📚 **Best Resources to Learn:**
- Git official book (FREE) — git-scm.com/book
- GitHub Skills (interactive, FREE) — skills.github.com
- Atlassian Git tutorials — atlassian.com/git/tutorials
- freeCodeCamp Git course — youtube.com (search "freeCodeCamp Git for beginners")

- [ ] Git basics (`init`, `add`, `commit`, `status`, `log`, `diff`)
- [ ] Branching (`branch`, `checkout`, `switch`, `merge`)
- [ ] Remote repositories (`clone`, `push`, `pull`, `fetch`)
- [ ] GitHub / GitLab workflows
- [ ] Pull requests / merge requests
- [ ] `.gitignore`
- [ ] Resolving merge conflicts
- [ ] Git for data science projects (DVC overview)


🏋️ **Exercises:**
1. Init a repo, make 10 meaningful commits
2. Create a feature branch, make changes, merge back to main
3. Deliberately create and resolve a merge conflict
4. Push a project to GitHub with README, .gitignore, and a PR

#### 1.2.7 Command Line / Shell Basics

📚 **Best Resources to Learn:**
- MIT "The Missing Semester" — missing.csail.mit.edu
- Linux Journey (FREE) — linuxjourney.com
- Codecademy "Learn the Command Line" (free tier)

- [ ] Navigating the file system (`cd`, `ls`, `pwd`, `mkdir`, `rm`)
- [ ] File manipulation (`cp`, `mv`, `cat`, `head`, `tail`, `grep`)
- [ ] Piping and redirection
- [ ] Environment variables
- [ ] Shell scripting basics
- [ ] SSH basics


🏋️ **Exercises:**
1. Navigate a file system using only terminal commands for 30 minutes
2. Write a bash script that organizes files in a directory by extension
3. Use `grep`, `sed`, `awk` to extract data from a log file

#### 1.2.8 Web Scraping & APIs

📚 **Best Resources to Learn:**
- Automate the Boring Stuff Ch.12–13 — automatetheboringstuff.com
- Real Python "Web Scraping" — realpython.com/python-web-scraping-practical-introduction
- Requests docs — docs.python-requests.org

- [ ] HTTP basics (GET, POST, status codes)
- [ ] `requests` library
- [ ] REST API fundamentals
- [ ] JSON parsing
- [ ] BeautifulSoup for HTML parsing
- [ ] Selenium for dynamic pages (overview)
- [ ] Rate limiting and ethical scraping


🏋️ **Exercises:**
1. Fetch data from 3 public APIs (OpenWeather, PokeAPI, REST Countries), parse JSON responses
2. Scrape a Wikipedia table, convert to Pandas DataFrame, save as CSV
3. Build a price tracker: scrape a product page daily, alert on price drop

🛠️ **PHASE 1 CAPSTONE PROJECTS:**
1. **COVID/Public Health Data Dashboard** — fetch API data, clean with Pandas, visualize with Plotly, store in SQLite
2. **Automated File Organizer** — scan directory, organize by type/date, configurable via JSON, with logging
3. **Personal Expense Tracker** — CLI app, Pandas analysis, monthly charts, CSV/Excel export

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 2: MATHEMATICS FOUNDATIONS
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 2.1 Linear Algebra

#### 2.1.1 Vectors

📚 **Best Resources to Learn:**
- 3Blue1Brown "Essence of Linear Algebra" (16 eps) — youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab — **START HERE, the best visual intro ever made**
- MIT 18.06 (Gilbert Strang) — ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011
- Khan Academy Linear Algebra — khanacademy.org/math/linear-algebra
- Mathematics for Machine Learning Ch.2 (FREE PDF) — mml-book.github.io

- [ ] Scalars, vectors, matrices, tensors — definitions
- [ ] Vector notation and representation
- [ ] Vector addition and scalar multiplication
- [ ] Geometric interpretation of vectors
- [ ] Dot product (inner product)
  - [ ] Algebraic definition
  - [ ] Geometric interpretation (projection)
  - [ ] Cosine similarity
- [ ] Cross product (3D)
- [ ] Vector norms
  - [ ] L1 norm (Manhattan distance)
  - [ ] L2 norm (Euclidean distance)
  - [ ] Lp norm (general)
  - [ ] L∞ norm (max norm)
  - [ ] Frobenius norm (for matrices)
- [ ] Unit vectors and normalization
- [ ] Linear independence
- [ ] Span of a set of vectors
- [ ] Basis and dimension
- [ ] Orthogonality and orthonormality


🏋️ **Exercises:**
1. Implement all vector operations in NumPy: addition, dot product, cross product, all norms
2. Compute cosine similarity between TF-IDF vectors of 5 documents
3. Show 3 vectors are linearly independent by computing the determinant
4. After each 3Blue1Brown episode, code the transformation in Python

#### 2.1.2 Matrices

📚 **Best Resources to Learn:**
- 3Blue1Brown Eps 3–9
- MIT 18.06 Lectures 1–16
- Mathematics for Machine Learning Ch.2

- [ ] Matrix notation and representation
- [ ] Matrix addition and scalar multiplication
- [ ] Matrix-vector multiplication
- [ ] Matrix-matrix multiplication
  - [ ] Rules and properties
  - [ ] Computational complexity
- [ ] Special matrices
  - [ ] Identity matrix
  - [ ] Diagonal matrix
  - [ ] Symmetric matrix
  - [ ] Skew-symmetric matrix
  - [ ] Orthogonal matrix
  - [ ] Positive definite / semi-definite matrices
  - [ ] Sparse matrices
  - [ ] Triangular matrices (upper, lower)
  - [ ] Idempotent matrix
  - [ ] Nilpotent matrix
  - [ ] Toeplitz matrix
  - [ ] Block matrices
- [ ] Transpose, conjugate transpose
- [ ] Trace of a matrix
- [ ] Matrix inverse
  - [ ] Existence conditions
  - [ ] Computing the inverse
  - [ ] Pseudo-inverse (Moore-Penrose)
- [ ] Matrix rank
  - [ ] Column rank, row rank
  - [ ] Rank-nullity theorem


🏋️ **Exercises:**
1. Implement matrix multiplication from scratch (no NumPy), verify with `np.matmul`
2. For a 4×4 matrix: compute rank, determinant, inverse; verify A × A⁻¹ = I
3. Write functions to check if a matrix is: symmetric, orthogonal, positive definite
4. Implement the pseudo-inverse computation using SVD

#### 2.1.3 Systems of Linear Equations

📚 **Best Resources to Learn:**
- MIT 18.06 Lectures 2–4
- Khan Academy — Systems of equations unit

- [ ] Representing systems as matrices (Ax = b)
- [ ] Gaussian elimination
- [ ] Row echelon form, reduced row echelon form
- [ ] Existence and uniqueness of solutions
- [ ] LU decomposition
- [ ] Cholesky decomposition


🏋️ **Exercises:**
1. Solve a 3×3 system by hand with Gaussian elimination, verify with NumPy
2. Implement Gaussian elimination from scratch
3. Solve a linear regression problem using the normal equation

#### 2.1.4 Determinants

📚 **Best Resources to Learn:**
- 3Blue1Brown Ep.6 "The Determinant"
- MIT 18.06 Lectures 18–19

- [ ] Definition and computation
- [ ] Properties of determinants
- [ ] Cofactor expansion
- [ ] Geometric interpretation (volume scaling)
- [ ] Determinant and invertibility
- [ ] Cramer's rule


🏋️ **Exercises:**
1. Compute 2×2, 3×3, 4×4 determinants by hand using cofactor expansion
2. Verify geometric interpretation: show det = area/volume scaling factor

#### 2.1.5 Vector Spaces

📚 **Best Resources to Learn:**
- MIT 18.06 Lectures 5–10
- 3Blue1Brown Eps 2, 7, 13
- Mathematics for Machine Learning Ch.2

- [ ] Definition and axioms
- [ ] Subspaces
- [ ] Column space (range)
- [ ] Null space (kernel)
- [ ] Row space
- [ ] Left null space
- [ ] Basis and dimension
- [ ] Change of basis
- [ ] Linear maps / transformations
  - [ ] Injective, surjective, bijective
  - [ ] Matrix representation of linear maps
  - [ ] Kernel and image


🏋️ **Exercises:**
1. For a given matrix, compute column space, null space, row space, left null space
2. Verify rank-nullity theorem numerically on 5 different matrices
3. Implement change of basis transformation

#### 2.1.6 Eigenvalues and Eigenvectors

📚 **Best Resources to Learn:**
- 3Blue1Brown Ep.14 "Eigenvectors and Eigenvalues" — **essential viewing**
- MIT 18.06 Lectures 21–22
- Mathematics for Machine Learning Ch.4
- StatQuest "Eigenvalues & Eigenvectors" video

- [ ] Definition and geometric interpretation
- [ ] Characteristic polynomial
- [ ] Computing eigenvalues and eigenvectors
- [ ] Eigendecomposition (spectral decomposition)
- [ ] Diagonalization
- [ ] Properties of eigenvalues (trace = sum, determinant = product)
- [ ] Eigenvalues of symmetric matrices (real, orthogonal eigenvectors)
- [ ] Power iteration method
- [ ] Applications to ML:
  - [ ] PCA (Principal Component Analysis)
  - [ ] Google PageRank
  - [ ] Spectral clustering
  - [ ] Markov chains (stationary distribution)


🏋️ **Exercises:**
1. Compute eigenvalues/eigenvectors of a 3×3 matrix by hand, verify with `np.linalg.eig`
2. Implement PCA from scratch using eigendecomposition of the covariance matrix
3. Implement the power iteration algorithm to find the dominant eigenvalue
4. Simulate Google PageRank on a small 6-page web graph using eigenvalues

#### 2.1.7 Singular Value Decomposition (SVD)

📚 **Best Resources to Learn:**
- MIT 18.06 Lecture 29
- Steve Brunton SVD tutorials — youtube.com (search "Steve Brunton SVD")
- Mathematics for Machine Learning Ch.4

- [ ] Definition: A = UΣVᵀ
- [ ] Relationship to eigendecomposition
- [ ] Geometric interpretation
- [ ] Compact / truncated SVD
- [ ] Low-rank approximation (Eckart-Young theorem)
- [ ] Applications:
  - [ ] Dimensionality reduction
  - [ ] Image compression
  - [ ] Recommender systems
  - [ ] Latent Semantic Analysis (LSA)
  - [ ] Pseudo-inverse computation


🏋️ **Exercises:**
1. Compute SVD of a matrix, reconstruct at different ranks, plot approximation error
2. Compress an image using truncated SVD (ranks: 1, 5, 10, 50, 100, full)
3. Build a simple recommender using SVD on a user-item ratings matrix

🛠️ **PROJECT: Image Compression with SVD** — Load high-res image, decompose, reconstruct at various ranks, plot error vs rank, compare with JPEG compression.

#### 2.1.8 Matrix Decompositions (Additional)

📚 **Best Resources to Learn:**
- MIT 18.06 (QR, LU lectures)
- SciPy linalg docs — docs.scipy.org/doc/scipy/reference/linalg.html

- [ ] QR decomposition
- [ ] LU decomposition
- [ ] Cholesky decomposition
- [ ] Schur decomposition
- [ ] Polar decomposition
- [ ] Non-negative Matrix Factorization (NMF)


🏋️ **Exercises:**
1. Compute QR, LU, and Cholesky decompositions using SciPy, verify each
2. Implement QR decomposition via Gram-Schmidt from scratch

#### 2.1.9 Inner Product Spaces

📚 **Best Resources to Learn:**
- MIT 18.06 Lectures 15–17 (Orthogonality)

- [ ] Inner product definition and properties
- [ ] Orthogonal projection
- [ ] Gram-Schmidt orthogonalization
- [ ] Orthogonal complements
- [ ] Projection matrices


🏋️ **Exercises:**
1. Implement Gram-Schmidt orthogonalization from scratch
2. Compute the projection of a vector onto a subspace

#### 2.1.10 Numerical Linear Algebra

📚 **Best Resources to Learn:**
- Numerical Linear Algebra (Trefethen & Bau) — advanced reference
- SciPy sparse matrix docs
- NumPy linalg docs

- [ ] Floating-point arithmetic and precision
- [ ] Condition number
- [ ] Numerical stability
- [ ] Iterative methods (conjugate gradient, GMRES)
- [ ] Sparse matrix representations (CSR, CSC, COO)
- [ ] NumPy/SciPy linear algebra routines


🏋️ **Exercises:**
1. Demonstrate floating-point issues: show 0.1 + 0.2 ≠ 0.3, compute condition numbers
2. Compare dense vs sparse matrix operations on a large (10000×10000) sparse matrix

---

### 2.2 Calculus

#### 2.2.1 Single Variable Calculus

📚 **Best Resources to Learn:**
- 3Blue1Brown "Essence of Calculus" (12 eps) — youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr — **START HERE**
- MIT 18.01SC — ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010
- Khan Academy Calculus AB + BC — khanacademy.org/math/ap-calculus-ab
- Professor Leonard Calculus — youtube.com/@ProfessorLeonard

- [ ] Limits and continuity
- [ ] Definition of a derivative
- [ ] Derivative rules
  - [ ] Power rule
  - [ ] Product rule
  - [ ] Quotient rule
  - [ ] Chain rule
- [ ] Derivatives of common functions (polynomial, exponential, logarithmic, trigonometric)
- [ ] Higher-order derivatives
- [ ] Implicit differentiation
- [ ] L'Hôpital's rule
- [ ] Mean Value Theorem
- [ ] Taylor series and Taylor expansion
  - [ ] Maclaurin series
  - [ ] Taylor remainder / error
  - [ ] Taylor approximation of functions (e^x, log, sigmoid)
- [ ] Integration
  - [ ] Definite and indefinite integrals
  - [ ] Fundamental Theorem of Calculus
  - [ ] Integration techniques (substitution, integration by parts)
  - [ ] Numerical integration (trapezoidal rule, Simpson's rule)


🏋️ **Exercises:**
1. Derive and plot the sigmoid function and its first derivative
2. Compute Taylor expansions for e^x, sin(x), log(1+x) — plot approximation vs true for different orders
3. Implement numerical integration (trapezoidal + Simpson's rule) from scratch
4. Differentiate the MSE loss (1/n)Σ(y - wx - b)² w.r.t. w and b by hand, verify with SymPy

#### 2.2.2 Multivariable Calculus

📚 **Best Resources to Learn:**
- MIT 18.02SC — ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010
- Khan Academy Multivariable Calculus — khanacademy.org/math/multivariable-calculus
- 3Blue1Brown multivariable episodes

- [ ] Functions of multiple variables
- [ ] Partial derivatives
- [ ] The gradient vector (∇f)
  - [ ] Definition
  - [ ] Geometric interpretation (direction of steepest ascent)
  - [ ] Gradient in machine learning (direction of steepest descent for loss)
- [ ] Directional derivatives
- [ ] Higher-order partial derivatives
- [ ] The Hessian matrix
  - [ ] Definition
  - [ ] Second derivative test
  - [ ] Convexity determination
  - [ ] Role in optimization (Newton's method)
- [ ] The Jacobian matrix
  - [ ] Definition
  - [ ] Chain rule with Jacobians
  - [ ] Applications in neural networks (backpropagation)
- [ ] Multiple integrals (double, triple)
- [ ] Change of variables (substitution in multiple integrals)
- [ ] Vector calculus
  - [ ] Divergence
  - [ ] Curl
  - [ ] Line integrals, surface integrals (overview)
- [ ] Implicit Function Theorem
- [ ] Lagrange multipliers (constrained optimization)


🏋️ **Exercises:**
1. Compute gradient of f(x,y) = x²y + sin(xy) at 5 points, plot gradient field
2. Implement gradient descent from scratch on f(x,y) = x² + 10y², visualize path on contour plot
3. Compute Hessian of a 2D function, determine if it's convex at different points
4. Use Lagrange multipliers to maximize f(x,y) = xy subject to x² + y² = 1

#### 2.2.3 Matrix Calculus

📚 **Best Resources to Learn:**
- MIT 18-S096 Matrix Calculus for ML — ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023
- "The Matrix Cookbook" (FREE PDF) — matrixcookbook.com
- Mathematics for Machine Learning Ch.5

- [ ] Derivative of scalar with respect to vector
- [ ] Derivative of vector with respect to vector (Jacobian)
- [ ] Derivative of scalar with respect to matrix
- [ ] Common matrix derivatives
  - [ ] ∂(xᵀa)/∂x = a
  - [ ] ∂(xᵀAx)/∂x = (A + Aᵀ)x
  - [ ] ∂(Ax)/∂x = Aᵀ
  - [ ] ∂ trace(AB) / ∂A = Bᵀ
- [ ] Chain rule for matrix derivatives
- [ ] Derivatives in neural networks
  - [ ] Backpropagation as chain rule application
  - [ ] Computational graphs
  - [ ] Automatic differentiation (forward mode, reverse mode)


🏋️ **Exercises:**
1. Derive the gradient of MSE loss w.r.t. weight vector using matrix calculus
2. Derive backpropagation for a 2-layer neural network entirely by hand
3. Verify hand-derived gradients using PyTorch autograd (numerical comparison)
4. Implement reverse-mode automatic differentiation for scalar functions

#### 2.2.4 Optimization Theory

📚 **Best Resources to Learn:**
- Convex Optimization (Boyd & Vandenberghe, FREE) — web.stanford.edu/~boyd/cvxbook
- Sebastian Ruder "Gradient Descent Optimization" — ruder.io/optimizing-gradient-descent
- d2l.ai Ch.12 (Optimization)
- Andrew Ng ML Specialization (optimization sections)

- [ ] **Unconstrained Optimization**
  - [ ] Critical points (stationary points)
  - [ ] First-order necessary conditions (∇f = 0)
  - [ ] Second-order sufficient conditions (Hessian positive definite)
  - [ ] Local vs global minima/maxima
  - [ ] Saddle points
  - [ ] Convex functions
    - [ ] Definition and properties
    - [ ] Convex sets
    - [ ] Jensen's inequality
    - [ ] Global optimality of local minima for convex functions
  - [ ] Non-convex optimization (landscape of neural networks)
- [ ] **Gradient-Based Optimization**
  - [ ] Gradient descent (batch)
    - [ ] Learning rate
    - [ ] Convergence criteria
    - [ ] Step size selection
  - [ ] Stochastic Gradient Descent (SGD)
    - [ ] Mini-batch SGD
    - [ ] Stochastic approximation
    - [ ] Variance of gradient estimates
  - [ ] SGD with Momentum
    - [ ] Classical momentum
    - [ ] Nesterov accelerated gradient (NAG)
  - [ ] Adaptive learning rate methods
    - [ ] Adagrad
    - [ ] Adadelta
    - [ ] RMSprop
    - [ ] Adam (Adaptive Moment Estimation)
    - [ ] AdamW (weight-decoupled Adam)
    - [ ] NAdam
    - [ ] RAdam (Rectified Adam)
    - [ ] LAMB / LARS (layer-wise adaptive methods)
    - [ ] Adafactor
    - [ ] Lion optimizer
    - [ ] Sophia optimizer
    - [ ] Schedule-free optimizers
  - [ ] Learning rate schedules
    - [ ] Step decay
    - [ ] Exponential decay
    - [ ] Cosine annealing
    - [ ] Warm restarts (cosine annealing with warm restarts)
    - [ ] Linear warmup
    - [ ] One-cycle policy
    - [ ] Cyclical learning rates
    - [ ] Polynomial decay
    - [ ] Reduce on plateau
  - [ ] Gradient clipping (by value, by norm)
  - [ ] Gradient accumulation
- [ ] **Constrained Optimization**
  - [ ] Equality constraints
  - [ ] Inequality constraints
  - [ ] Lagrangian and Lagrange multipliers
  - [ ] KKT (Karush-Kuhn-Tucker) conditions
  - [ ] Duality (primal and dual problems)
  - [ ] Applications: SVM optimization, constrained neural networks
- [ ] **Second-Order Methods**
  - [ ] Newton's method
  - [ ] Quasi-Newton methods (BFGS, L-BFGS)
  - [ ] Natural gradient descent
  - [ ] Gauss-Newton method
- [ ] **Convex Optimization**
  - [ ] Linear programming
  - [ ] Quadratic programming
  - [ ] Semidefinite programming (overview)
  - [ ] Interior point methods (overview)


🏋️ **Exercises:**
1. Implement SGD, SGD+Momentum, Nesterov, Adam, AdamW from scratch in NumPy
2. Compare convergence: SGD vs Momentum vs Adam on Rosenbrock function — visualize 2D paths
3. Implement cosine annealing with warm restarts — plot the LR schedule over 100 epochs
4. Derive KKT conditions for the soft-margin SVM optimization problem
5. Implement a learning rate finder that sweeps LR from 1e-7 to 10, plots loss vs LR

🛠️ **PROJECT: Gradient Descent Visualizer** — Interactive matplotlib animation comparing 5+ optimizers on 2D loss surfaces. Adjustable LR, momentum, contour plots, convergence tracking.

---

### 2.3 Probability and Statistics

#### 2.3.1 Combinatorics

📚 **Best Resources to Learn:**
- Harvard STAT 110 Lectures 1–3 — youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo
- Khan Academy Counting, Permutations, Combinations

- [ ] Counting principles (multiplication, addition)
- [ ] Permutations
- [ ] Combinations
- [ ] Binomial coefficients
- [ ] Multinomial coefficients
- [ ] Stars and bars (overview)


🏋️ **Exercises:**
1. Compute permutations and combinations for 5 different scenarios by hand and with `math.comb`
2. Implement `nCr` and `nPr` from scratch

#### 2.3.2 Probability Fundamentals

📚 **Best Resources to Learn:**
- Harvard STAT 110 (Joe Blitzstein) — **THE gold-standard probability course** — youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo
- FREE textbook: probabilitybook.net
- Khan Academy Statistics & Probability
- StatQuest — youtube.com/@statquest

- [ ] Sample space, events, outcomes
- [ ] Axioms of probability (Kolmogorov)
- [ ] Conditional probability P(A|B)
- [ ] Independence of events
- [ ] Law of total probability
- [ ] Bayes' theorem
  - [ ] Prior, likelihood, posterior, evidence
  - [ ] Bayesian updating
  - [ ] Bayesian vs frequentist interpretation
- [ ] Joint probability, marginal probability
- [ ] Chain rule of probability


🏋️ **Exercises:**
1. Birthday problem: compute exact probability for N=23, simulate 10,000 trials
2. Monty Hall: prove analytically and simulate 100,000 games
3. Medical test: compute P(disease|positive test) with different base rates using Bayes' theorem
4. STAT 110 Strategic Practice problems #1–3

#### 2.3.3 Random Variables

📚 **Best Resources to Learn:**
- STAT 110 Lectures 6–20
- Khan Academy Random Variables unit
- StatQuest "Mean", "Variance", "Covariance" videos

- [ ] Discrete random variables
  - [ ] Probability mass function (PMF)
  - [ ] Cumulative distribution function (CDF)
- [ ] Continuous random variables
  - [ ] Probability density function (PDF)
  - [ ] CDF for continuous variables
- [ ] Mixed random variables
- [ ] Expected value (mean)
  - [ ] Linearity of expectation
  - [ ] Law of the unconscious statistician (LOTUS)
- [ ] Variance and standard deviation
  - [ ] Var(X) = E[X²] - (E[X])²
  - [ ] Properties of variance
- [ ] Covariance
  - [ ] Definition
  - [ ] Covariance matrix
- [ ] Correlation
  - [ ] Pearson correlation coefficient
  - [ ] Spearman rank correlation
  - [ ] Kendall's tau
  - [ ] Correlation vs causation
- [ ] Moments (first, second, third = skewness, fourth = kurtosis)
- [ ] Moment generating functions (MGF)
- [ ] Characteristic functions (overview)


🏋️ **Exercises:**
1. For a custom discrete RV: compute PMF, CDF, E[X], Var(X) by hand and simulation
2. Prove linearity of expectation with 3 examples
3. Build a covariance matrix for 5 random variables from simulated data

#### 2.3.4 Common Probability Distributions

📚 **Best Resources to Learn:**
- StatQuest — individual video for EACH distribution — **watch all**
- STAT 110 (distribution lectures)
- Seeing Theory (interactive) — seeing-theory.brown.edu
- SciPy stats documentation — docs.scipy.org/doc/scipy/reference/stats.html

- [ ] **Discrete Distributions**
  - [ ] Bernoulli distribution
  - [ ] Binomial distribution
  - [ ] Multinomial distribution
  - [ ] Geometric distribution
  - [ ] Negative binomial distribution
  - [ ] Poisson distribution
  - [ ] Hypergeometric distribution
  - [ ] Categorical distribution
  - [ ] Discrete uniform distribution
- [ ] **Continuous Distributions**
  - [ ] Uniform distribution
  - [ ] Normal (Gaussian) distribution
    - [ ] Standard normal, Z-scores
    - [ ] 68-95-99.7 rule
    - [ ] Multivariate normal distribution
  - [ ] Exponential distribution
  - [ ] Gamma distribution
  - [ ] Beta distribution
  - [ ] Chi-squared distribution
  - [ ] Student's t-distribution
  - [ ] F-distribution
  - [ ] Log-normal distribution
  - [ ] Cauchy distribution
  - [ ] Weibull distribution
  - [ ] Dirichlet distribution
  - [ ] Laplace distribution
  - [ ] Pareto distribution


🏋️ **Exercises:**
1. For each of the 20+ distributions: implement PDF/PMF from scratch, plot, generate samples, verify E[X] and Var(X)
2. Demonstrate CLT: sample from Exponential distribution, show sample mean → Normal as n grows
3. Fit distributions to real data using `scipy.stats`, compare with Q-Q plots
4. STAT 110 Strategic Practice #4–8

#### 2.3.5 Joint Distributions and Transformations

📚 **Best Resources to Learn:**
- STAT 110 Lectures 13–16
- Mathematics for Machine Learning Ch.6

- [ ] Joint distributions (discrete and continuous)
- [ ] Marginal distributions
- [ ] Conditional distributions
- [ ] Independence of random variables
- [ ] Functions of random variables
- [ ] Transformation of distributions
- [ ] Convolution of distributions (sum of independent RVs)
- [ ] Order statistics


🏋️ **Exercises:**
1. Compute marginal and conditional distributions from a joint PMF table
2. Show that uncorrelated ≠ independent with a counterexample
3. Derive the distribution of Z = X + Y for two independent normals

#### 2.3.6 Limit Theorems

📚 **Best Resources to Learn:**
- STAT 110 Lectures 28–32
- StatQuest "Central Limit Theorem" video

- [ ] Law of Large Numbers (weak and strong)
- [ ] Central Limit Theorem (CLT)
  - [ ] Statement and implications
  - [ ] Convergence rate
- [ ] Convergence in probability, in distribution, almost surely
- [ ] Chebyshev's inequality
- [ ] Markov's inequality
- [ ] Hoeffding's inequality
- [ ] Chernoff bound


🏋️ **Exercises:**
1. Demonstrate LLN: plot running average converging to true mean for different distributions
2. Demonstrate CLT: histogram of sample means for n=1,5,30,100 from a Uniform distribution
3. Apply Chebyshev's inequality: bound P(|X-μ| > kσ) and compare with actual probability

#### 2.3.7 Estimation Theory

📚 **Best Resources to Learn:**
- STAT 110 (estimation lectures)
- MIT 18.05 (estimation module) — ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022
- StatQuest "Maximum Likelihood" and "MLE vs MAP" videos

- [ ] Point estimation
  - [ ] Maximum Likelihood Estimation (MLE)
    - [ ] Likelihood function
    - [ ] Log-likelihood
    - [ ] Deriving MLE for common distributions
    - [ ] Properties: consistency, asymptotic normality, efficiency
  - [ ] Maximum A Posteriori (MAP) estimation
    - [ ] Relationship to MLE with prior
    - [ ] Relationship to regularization
  - [ ] Method of Moments
  - [ ] Properties of estimators
    - [ ] Bias
    - [ ] Variance
    - [ ] Mean Squared Error (MSE)
    - [ ] Consistency
    - [ ] Efficiency
    - [ ] Sufficiency
    - [ ] Cramér-Rao lower bound
    - [ ] Fisher information
- [ ] Interval estimation
  - [ ] Confidence intervals
  - [ ] Pivotal quantities
  - [ ] Bootstrap methods for confidence intervals


🏋️ **Exercises:**
1. Derive MLE for Bernoulli, Gaussian (both μ and σ²), Poisson, Exponential — all by hand
2. Implement MLE from scratch for Gaussian: estimate μ and σ from data
3. Compute bootstrap confidence intervals for the mean (10,000 bootstrap samples)
4. Show that MAP with Gaussian prior on weights = L2 regularization

#### 2.3.8 Hypothesis Testing

📚 **Best Resources to Learn:**
- StatQuest "Hypothesis Testing" playlist — **watch all 10+ videos**
- MIT 18.05 (hypothesis testing units)
- Khan Academy Significance Tests unit

- [ ] Null and alternative hypotheses
- [ ] Test statistics
- [ ] P-values
- [ ] Significance levels (α)
- [ ] Type I and Type II errors
- [ ] Statistical power
- [ ] One-sample tests
  - [ ] z-test
  - [ ] t-test
- [ ] Two-sample tests
  - [ ] Independent two-sample t-test
  - [ ] Paired t-test
  - [ ] Welch's t-test
- [ ] Chi-squared test (goodness of fit, independence)
- [ ] ANOVA (one-way, two-way)
- [ ] F-test
- [ ] Non-parametric tests
  - [ ] Mann-Whitney U test
  - [ ] Wilcoxon signed-rank test
  - [ ] Kruskal-Wallis test
  - [ ] Kolmogorov-Smirnov test
- [ ] Multiple testing correction
  - [ ] Bonferroni correction
  - [ ] False Discovery Rate (FDR)
  - [ ] Benjamini-Hochberg procedure
- [ ] A/B testing
  - [ ] Design and analysis
  - [ ] Sample size calculation
  - [ ] Sequential testing
  - [ ] Multi-armed bandits (overview)


🏋️ **Exercises:**
1. Design an A/B test: compute required sample size, simulate, compute p-value, interpret
2. Perform t-test, chi-squared test, and Mann-Whitney U on real datasets
3. Apply Bonferroni correction to 20 simultaneous hypothesis tests, compute corrected p-values
4. Implement a simple A/B test simulator: generate data, run tests, compute false positive rate

🛠️ **Mini-Project: A/B Test Simulator** — Generate fake user data for 20 experiments (some with real effects, some null), apply multiple testing correction, report false discovery rate.

#### 2.3.9 Bayesian Statistics

📚 **Best Resources to Learn:**
- Statistical Rethinking (Richard McElreath lectures) — youtube.com (search "Statistical Rethinking 2024")
- Bayesian Methods for Hackers (FREE) — camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
- STAT 110 Bayesian lectures

- [ ] Prior distributions
  - [ ] Informative vs non-informative priors
  - [ ] Conjugate priors
  - [ ] Jeffreys prior
- [ ] Posterior distributions
- [ ] Posterior predictive distribution
- [ ] Bayesian inference
- [ ] Markov Chain Monte Carlo (MCMC)
  - [ ] Metropolis-Hastings algorithm
  - [ ] Gibbs sampling
- [ ] Variational inference (overview)
- [ ] Empirical Bayes
- [ ] Bayesian model comparison


🏋️ **Exercises:**
1. Implement Bayesian updating for coin flip (Beta-Binomial conjugate) — animate the posterior evolving
2. Build a Metropolis-Hastings sampler from scratch for a 1D distribution
3. Use PyMC to fit Bayesian linear regression, compare posterior with frequentist estimates

#### 2.3.10 Information Theory

📚 **Best Resources to Learn:**
- Chris Olah "Visual Information Theory" — colah.github.io/posts/2015-09-Visual-Information — **beautiful visual explanation**
- StatQuest "Entropy" and "Cross-Entropy" videos
- Information Theory, Inference and Learning (MacKay, FREE) — inference.org.uk/mackay/itila

- [ ] Entropy (Shannon entropy)
  - [ ] Definition: H(X) = -Σ p(x) log p(x)
  - [ ] Properties of entropy
  - [ ] Maximum entropy principle
- [ ] Cross-entropy
  - [ ] Definition: H(p, q) = -Σ p(x) log q(x)
  - [ ] Cross-entropy as loss function
- [ ] KL divergence (Kullback-Leibler)
  - [ ] Definition: KL(p || q) = Σ p(x) log(p(x)/q(x))
  - [ ] Asymmetry
  - [ ] Relationship to cross-entropy
  - [ ] Applications in ML (VAEs, information gain)
- [ ] Mutual information: I(X; Y)
- [ ] Conditional entropy
- [ ] Joint entropy
- [ ] Jensen-Shannon divergence
- [ ] Data processing inequality
- [ ] Rate-distortion theory (overview)
- [ ] Minimum description length (MDL)


🏋️ **Exercises:**
1. Compute entropy of a fair vs biased coin (p=0.5 vs p=0.9), plot H(p) for p ∈ [0,1]
2. Compute KL divergence between two Gaussians (analytically and numerically)
3. Show cross-entropy loss ≥ entropy (Gibbs' inequality) with numerical examples
4. Compute mutual information between two features in a dataset

#### 2.3.11 Stochastic Processes (Fundamentals)

📚 **Best Resources to Learn:**
- STAT 110 Markov chain lectures
- Khan Academy — Markov chains intro

- [ ] Markov chains
  - [ ] Transition matrices
  - [ ] Stationary distributions
  - [ ] Ergodicity
  - [ ] Absorbing chains
- [ ] Random walks
- [ ] Poisson processes
- [ ] Brownian motion / Wiener process (overview)
- [ ] Martingales (overview)


🏋️ **Exercises:**
1. Simulate a random walk in 1D and 2D, plot 100 trajectories
2. Build a Markov chain text generator: learn transition probabilities from a corpus, generate text
3. Find stationary distribution of a 4-state Markov chain by solving πP = π

🛠️ **PHASE 2 CAPSTONE PROJECT: Monte Carlo Simulation Lab** — Simulate 10 classic probability problems (birthday, Monty Hall, coupon collector, gambler's ruin, random walks, etc.), compare with analytical solutions, create convergence visualizations.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 3: CORE MACHINE LEARNING
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 3.1 Machine Learning Fundamentals

#### 3.1.1 Core Concepts

📚 **Best Resources to Learn:**
- Andrew Ng ML Specialization (Coursera, audit free) — coursera.org/specializations/machine-learning-introduction — **START HERE**
- Google ML Crash Course (FREE) — developers.google.com/machine-learning/crash-course
- StatQuest ML playlist — youtube.com/@statquest
- ISLR textbook (FREE) — statlearning.com

- [ ] What is machine learning (formal definition)
- [ ] Types of ML: supervised, unsupervised, semi-supervised, self-supervised, reinforcement learning
- [ ] Training, validation, test sets
- [ ] Overfitting vs underfitting
- [ ] Bias-variance tradeoff
  - [ ] Bias-variance decomposition of MSE
  - [ ] High bias = underfitting
  - [ ] High variance = overfitting
- [ ] No Free Lunch theorem
- [ ] Occam's razor / model parsimony
- [ ] Curse of dimensionality
- [ ] Inductive bias
- [ ] Parametric vs non-parametric models
- [ ] Generative vs discriminative models
- [ ] Online learning vs batch learning
- [ ] Instance-based vs model-based learning


🏋️ **Exercises:**
1. Demonstrate overfitting/underfitting: fit polynomials of degree 1,3,10,20 to noisy sine data
2. Implement train/val/test split from scratch, show validation prevents overfitting
3. Plot learning curves (train error vs val error vs dataset size) for 3 models
4. Create a visual explanation of the bias-variance tradeoff with bullseye diagrams

#### 3.1.2 Learning Theory

📚 **Best Resources to Learn:**
- Caltech "Learning from Data" (FREE) — work.caltech.edu/telecourse — **most rigorous ML theory course**
- CS229 lecture notes (learning theory section)
- Understanding Machine Learning (Shalev-Shwartz & Ben-David, FREE) — cs.huji.ac.il/~shais/UnderstandingMachineLearning

- [ ] PAC learning (Probably Approximately Correct)
- [ ] VC dimension (Vapnik-Chervonenkis)
- [ ] Rademacher complexity (overview)
- [ ] Generalization bounds
- [ ] Structural Risk Minimization (SRM)
- [ ] Empirical Risk Minimization (ERM)
- [ ] Sample complexity
- [ ] Double descent phenomenon


🏋️ **Exercises:**
1. Compute VC dimension of linear classifiers in 2D (prove it's 3 via shattering examples)
2. Implement the double descent curve: train models of increasing complexity, plot train and test error

---

### 3.2 Supervised Learning

#### 3.2.1 Linear Regression

📚 **Best Resources to Learn:**
- Andrew Ng ML Specialization Course 1, Weeks 1–2
- StatQuest "Linear Regression" — 4 videos — **watch all**
- ISLR Ch.3 (FREE)
- CS229 Lecture Notes 1

- [ ] Simple linear regression (one variable)
- [ ] Multiple linear regression
- [ ] Normal equation (closed-form solution)
- [ ] Gradient descent for linear regression
- [ ] Assumptions of linear regression
  - [ ] Linearity
  - [ ] Independence of errors
  - [ ] Homoscedasticity
  - [ ] Normality of residuals
  - [ ] No multicollinearity
- [ ] R-squared and adjusted R-squared
- [ ] Residual analysis
- [ ] Polynomial regression
- [ ] Interaction terms
- [ ] Regularized regression
  - [ ] Ridge regression (L2 penalty)
  - [ ] Lasso regression (L1 penalty)
  - [ ] Elastic Net (L1 + L2)
  - [ ] Regularization path
- [ ] Bayesian linear regression


🏋️ **Exercises:**
1. Implement linear regression from scratch (gradient descent AND normal equation)
2. Train on California Housing: interpret coefficients, check assumptions
3. Compare Ridge, Lasso, Elastic Net on a dataset with multicollinearity
4. Plot Lasso regularization path (coefficients vs lambda)
5. Implement Bayesian linear regression, visualize uncertainty in predictions

🛠️ **PROJECT: Kaggle House Price Prediction** — Full EDA, feature engineering, compare linear/Ridge/Lasso/Elastic Net, residual analysis.

#### 3.2.2 Logistic Regression

📚 **Best Resources to Learn:**
- Andrew Ng ML Specialization Course 1, Week 3
- StatQuest "Logistic Regression" — 3 videos
- ISLR Ch.4

- [ ] Sigmoid / logistic function
- [ ] Binary logistic regression
- [ ] Decision boundary
- [ ] Log-odds (logit)
- [ ] Maximum likelihood estimation for logistic regression
- [ ] Cross-entropy loss (log loss)
- [ ] Gradient descent for logistic regression
- [ ] Multinomial logistic regression (softmax regression)
- [ ] Regularization for logistic regression


🏋️ **Exercises:**
1. Implement logistic regression from scratch with gradient descent on binary cross-entropy
2. Plot decision boundary on 2D data for different regularization strengths
3. Derive the gradient of cross-entropy loss by hand
4. Implement multinomial logistic regression (softmax) from scratch

#### 3.2.3 K-Nearest Neighbors (KNN)

📚 **Best Resources to Learn:**
- StatQuest "KNN" video
- ISLR Ch.2.2.3
- scikit-learn KNN docs

- [ ] Algorithm description
- [ ] Distance metrics
  - [ ] Euclidean distance
  - [ ] Manhattan distance
  - [ ] Minkowski distance
  - [ ] Cosine distance
  - [ ] Hamming distance
  - [ ] Mahalanobis distance
- [ ] Choosing K (odd vs even, elbow method)
- [ ] Weighted KNN
- [ ] KNN for classification and regression
- [ ] KD-Trees, Ball Trees for efficient search
- [ ] Curse of dimensionality with KNN
- [ ] Approximate nearest neighbors (ANN)


🏋️ **Exercises:**
1. Implement KNN from scratch (classification + regression)
2. Plot decision boundaries for K=1,3,5,15,50 — observe overfitting vs underfitting
3. Compare Euclidean, Manhattan, Cosine distances on same dataset
4. Implement a KD-Tree for efficient nearest neighbor search

#### 3.2.4 Naive Bayes

📚 **Best Resources to Learn:**
- StatQuest "Naive Bayes" — 2 videos
- scikit-learn Naive Bayes docs

- [ ] Bayes' theorem application
- [ ] Conditional independence assumption
- [ ] Gaussian Naive Bayes
- [ ] Multinomial Naive Bayes
- [ ] Bernoulli Naive Bayes
- [ ] Complement Naive Bayes
- [ ] Laplace smoothing
- [ ] Applications in text classification


🏋️ **Exercises:**
1. Implement Gaussian Naive Bayes from scratch
2. Build a spam email classifier using Multinomial NB
3. Compare NB vs Logistic Regression on 3 text classification datasets

#### 3.2.5 Support Vector Machines (SVM)

📚 **Best Resources to Learn:**
- StatQuest "SVM" — 3 videos
- ISLR Ch.9
- CS229 Lecture Notes 3
- Caltech Lectures 14–16

- [ ] Maximum margin classifier (hard margin)
- [ ] Support vectors
- [ ] Soft margin SVM (C parameter)
- [ ] Hinge loss
- [ ] The kernel trick
  - [ ] Linear kernel
  - [ ] Polynomial kernel
  - [ ] Radial Basis Function (RBF / Gaussian) kernel
  - [ ] Sigmoid kernel
  - [ ] Custom kernels
- [ ] Kernel matrix (Gram matrix)
- [ ] SVM for multi-class classification (one-vs-one, one-vs-rest)
- [ ] SVM for regression (SVR)
- [ ] Sequential Minimal Optimization (SMO) algorithm
- [ ] Dual formulation


🏋️ **Exercises:**
1. Implement linear SVM from scratch using gradient descent on hinge loss
2. Visualize decision boundaries with linear, polynomial (degree 2,3,5), and RBF kernels
3. Tune C and gamma for RBF-SVM with grid search, plot accuracy heatmap
4. Show the kernel trick: map 2D data to higher dimensions, find linear separator

#### 3.2.6 Decision Trees

📚 **Best Resources to Learn:**
- StatQuest "Decision Trees" — 4 videos
- ISLR Ch.8
- scikit-learn Decision Tree docs

- [ ] Tree structure: root, nodes, leaves
- [ ] Splitting criteria
  - [ ] Information gain (entropy)
  - [ ] Gini impurity
  - [ ] Variance reduction (for regression)
- [ ] ID3 algorithm
- [ ] C4.5 / C5.0 algorithm
- [ ] CART (Classification and Regression Trees)
- [ ] Pruning
  - [ ] Pre-pruning (early stopping)
  - [ ] Post-pruning (cost-complexity pruning)
- [ ] Handling continuous features
- [ ] Handling missing values
- [ ] Feature importance from trees
- [ ] Advantages and limitations


🏋️ **Exercises:**
1. Implement a decision tree classifier from scratch using Gini impurity
2. Visualize a trained tree with `plot_tree` or graphviz
3. Compare Gini vs Entropy splitting criteria on 3 datasets
4. Demonstrate pruning: show overfitting with deep tree, improvement with pruning

#### 3.2.7 Ensemble Methods

📚 **Best Resources to Learn:**
- StatQuest "Random Forests", "AdaBoost", "Gradient Boost", "XGBoost" — **all essential**
- ISLR Ch.8 (Bagging, RF, Boosting)
- 💰 Hands-On ML Ch.7
- XGBoost docs — xgboost.readthedocs.io
- LightGBM docs — lightgbm.readthedocs.io
- CatBoost docs — catboost.ai

- [ ] **Bagging (Bootstrap Aggregating)**
  - [ ] Bootstrap sampling
  - [ ] Averaging predictions
  - [ ] Variance reduction
- [ ] **Random Forest**
  - [ ] Random feature subsets at each split
  - [ ] Out-of-bag (OOB) error
  - [ ] Feature importance (MDI, permutation importance)
  - [ ] Hyperparameters (n_estimators, max_depth, max_features, min_samples_split)
- [ ] **Boosting**
  - [ ] AdaBoost (Adaptive Boosting)
    - [ ] Sample weighting
    - [ ] Weak learner combination
  - [ ] Gradient Boosting Machines (GBM)
    - [ ] Gradient boosting framework
    - [ ] Learning rate (shrinkage)
    - [ ] Subsampling (stochastic gradient boosting)
  - [ ] XGBoost
    - [ ] Regularization terms
    - [ ] Column subsampling
    - [ ] Handling sparse data
    - [ ] Approximate split finding
    - [ ] System design: cache-aware, out-of-core
  - [ ] LightGBM
    - [ ] Leaf-wise growth (vs level-wise)
    - [ ] Gradient-based one-side sampling (GOSS)
    - [ ] Exclusive feature bundling (EFB)
  - [ ] CatBoost
    - [ ] Ordered boosting
    - [ ] Handling categorical features natively
    - [ ] Symmetric trees
- [ ] **Stacking (Stacked Generalization)**
  - [ ] Base learners and meta-learner
  - [ ] Cross-validated stacking
- [ ] **Voting classifiers**
  - [ ] Hard voting, soft voting
- [ ] **Blending**


🏋️ **Exercises:**
1. Implement bagging from scratch with decision tree stumps
2. Train Random Forest, XGBoost, LightGBM, CatBoost on same data — compare
3. Implement a 2-level stacking ensemble from scratch
4. Tune XGBoost with Optuna (Bayesian optimization), achieve best possible score
5. Analyze feature importance: compare MDI, permutation, SHAP across models

🛠️ **PROJECT: Kaggle Tabular Competition** — Enter a Playground Series competition, full pipeline: EDA → feature engineering → 5+ models → ensembling → submit.

#### 3.2.8 Perceptron and Linear Models

📚 **Best Resources to Learn:**
- ISLR Ch.4 (LDA/QDA)
- CS229 lecture notes

- [ ] Perceptron algorithm
- [ ] Perceptron convergence theorem
- [ ] Linear Discriminant Analysis (LDA)
- [ ] Quadratic Discriminant Analysis (QDA)


🏋️ **Exercises:**
1. Implement the perceptron algorithm from scratch, visualize convergence on linearly separable data
2. Compare LDA vs QDA vs Logistic Regression on same dataset

---

### 3.3 Unsupervised Learning

#### 3.3.1 Clustering

📚 **Best Resources to Learn:**
- Andrew Ng ML Specialization Course 3
- StatQuest "K-Means", "Hierarchical Clustering", "DBSCAN" videos
- ISLR Ch.12
- scikit-learn clustering docs

- [ ] **K-Means Clustering**
  - [ ] Algorithm: initialize, assign, update
  - [ ] K-Means++ initialization
  - [ ] Choosing K: elbow method, silhouette score, gap statistic
  - [ ] Limitations (spherical clusters, fixed K, outlier sensitivity)
  - [ ] Mini-batch K-Means
- [ ] **Hierarchical Clustering**
  - [ ] Agglomerative (bottom-up)
  - [ ] Divisive (top-down)
  - [ ] Linkage criteria (single, complete, average, Ward)
  - [ ] Dendrograms
  - [ ] Cutting the dendrogram
- [ ] **DBSCAN**
  - [ ] Core points, border points, noise points
  - [ ] Epsilon and MinPts parameters
  - [ ] Density-reachability, density-connectivity
  - [ ] Handling arbitrary cluster shapes
  - [ ] OPTICS algorithm (ordering points)
  - [ ] HDBSCAN
- [ ] **Gaussian Mixture Models (GMM)**
  - [ ] Mixture of Gaussians
  - [ ] Expectation-Maximization (EM) algorithm
    - [ ] E-step and M-step
    - [ ] Convergence properties
    - [ ] Initialization sensitivity
  - [ ] Soft clustering / probabilistic assignments
  - [ ] Selecting number of components (BIC, AIC)
- [ ] **Spectral Clustering**
  - [ ] Graph Laplacian
  - [ ] Normalized cuts
  - [ ] Connection to eigenvalue decomposition
- [ ] **Mean Shift**
- [ ] **Affinity Propagation**
- [ ] **BIRCH**
- [ ] **Cluster evaluation metrics**
  - [ ] Silhouette score
  - [ ] Calinski-Harabasz index
  - [ ] Davies-Bouldin index
  - [ ] Adjusted Rand Index (ARI)
  - [ ] Normalized Mutual Information (NMI)
  - [ ] V-measure


🏋️ **Exercises:**
1. Implement K-Means from scratch, visualize convergence step by step on 2D data
2. Implement the EM algorithm for GMM from scratch
3. Compare K-Means, DBSCAN, GMM on datasets with different cluster shapes (blobs, moons, circles)
4. Run hierarchical clustering, visualize dendrogram, cut at 3 different heights

🛠️ **PROJECT: Customer Segmentation** — Retail dataset, RFM analysis, K-Means/GMM segmentation, profile segments, visualize with UMAP.

#### 3.3.2 Dimensionality Reduction

📚 **Best Resources to Learn:**
- StatQuest "PCA" (2 videos), "t-SNE" video
- ISLR Ch.12
- UMAP docs — umap-learn.readthedocs.io
- 💰 Hands-On ML Ch.8

- [ ] **Principal Component Analysis (PCA)**
  - [ ] Derivation via variance maximization
  - [ ] Derivation via eigendecomposition of covariance matrix
  - [ ] Derivation via SVD
  - [ ] Explained variance ratio
  - [ ] Choosing number of components (scree plot, cumulative variance)
  - [ ] Kernel PCA
  - [ ] Incremental PCA
  - [ ] Sparse PCA
  - [ ] Robust PCA
- [ ] **t-SNE (t-distributed Stochastic Neighbor Embedding)**
  - [ ] Perplexity parameter
  - [ ] Non-convex optimization
  - [ ] Limitations (global structure, randomness)
  - [ ] Barnes-Hut approximation
- [ ] **UMAP (Uniform Manifold Approximation and Projection)**
  - [ ] Fuzzy simplicial sets
  - [ ] Preserving global and local structure
  - [ ] Comparison to t-SNE
- [ ] **Linear Discriminant Analysis (LDA) for dimensionality reduction**
  - [ ] Maximizing between-class / within-class variance ratio
- [ ] **Autoencoders (overview — detailed in Phase 4)**
- [ ] **Factor Analysis**
- [ ] **Independent Component Analysis (ICA)**
  - [ ] Cocktail party problem
  - [ ] Non-Gaussianity
- [ ] **Random projections (Johnson-Lindenstrauss lemma)**
- [ ] **Multidimensional Scaling (MDS)**
- [ ] **Isomap**
- [ ] **Locally Linear Embedding (LLE)**
- [ ] **Truncated SVD / LSA**
- [ ] **Feature selection vs feature extraction**


🏋️ **Exercises:**
1. Implement PCA from scratch (eigendecomposition of covariance matrix)
2. Apply PCA to MNIST: plot explained variance curve, visualize in 2D and 3D
3. Compare PCA vs t-SNE vs UMAP on MNIST — analyze preservation of global vs local structure
4. Implement ICA from scratch, apply to the cocktail party problem (audio unmixing)

#### 3.3.3 Anomaly Detection

📚 **Best Resources to Learn:**
- Andrew Ng ML Specialization (anomaly detection section)
- scikit-learn Anomaly Detection docs

- [ ] Statistical methods (Z-score, IQR)
- [ ] Isolation Forest
- [ ] One-class SVM
- [ ] Local Outlier Factor (LOF)
- [ ] Elliptic Envelope (Mahalanobis distance)
- [ ] Autoencoder-based anomaly detection
- [ ] DBSCAN as anomaly detector


🏋️ **Exercises:**
1. Compare Isolation Forest, One-class SVM, LOF on same dataset
2. Build autoencoder-based anomaly detector, set threshold at 95th percentile of reconstruction error

🛠️ **PROJECT: Credit Card Fraud Detection** — Kaggle dataset, handle severe imbalance, compare anomaly detection methods, evaluate with AUC-PR.

#### 3.3.4 Association Rule Learning

📚 **Best Resources to Learn:**
- mlxtend library docs — rasbt.github.io/mlxtend/user_guide/frequent_patterns

- [ ] Market basket analysis
- [ ] Support, confidence, lift
- [ ] Apriori algorithm
- [ ] FP-Growth algorithm


🏋️ **Exercises:**
1. Implement Apriori from scratch on a market basket dataset
2. Analyze a retail dataset: find rules with support > 0.01, confidence > 0.5

---

### 3.4 Data Preprocessing and Feature Engineering

#### 3.4.1 Data Cleaning

📚 **Best Resources to Learn:**
- Kaggle "Data Cleaning" micro-course (FREE)
- 💰 Hands-On ML Ch.2

- [ ] Handling missing values
  - [ ] Deletion (listwise, pairwise)
  - [ ] Imputation (mean, median, mode, KNN, MICE, iterative)
  - [ ] Indicator variables for missingness
- [ ] Handling outliers
  - [ ] Detection: IQR, Z-score, isolation forest
  - [ ] Treatment: capping, winsorization, removal, transformation
- [ ] Handling duplicates
- [ ] Data type correction
- [ ] Consistency checks


🏋️ **Exercises:**
1. Take a genuinely messy dataset, apply all imputation strategies, compare model performance
2. Compare listwise deletion vs mean vs KNN vs MICE imputation on same data with missing values

#### 3.4.2 Feature Scaling / Normalization

📚 **Best Resources to Learn:**
- scikit-learn preprocessing docs
- StatQuest "Feature Scaling" video

- [ ] Min-Max scaling (normalization to [0, 1])
- [ ] Standardization (Z-score normalization)
- [ ] Robust scaling (using median and IQR)
- [ ] Max-Abs scaling
- [ ] Log transformation
- [ ] Box-Cox transformation
- [ ] Yeo-Johnson transformation
- [ ] Power transforms
- [ ] When to scale (and when not to)


🏋️ **Exercises:**
1. Apply MinMax, StandardScaler, RobustScaler to same dataset — compare KNN performance
2. Apply Box-Cox and Yeo-Johnson transforms to skewed features, visualize before/after distributions

#### 3.4.3 Encoding Categorical Variables

📚 **Best Resources to Learn:**
- Kaggle "Feature Engineering" micro-course (FREE)
- category_encoders library docs

- [ ] Label encoding
- [ ] One-hot encoding
- [ ] Ordinal encoding
- [ ] Binary encoding
- [ ] Target encoding (mean encoding)
- [ ] Frequency encoding
- [ ] Hash encoding
- [ ] Embedding layers (for high-cardinality — deep learning)


🏋️ **Exercises:**
1. Encode a dataset with 5 categorical columns using all encoding methods, compare model performance
2. Implement target encoding from scratch with proper cross-validation to avoid leakage

#### 3.4.4 Feature Engineering

📚 **Best Resources to Learn:**
- Kaggle "Feature Engineering" micro-course (FREE)
- 💰 Feature Engineering and Selection (Kuhn, FREE online) — bookdown.org/max/FES
- Winning Kaggle solution notebooks

- [ ] Polynomial features
- [ ] Interaction features
- [ ] Binning / discretization
- [ ] Log, square root, reciprocal transforms
- [ ] Date/time feature extraction (year, month, day, hour, day of week, is_weekend)
- [ ] Text feature engineering (TF-IDF, word counts, n-grams)
- [ ] Aggregation features (mean, max, min, count per group)
- [ ] Lag features (for time series)
- [ ] Rolling statistics
- [ ] Domain-specific feature engineering


🏋️ **Exercises:**
1. Engineer 15+ new features for Kaggle House Prices dataset from existing columns
2. Create lag/rolling features for a time series dataset
3. Extract 10 features from a datetime column (year, month, day of week, is_weekend, quarter, etc.)

#### 3.4.5 Feature Selection

📚 **Best Resources to Learn:**
- scikit-learn feature selection docs
- StatQuest "Feature Selection" video

- [ ] Filter methods
  - [ ] Correlation-based
  - [ ] Chi-squared test
  - [ ] ANOVA F-test
  - [ ] Mutual information
  - [ ] Variance threshold
- [ ] Wrapper methods
  - [ ] Forward selection
  - [ ] Backward elimination
  - [ ] Recursive Feature Elimination (RFE)
- [ ] Embedded methods
  - [ ] L1 regularization (Lasso)
  - [ ] Tree-based feature importance
  - [ ] Permutation importance
- [ ] Dimensionality reduction (PCA, etc.)


🏋️ **Exercises:**
1. Compare filter (mutual information), wrapper (RFE), and embedded (Lasso) feature selection
2. Show that Lasso naturally performs feature selection — plot coefficients going to zero

#### 3.4.6 Handling Imbalanced Data

📚 **Best Resources to Learn:**
- imbalanced-learn library docs — imbalanced-learn.org
- Google ML Crash Course (data section)

- [ ] Oversampling
  - [ ] Random oversampling
  - [ ] SMOTE (Synthetic Minority Over-sampling Technique)
  - [ ] ADASYN
  - [ ] Borderline-SMOTE
- [ ] Undersampling
  - [ ] Random undersampling
  - [ ] Tomek links
  - [ ] Edited Nearest Neighbors
  - [ ] NearMiss
- [ ] Combined methods (SMOTE + Tomek)
- [ ] Class weights in algorithms
- [ ] Threshold tuning
- [ ] Anomaly detection approach
- [ ] Cost-sensitive learning


🏋️ **Exercises:**
1. Apply SMOTE, ADASYN, Tomek links, random oversampling to same imbalanced dataset
2. Compare SMOTE + model vs class_weight='balanced' vs threshold tuning — which works best?

---

### 3.5 Model Evaluation and Selection

#### 3.5.1 Classification Metrics

📚 **Best Resources to Learn:**
- StatQuest "ROC and AUC", "Confusion Matrix", "Sensitivity and Specificity" videos — **watch all**
- scikit-learn metrics docs
- Google ML Crash Course (classification section)

- [ ] Confusion matrix (TP, TN, FP, FN)
- [ ] Accuracy
- [ ] Precision (positive predictive value)
- [ ] Recall (sensitivity, true positive rate)
- [ ] F1 score (harmonic mean of precision and recall)
- [ ] F-beta score
- [ ] Specificity (true negative rate)
- [ ] ROC curve
- [ ] AUC-ROC (Area Under the ROC Curve)
- [ ] Precision-Recall curve
- [ ] AUC-PR (Area Under the Precision-Recall Curve)
- [ ] Log loss (cross-entropy loss)
- [ ] Cohen's Kappa
- [ ] Matthews Correlation Coefficient (MCC)
- [ ] Balanced accuracy
- [ ] Top-K accuracy
- [ ] Multi-class metrics (macro, micro, weighted averaging)


🏋️ **Exercises:**
1. For a binary classifier: compute ALL metrics (accuracy, precision, recall, F1, MCC, etc.) from confusion matrix
2. Plot ROC curve and PR curve, compute AUC for both
3. Show why accuracy is misleading for imbalanced data — use AUC-PR instead

#### 3.5.2 Regression Metrics

📚 **Best Resources to Learn:**
- StatQuest "R-squared" video
- scikit-learn regression metrics docs

- [ ] Mean Absolute Error (MAE)
- [ ] Mean Squared Error (MSE)
- [ ] Root Mean Squared Error (RMSE)
- [ ] R-squared (coefficient of determination)
- [ ] Adjusted R-squared
- [ ] Mean Absolute Percentage Error (MAPE)
- [ ] Symmetric MAPE (sMAPE)
- [ ] Median Absolute Error
- [ ] Mean Squared Logarithmic Error (MSLE)
- [ ] Explained variance score
- [ ] Huber loss
- [ ] Quantile loss


🏋️ **Exercises:**
1. Compute MAE, MSE, RMSE, R², MAPE for a regression model, interpret each
2. Show when MAE is preferred over MSE (robustness to outliers)

#### 3.5.3 Ranking and Recommendation Metrics
- [ ] Precision@K
- [ ] Recall@K
- [ ] NDCG (Normalized Discounted Cumulative Gain)
- [ ] MAP (Mean Average Precision)
- [ ] MRR (Mean Reciprocal Rank)
- [ ] Hit Rate

#### 3.5.4 Clustering Metrics
- [ ] Silhouette score
- [ ] Adjusted Rand Index
- [ ] Normalized Mutual Information
- [ ] Homogeneity, Completeness, V-measure
- [ ] Calinski-Harabasz, Davies-Bouldin

#### 3.5.5 Validation Strategies

📚 **Best Resources to Learn:**
- StatQuest "Cross Validation" video
- scikit-learn model selection docs

- [ ] Train-test split
- [ ] K-fold cross-validation
- [ ] Stratified K-fold
- [ ] Leave-One-Out Cross-Validation (LOOCV)
- [ ] Leave-P-Out
- [ ] Repeated K-fold
- [ ] Time series cross-validation (walk-forward, expanding window, sliding window)
- [ ] Nested cross-validation
- [ ] Group K-fold (for grouped data)
- [ ] Bootstrap validation


🏋️ **Exercises:**
1. Implement K-fold cross-validation from scratch
2. Compare simple holdout vs 5-fold vs 10-fold vs LOOCV — plot variance of estimates
3. Implement time series walk-forward validation from scratch

#### 3.5.6 Hyperparameter Tuning

📚 **Best Resources to Learn:**
- Optuna docs — optuna.org
- scikit-learn GridSearchCV, RandomizedSearchCV docs

- [ ] Grid search
- [ ] Random search
- [ ] Bayesian optimization (Optuna, Hyperopt, Ax)
- [ ] Successive halving (Hyperband)
- [ ] Early stopping
- [ ] Population-based training (PBT)
- [ ] Neural Architecture Search (NAS) — overview
- [ ] Cross-validated hyperparameter search


🏋️ **Exercises:**
1. Use grid search, random search, and Optuna (Bayesian) to tune XGBoost — compare efficiency
2. Implement early stopping within a cross-validation loop

#### 3.5.7 Model Interpretation and Explainability

📚 **Best Resources to Learn:**
- 📖 Interpretable ML (Molnar, FREE) — christophm.github.io/interpretable-ml-book
- SHAP docs — shap.readthedocs.io
- LIME docs — github.com/marcotcr/lime

- [ ] Feature importance
- [ ] Partial Dependence Plots (PDP)
- [ ] Individual Conditional Expectation (ICE) plots
- [ ] SHAP (SHapley Additive exPlanations)
  - [ ] Shapley values from cooperative game theory
  - [ ] TreeSHAP, DeepSHAP, KernelSHAP
- [ ] LIME (Local Interpretable Model-agnostic Explanations)
- [ ] Permutation importance
- [ ] Counterfactual explanations
- [ ] Attention visualization (for neural networks)
- [ ] Saliency maps
- [ ] Integrated gradients
- [ ] Global vs local interpretability


🏋️ **Exercises:**
1. Apply SHAP to a Random Forest: create summary plot, force plot, dependence plots
2. Apply LIME to explain 5 individual predictions from a black-box model
3. Create PDP and ICE plots for top 3 features
4. Compare feature importance rankings from: MDI, permutation, SHAP — are they consistent?

🛠️ **PHASE 3 CAPSTONE PROJECT: End-to-End ML Pipeline** — Choose a real-world problem (churn/fraud/disease). Complete pipeline: EDA → preprocessing → feature engineering → 5+ models → ensemble → tuning → SHAP interpretation → Flask/FastAPI deployment → technical write-up.

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 4: DEEP LEARNING FOUNDATIONS
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 4.1 Neural Network Fundamentals

#### 4.1.1 The Neuron / Perceptron

📚 **Best Resources to Learn:**
- 3Blue1Brown "Neural Networks" (4 eps) — youtube.com
- Andrew Ng DL Specialization Course 1, Weeks 1–2
- 📖 Neural Networks & DL (Nielsen, FREE) — neuralnetworksanddeeplearning.com Ch.1

- [ ] Biological inspiration
- [ ] Mathematical model: z = wᵀx + b, a = σ(z)
- [ ] Weights, biases, activation
- [ ] Single-layer perceptron
- [ ] Multi-layer perceptron (MLP) / feedforward network
- [ ] Universal approximation theorem


🏋️ **Exercises:**
1. Implement a single neuron (perceptron) from scratch, train on AND/OR/XOR gates
2. Show XOR fails for single-layer perceptron but works with 2 layers
3. Implement a 2-layer MLP from scratch in NumPy that solves XOR

#### 4.1.2 Activation Functions

📚 **Best Resources to Learn:**
- d2l.ai Ch.5 (activation functions section)
- PyTorch activation functions docs

- [ ] Sigmoid (logistic): σ(z) = 1/(1+e⁻ᶻ)
  - [ ] Vanishing gradient problem
- [ ] Tanh: tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
- [ ] ReLU: max(0, z)
  - [ ] Dying ReLU problem
- [ ] Leaky ReLU: max(αz, z)
- [ ] Parametric ReLU (PReLU)
- [ ] Exponential Linear Unit (ELU)
- [ ] Scaled Exponential Linear Unit (SELU)
- [ ] Gaussian Error Linear Unit (GELU)
- [ ] Swish / SiLU: x · σ(x)
- [ ] Mish: x · tanh(softplus(x))
- [ ] Softmax (for multi-class output)
- [ ] Softplus: log(1 + eˣ)
- [ ] Hard sigmoid, hard swish
- [ ] Choosing activation functions


🏋️ **Exercises:**
1. Implement and plot ALL activation functions AND their derivatives (15+ functions)
2. Show vanishing gradient: compare gradient magnitudes through 10 sigmoid layers vs 10 ReLU layers
3. Train identical networks with sigmoid, ReLU, GELU, Swish — compare convergence speed

#### 4.1.3 Loss Functions

📚 **Best Resources to Learn:**
- d2l.ai Ch.3–4 (loss functions sections)
- PyTorch loss functions docs

- [ ] **Regression losses**
  - [ ] Mean Squared Error (MSE) / L2 loss
  - [ ] Mean Absolute Error (MAE) / L1 loss
  - [ ] Huber loss (smooth L1)
  - [ ] Log-cosh loss
  - [ ] Quantile loss
- [ ] **Classification losses**
  - [ ] Binary cross-entropy (BCE)
  - [ ] Categorical cross-entropy
  - [ ] Sparse categorical cross-entropy
  - [ ] Focal loss (for class imbalance)
  - [ ] Hinge loss (SVM-style)
  - [ ] Squared hinge loss
  - [ ] Label smoothing
- [ ] **Contrastive and metric losses**
  - [ ] Contrastive loss
  - [ ] Triplet loss
  - [ ] InfoNCE loss
  - [ ] NT-Xent loss (SimCLR)
  - [ ] Circle loss
- [ ] **Other specialized losses**
  - [ ] CTC loss (for sequence tasks)
  - [ ] Dice loss (for segmentation)
  - [ ] IoU loss / GIoU loss (for detection)
  - [ ] Wasserstein loss (WGAN)
  - [ ] Reconstruction loss (for autoencoders)
  - [ ] KL divergence loss (for VAEs)
  - [ ] Perceptual loss / feature matching loss
  - [ ] Multi-task loss (weighted sum of losses)


🏋️ **Exercises:**
1. Implement MSE, BCE, focal loss, triplet loss from scratch in PyTorch
2. Show focal loss effect: plot loss curves for easy vs hard examples at different γ values
3. Train same model with MSE vs Huber vs MAE on data with outliers — compare robustness

#### 4.1.4 Forward Propagation

📚 **Best Resources to Learn:**
- 3Blue1Brown Neural Networks Ep.1
- Andrew Ng DL Specialization Course 1, Week 2

- [ ] Layer-by-layer computation
- [ ] Matrix formulation of forward pass
- [ ] Computational graphs


🏋️ **Exercises:**
1. Implement forward pass for a 3-layer MLP from scratch in NumPy (matrix formulation)
2. Draw and label a computational graph for a simple 2-layer network

#### 4.1.5 Backpropagation

📚 **Best Resources to Learn:**
- 3Blue1Brown "Backpropagation" (Eps 3–4)
- Andrej Karpathy "micrograd" — youtube.com (builds autograd from scratch in 2 hours)
- 📖 Nielsen Ch.2 — neuralnetworksanddeeplearning.com
- CS231n Backprop notes — cs231n.github.io/optimization-2

- [ ] Chain rule of calculus applied to computational graphs
- [ ] Gradient computation for each layer
- [ ] Backpropagation through time (BPTT) — for RNNs
- [ ] Gradient flow analysis
- [ ] Vanishing gradients
  - [ ] Causes (sigmoid/tanh saturation, deep networks)
  - [ ] Solutions (ReLU, skip connections, normalization, LSTM/GRU)
- [ ] Exploding gradients
  - [ ] Causes (large weight matrices)
  - [ ] Solutions (gradient clipping, weight initialization, normalization)
- [ ] Automatic differentiation
  - [ ] Forward mode AD
  - [ ] Reverse mode AD (what PyTorch/TensorFlow use)
  - [ ] Computational graph construction (static vs dynamic)


🏋️ **Exercises:**
1. Follow Karpathy's micrograd tutorial: build a complete autograd engine from scratch
2. Implement backprop for a 3-layer MLP from scratch in NumPy (no frameworks)
3. Verify gradients with numerical gradient checking (finite differences)
4. Train your from-scratch MLP on MNIST, achieve >95% accuracy

#### 4.1.6 Weight Initialization

📚 **Best Resources to Learn:**
- d2l.ai Ch.5 (initialization section)
- "Understanding Difficulty of Training Deep Feedforward NNs" (Glorot & Bengio paper)

- [ ] Zero initialization (why it fails)
- [ ] Random initialization (small random values)
- [ ] Xavier / Glorot initialization (for sigmoid/tanh)
- [ ] He / Kaiming initialization (for ReLU)
- [ ] LeCun initialization
- [ ] Orthogonal initialization
- [ ] Sparse initialization
- [ ] LSUV (Layer-Sequential Unit-Variance)
- [ ] Fixup initialization


🏋️ **Exercises:**
1. Train identical networks with zero, random, Xavier, He init — plot training curves
2. Show that zero init causes all neurons to learn identical features

---

### 4.2 Training Deep Neural Networks

#### 4.2.1 Regularization for Deep Learning

📚 **Best Resources to Learn:**
- Andrew Ng DL Specialization Course 2 (all of it)
- StatQuest "Regularization" videos
- d2l.ai Ch.4 (regularization)

- [ ] **L1 regularization (weight sparsity)**
- [ ] **L2 regularization (weight decay)**
- [ ] **Dropout**
  - [ ] Standard dropout
  - [ ] Inverted dropout
  - [ ] DropConnect
  - [ ] Spatial dropout (for CNNs)
  - [ ] DropBlock
  - [ ] Variational dropout
- [ ] **Batch Normalization**
  - [ ] Training vs inference behavior
  - [ ] Running mean and variance
  - [ ] Learnable parameters (γ, β)
  - [ ] Internal covariate shift (original motivation)
  - [ ] Benefits: faster training, regularization effect
- [ ] **Layer Normalization**
- [ ] **Instance Normalization**
- [ ] **Group Normalization**
- [ ] **RMSNorm (Root Mean Square Normalization)**
- [ ] **Weight normalization**
- [ ] **Spectral normalization**
- [ ] **Data augmentation** (detailed in CV section)
- [ ] **Early stopping**
  - [ ] Monitoring validation loss
  - [ ] Patience parameter
- [ ] **Noise injection**
  - [ ] Input noise
  - [ ] Weight noise
  - [ ] Label smoothing
- [ ] **Mixup** and **CutMix**
- [ ] **Stochastic depth**
- [ ] **Max-norm constraint**
- [ ] **Gradient penalty**


🏋️ **Exercises:**
1. Train with/without dropout, compare training vs validation curves
2. Implement Batch Normalization from scratch (forward + backward pass)
3. Compare BatchNorm vs LayerNorm vs GroupNorm on same model
4. Implement Mixup and CutMix data augmentation from scratch

#### 4.2.2 Optimizers (detailed)
- [ ] SGD (Stochastic Gradient Descent)
- [ ] SGD with momentum
- [ ] SGD with Nesterov momentum
- [ ] Adagrad
- [ ] Adadelta
- [ ] RMSprop
- [ ] Adam
- [ ] AdamW (decoupled weight decay)
- [ ] NAdam
- [ ] RAdam
- [ ] LAMB
- [ ] LARS
- [ ] Adafactor
- [ ] Lion
- [ ] Sophia
- [ ] Shampoo
- [ ] CAME
- [ ] Schedule-free optimizers
- [ ] Lookahead optimizer
- [ ] Gradient centralization
- [ ] SAM (Sharpness-Aware Minimization)

#### 4.2.3 Learning Rate Strategies
- [ ] Constant learning rate
- [ ] Step decay
- [ ] Exponential decay
- [ ] Polynomial decay
- [ ] Cosine annealing
- [ ] Cosine annealing with warm restarts (SGDR)
- [ ] Linear warmup + cosine decay
- [ ] One-cycle policy (super-convergence)
- [ ] Cyclical learning rates (CLR)
- [ ] Reduce on plateau (ReduceLROnPlateau)
- [ ] Learning rate range test
- [ ] Linear warmup + linear decay
- [ ] WSD (Warmup-Stable-Decay) schedule

#### 4.2.4 Training Practices
- [ ] Mini-batch training
- [ ] Epoch, batch, iteration terminology
- [ ] Data loading and batching (DataLoader, Dataset)
- [ ] Shuffling and sampling strategies
- [ ] Mixed precision training (FP16, BF16)
  - [ ] Loss scaling
  - [ ] Automatic mixed precision (AMP)
- [ ] Gradient accumulation (simulating larger batch sizes)
- [ ] Gradient checkpointing (memory-compute tradeoff)
- [ ] Distributed training
  - [ ] Data parallelism
  - [ ] Model parallelism
  - [ ] Pipeline parallelism
  - [ ] Tensor parallelism
  - [ ] ZeRO (Zero Redundancy Optimizer) stages 1, 2, 3
  - [ ] FSDP (Fully Sharded Data Parallel)
  - [ ] DeepSpeed library
  - [ ] Horovod (overview)
- [ ] Model checkpointing and resuming
- [ ] Reproducibility (random seeds, deterministic mode)
- [ ] Debugging neural networks
  - [ ] Gradient checking
  - [ ] Overfit a small batch
  - [ ] Learning curves analysis
  - [ ] Weight/gradient histograms
  - [ ] Dead neurons detection

---

### 4.3 Convolutional Neural Networks (CNNs)

#### 4.3.1 Convolution Operations

📚 **Best Resources to Learn:**
- CS231n ConvNet notes — cs231n.github.io/convolutional-networks — **gold standard**
- Andrew Ng DL Specialization Course 4 (CNNs)
- d2l.ai Ch.7

- [ ] 1D, 2D, 3D convolutions
- [ ] Kernel / filter
- [ ] Stride
- [ ] Padding (valid, same, full)
- [ ] Output size calculation
- [ ] Dilated / atrous convolutions
- [ ] Transposed convolutions (deconvolution)
- [ ] Depthwise separable convolutions
- [ ] Grouped convolutions
- [ ] 1×1 convolutions (pointwise)
- [ ] Deformable convolutions
- [ ] Receptive field
- [ ] Feature maps


🏋️ **Exercises:**
1. Implement 2D convolution from scratch in NumPy (forward pass)
2. Compute output sizes for 10 different stride/padding/dilation combinations
3. Visualize learned filters of first layer of a trained CNN

#### 4.3.2 Pooling Layers
- [ ] Max pooling
- [ ] Average pooling
- [ ] Global average pooling (GAP)
- [ ] Global max pooling
- [ ] Adaptive pooling
- [ ] Strided convolution as alternative to pooling

#### 4.3.3 CNN Architectures (Historical Evolution)

📚 **Best Resources to Learn:**
- CS231n "ConvNet Architectures" — cs231n.github.io
- Papers With Code — paperswithcode.com/area/computer-vision
- torchvision.models documentation

- [ ] **LeNet-5** (1998) — first practical CNN
- [ ] **AlexNet** (2012) — deep learning revolution
- [ ] **ZFNet** (2013) — visualization of learned features
- [ ] **VGGNet** (VGG-16, VGG-19) (2014) — depth with 3×3 kernels
- [ ] **GoogLeNet / Inception v1** (2014) — inception modules, multi-scale
  - [ ] Inception v2, v3, v4
  - [ ] Inception-ResNet
- [ ] **ResNet** (2015) — residual connections, skip connections
  - [ ] ResNet-18, 34, 50, 101, 152
  - [ ] Bottleneck blocks
  - [ ] Pre-activation ResNet (ResNet v2)
  - [ ] ResNeXt (grouped convolutions + residual)
  - [ ] Wide ResNet (WRN)
- [ ] **DenseNet** (2017) — dense connections
- [ ] **SqueezeNet** — efficient architecture
- [ ] **MobileNet** (v1, v2, v3) — mobile/edge deployment
  - [ ] Depthwise separable convolutions
  - [ ] Inverted residuals
  - [ ] Squeeze-and-excitation
- [ ] **ShuffleNet** (v1, v2)
- [ ] **NASNet** — Neural Architecture Search
- [ ] **EfficientNet** (B0–B7) — compound scaling
  - [ ] EfficientNetV2
- [ ] **ConvNeXt** — modernizing CNNs with Transformer ideas
  - [ ] ConvNeXt V2
- [ ] **RegNet** — designing network design spaces
- [ ] **SENet** (Squeeze-and-Excitation Networks) — channel attention


🏋️ **Exercises:**
1. Implement LeNet-5 from scratch in PyTorch, train on MNIST
2. Implement ResNet-18 with skip connections from scratch, train on CIFAR-10
3. Load pre-trained ResNet-50, use for transfer learning on custom dataset
4. Compare parameter counts and accuracy: VGG-16 vs ResNet-50 vs EfficientNet-B0

#### 4.3.4 Transfer Learning

📚 **Best Resources to Learn:**
- fast.ai Part 1 — course.fast.ai
- PyTorch Transfer Learning tutorial — pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

- [ ] Pre-trained models and fine-tuning
- [ ] Feature extraction (frozen backbone)
- [ ] Fine-tuning strategies
  - [ ] Full fine-tuning
  - [ ] Gradual unfreezing
  - [ ] Discriminative learning rates
- [ ] ImageNet pre-trained models
- [ ] Domain adaptation basics


🏋️ **Exercises:**
1. Fine-tune ResNet on a custom 5-class dataset: compare frozen backbone vs full fine-tuning
2. Implement gradual unfreezing: unfreeze one layer at a time, track accuracy

🛠️ **PROJECT: Custom Image Classifier** — Collect custom dataset (5–10 classes), transfer learning with EfficientNet, compare strategies, deploy as Gradio app.

---

### 4.4 Recurrent Neural Networks (RNNs)

#### 4.4.1 RNN Fundamentals

📚 **Best Resources to Learn:**
- Andrew Ng DL Specialization Course 5
- Colah "Understanding LSTMs" — colah.github.io/posts/2015-08-Understanding-LSTMs — **essential reading**
- d2l.ai Ch.9–10

- [ ] Vanilla RNN architecture
- [ ] Hidden state computation
- [ ] Unfolding through time
- [ ] Backpropagation Through Time (BPTT)
- [ ] Truncated BPTT
- [ ] Vanishing / exploding gradients in RNNs


🏋️ **Exercises:**
1. Implement vanilla RNN from scratch in PyTorch (using only `nn.Linear`)
2. Build LSTM for IMDB sentiment analysis
3. Compare RNN vs LSTM vs GRU on sequence prediction — demonstrate vanishing gradient

#### 4.4.2 Gated Architectures
- [ ] **Long Short-Term Memory (LSTM)**
  - [ ] Cell state
  - [ ] Forget gate
  - [ ] Input gate
  - [ ] Output gate
  - [ ] Peephole connections
- [ ] **Gated Recurrent Unit (GRU)**
  - [ ] Update gate
  - [ ] Reset gate
  - [ ] Comparison to LSTM
- [ ] Bidirectional RNNs
- [ ] Stacked / multi-layer RNNs
- [ ] Deep RNNs

#### 4.4.3 RNN Applications
- [ ] Sequence classification
- [ ] Sequence-to-sequence (Seq2Seq) with encoder-decoder
- [ ] Language modeling
- [ ] Time series prediction
- [ ] Speech recognition (overview)

---

### 4.5 Attention Mechanisms and Transformers

#### 4.5.1 Attention Fundamentals

📚 **Best Resources to Learn:**
- Jay Alammar "The Illustrated Transformer" — jalammar.github.io/illustrated-transformer — **ESSENTIAL**
- "Attention Is All You Need" paper — arxiv.org/abs/1706.03762
- Harvard "The Annotated Transformer" — nlp.seas.harvard.edu/annotated-transformer

- [ ] Motivation: limitations of fixed-length context vectors
- [ ] Bahdanau attention (additive attention)
- [ ] Luong attention (dot-product, general, concat)
- [ ] Attention weights visualization
- [ ] Self-attention


🏋️ **Exercises:**
1. Implement Bahdanau (additive) attention from scratch
2. Implement scaled dot-product attention from scratch in PyTorch
3. Visualize attention weights for a trained Seq2Seq model

#### 4.5.2 The Transformer Architecture

📚 **Best Resources to Learn:**
- Karpathy "Let's build GPT from scratch" (2hr video) — youtu.be/kCc8FmEb1nY — **transformative tutorial**
- Jay Alammar "Illustrated Transformer" + "Illustrated GPT-2"
- d2l.ai Ch.11
- Andrej Karpathy "Zero to Hero" series

- [ ] **"Attention Is All You Need" (Vaswani et al., 2017)**
- [ ] **Encoder-Decoder structure**
- [ ] **Scaled Dot-Product Attention**
  - [ ] Query, Key, Value (Q, K, V)
  - [ ] Attention score computation: softmax(QKᵀ / √d_k) V
  - [ ] Scaling factor √d_k
  - [ ] Attention masking (causal / padding masks)
- [ ] **Multi-Head Attention (MHA)**
  - [ ] Parallel attention heads
  - [ ] Concatenation and linear projection
  - [ ] Number of heads as hyperparameter
- [ ] **Position-wise Feed-Forward Networks (FFN)**
  - [ ] Two linear layers with activation
  - [ ] Hidden dimension expansion (typically 4×)
- [ ] **Positional Encoding**
  - [ ] Sinusoidal positional encoding
  - [ ] Learned positional embeddings
  - [ ] Rotary Position Embedding (RoPE)
  - [ ] ALiBi (Attention with Linear Biases)
  - [ ] Relative positional encoding
- [ ] **Residual connections and layer normalization**
  - [ ] Pre-norm vs post-norm
- [ ] **Encoder architecture (stack of N layers)**
- [ ] **Decoder architecture**
  - [ ] Masked self-attention (causal masking)
  - [ ] Cross-attention (encoder-decoder attention)
  - [ ] Autoregressive generation
- [ ] **Training: teacher forcing**
- [ ] **Complexity: O(n²) attention**


🏋️ **Exercises:**
1. Follow Karpathy's "Let's build GPT" end-to-end
2. Implement multi-head attention from scratch
3. Build a complete Transformer encoder block (attention + FFN + norm + residual)
4. Train a character-level Transformer language model on Shakespeare

🛠️ **PROJECT: Build GPT from Scratch** — Follow Karpathy tutorial, train on Shakespeare, generate text, visualize attention, extend to your own corpus.

#### 4.5.3 Efficient Attention Variants

📚 **Best Resources to Learn:**
- "FlashAttention" paper — arxiv.org/abs/2205.14135
- Hugging Face "Optimizing LLMs for Speed and Memory" — huggingface.co/docs/transformers/llm_tutorial_optimization
- Various architecture survey papers

- [ ] Multi-Query Attention (MQA)
- [ ] Grouped-Query Attention (GQA)
- [ ] Flash Attention (I/O-aware exact attention)
- [ ] Flash Attention 2, 3
- [ ] Linear attention (kernel-based)
- [ ] Sparse attention (local, strided, fixed)
- [ ] Linformer
- [ ] Performer
- [ ] Longformer (local + global attention)
- [ ] BigBird
- [ ] Reformer (LSH attention)
- [ ] Multi-scale attention
- [ ] Sliding window attention
- [ ] Ring attention (for very long sequences)
- [ ] Paged Attention (vLLM)


🏋️ **Exercises:**
1. Compare standard attention vs Flash Attention speeds using PyTorch benchmarks
2. Implement sliding window attention from scratch

#### 4.5.4 Transformer Variants
- [ ] Encoder-only models (BERT family)
- [ ] Decoder-only models (GPT family)
- [ ] Encoder-decoder models (T5, BART)
- [ ] Vision Transformer (ViT) — overview (detailed in CV)
- [ ] Mixture-of-Experts Transformers (overview)

---

### 4.6 Autoencoders

#### 4.6.1 Basic Autoencoders

📚 **Best Resources to Learn:**
- d2l.ai autoencoder sections
- 📖 Understanding Deep Learning Ch.17

- [ ] Encoder-decoder structure
- [ ] Bottleneck / latent representation
- [ ] Undercomplete autoencoders
- [ ] Overcomplete autoencoders
- [ ] Reconstruction loss


🏋️ **Exercises:**
1. Build a convolutional autoencoder for MNIST: visualize reconstructions + latent space
2. Build a denoising autoencoder: add noise, train to reconstruct clean images
3. Use autoencoder for anomaly detection: threshold on reconstruction error

#### 4.6.2 Autoencoder Variants
- [ ] Denoising autoencoders
- [ ] Sparse autoencoders
- [ ] Contractive autoencoders
- [ ] Variational Autoencoders (VAE) — (detailed in generative AI)
- [ ] Convolutional autoencoders
- [ ] Recurrent autoencoders

#### 4.6.3 Applications
- [ ] Dimensionality reduction
- [ ] Feature learning
- [ ] Anomaly detection
- [ ] Image denoising
- [ ] Data generation (VAE)

---

### 4.7 Frameworks and Tools

#### 4.7.1 PyTorch

📚 **Best Resources to Learn:**
- PyTorch official tutorials — pytorch.org/tutorials — **START HERE**
- PyTorch Lightning docs — lightning.ai/docs/pytorch
- "Deep Learning with PyTorch" (FREE book) — pytorch.org/deep-learning-with-pytorch

- [ ] Tensors and operations
- [ ] Autograd (automatic differentiation)
- [ ] `nn.Module` and custom models
- [ ] `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`
- [ ] Common layers (`Linear`, `Conv2d`, `BatchNorm2d`, `Dropout`, `Embedding`)
- [ ] Loss functions in PyTorch
- [ ] Optimizers (`torch.optim`)
- [ ] Learning rate schedulers
- [ ] `DataLoader` and `Dataset` classes
- [ ] Custom datasets
- [ ] Training loop implementation
- [ ] Model saving and loading (`state_dict`, `torch.save`, `torch.load`)
- [ ] GPU training (`.to(device)`, `cuda`)
- [ ] PyTorch Lightning (structured training)
- [ ] TorchMetrics
- [ ] TorchVision, TorchText, TorchAudio
- [ ] ONNX export
- [ ] TorchScript and JIT compilation
- [ ] torch.compile (PyTorch 2.0+)


🏋️ **Exercises:**
1. Rewrite the same CNN in both PyTorch and TensorFlow, compare
2. Set up W&B, log hyperparameters + loss curves + model artifacts for a training run
3. Build a custom Dataset and DataLoader for a non-standard format
4. Refactor a raw training loop into PyTorch Lightning

🛠️ **PHASE 4 CAPSTONE PROJECT: Full Deep Learning System** — Build a complete system (image classifier + text describer, or multimodal Q&A). PyTorch, W&B logging, proper evaluation, deployed as Gradio app.

#### 4.7.2 TensorFlow / Keras
- [ ] Tensors and eager execution
- [ ] `tf.keras.Sequential` and Functional API
- [ ] Subclassing `tf.keras.Model`
- [ ] Common layers
- [ ] Callbacks (EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
- [ ] `tf.data` pipeline
- [ ] `tf.GradientTape` for custom training
- [ ] SavedModel and TF Lite
- [ ] TF Serving (overview)

#### 4.7.3 Experiment Tracking
- [ ] TensorBoard
- [ ] Weights & Biases (wandb)
- [ ] MLflow
- [ ] Neptune.ai (overview)
- [ ] Comet ML (overview)

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 5: SPECIALIZATION TRACKS
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 5.1 Natural Language Processing (NLP)

#### 5.1.1 Text Preprocessing

📚 **Best Resources to Learn:**
- Stanford CS224n — web.stanford.edu/class/cs224n (FREE video lectures)
- Hugging Face LLM Course — huggingface.co/learn/llm-course
- 📖 Speech & Language Processing (FREE) — web.stanford.edu/~jurafsky/slp3
- spaCy free course — course.spacy.io

- [ ] Tokenization (word, subword, character)
- [ ] Lowercasing
- [ ] Stopword removal
- [ ] Stemming (Porter, Snowball)
- [ ] Lemmatization (WordNet, spaCy)
- [ ] Punctuation removal
- [ ] Regular expressions for text cleaning
- [ ] Sentence segmentation
- [ ] Language detection


🏋️ **Exercises:**
1. Build a complete text preprocessing pipeline: tokenize, lowercase, remove stopwords, lemmatize, clean
2. Compare word tokenization vs subword (BPE) tokenization on the same corpus
3. Implement a BPE tokenizer from scratch in Python

#### 5.1.2 Text Representation
- [ ] Bag-of-Words (BoW)
- [ ] Term Frequency (TF)
- [ ] TF-IDF (Term Frequency–Inverse Document Frequency)
- [ ] N-grams (unigrams, bigrams, trigrams)
- [ ] One-hot encoding for words
- [ ] Co-occurrence matrices

#### 5.1.3 Word Embeddings

📚 **Best Resources to Learn:**
- CS224n Lectures 1–2
- Gensim Word2Vec tutorial — radimrehurek.com/gensim
- Jay Alammar "Illustrated Word2Vec" — jalammar.github.io/illustrated-word2vec

- [ ] Word2Vec
  - [ ] Skip-gram model
  - [ ] CBOW (Continuous Bag of Words)
  - [ ] Negative sampling
  - [ ] Hierarchical softmax
- [ ] GloVe (Global Vectors)
- [ ] FastText (subword embeddings)
- [ ] ELMo (Embeddings from Language Models)
  - [ ] Contextualized embeddings
  - [ ] Character-level CNN + BiLSTM
- [ ] Word embedding properties
  - [ ] Analogy relationships (king - man + woman = queen)
  - [ ] Cosine similarity
  - [ ] Embedding visualization (t-SNE, PCA)


🏋️ **Exercises:**
1. Train Word2Vec on a corpus using Gensim, explore analogies (king-man+woman=queen)
2. Visualize embeddings with t-SNE, color by semantic category
3. Compare Word2Vec, GloVe, FastText embeddings on word similarity benchmarks

#### 5.1.4 Subword Tokenization
- [ ] Byte Pair Encoding (BPE)
- [ ] WordPiece
- [ ] Unigram / SentencePiece
- [ ] Byte-level BPE (GPT-2 style)
- [ ] Tokenizer libraries (Hugging Face Tokenizers)

#### 5.1.5 Classical NLP Models
- [ ] Naive Bayes for text classification
- [ ] TF-IDF + Logistic Regression
- [ ] Hidden Markov Models (HMM) for POS tagging
- [ ] Conditional Random Fields (CRF)
- [ ] Latent Dirichlet Allocation (LDA) for topic modeling

#### 5.1.6 Neural NLP Models
- [ ] RNN/LSTM/GRU for text
- [ ] Bidirectional models
- [ ] Sequence-to-sequence (Seq2Seq)
  - [ ] Encoder-decoder architecture
  - [ ] Attention mechanism in Seq2Seq
  - [ ] Beam search decoding
  - [ ] Greedy decoding
  - [ ] Top-k sampling, top-p (nucleus) sampling
  - [ ] Temperature scaling
- [ ] Text CNN (Kim, 2014)

#### 5.1.7 Pre-trained Language Models

📚 **Best Resources to Learn:**
- Hugging Face Transformers docs — huggingface.co/docs/transformers
- Jay Alammar "Illustrated BERT" — jalammar.github.io/illustrated-bert
- CS224n Lectures 9–14

- [ ] **BERT (Bidirectional Encoder Representations from Transformers)**
  - [ ] Masked Language Modeling (MLM)
  - [ ] Next Sentence Prediction (NSP)
  - [ ] [CLS] token for classification
  - [ ] Fine-tuning BERT
  - [ ] BERT variants: RoBERTa, ALBERT, DistilBERT, DeBERTa
- [ ] **GPT family**
  - [ ] Autoregressive language modeling
  - [ ] GPT-1, GPT-2, GPT-3, GPT-4
  - [ ] In-context learning
  - [ ] Few-shot, zero-shot, one-shot learning
- [ ] **T5 (Text-to-Text Transfer Transformer)**
- [ ] **BART (Bidirectional and Auto-Regressive Transformers)**
- [ ] **XLNet** (permutation language modeling)
- [ ] **ELECTRA** (replaced token detection)
- [ ] **Sentence Transformers** (sentence embeddings for similarity)


🏋️ **Exercises:**
1. Fine-tune BERT for sentiment classification using Hugging Face
2. Fine-tune a token classification model for NER
3. Compare BERT, RoBERTa, DistilBERT on 3 NLP benchmarks
4. Use Sentence Transformers for semantic search on a document collection

🛠️ **NLP PROJECTS:**
1. **Sentiment Analysis Pipeline** — scrape reviews, compare TF-IDF+LR vs BERT, deploy
2. **Document Summarizer** — fine-tune T5/BART for abstractive summarization
3. **Q&A System** — extractive QA with BERT on custom knowledge base

#### 5.1.8 NLP Tasks
- [ ] Text classification (sentiment, topic, intent)
- [ ] Named Entity Recognition (NER)
  - [ ] BIO / BILOU tagging
  - [ ] Token classification with transformers
- [ ] Part-of-Speech (POS) tagging
- [ ] Dependency parsing
- [ ] Constituency parsing
- [ ] Machine translation
- [ ] Text summarization (extractive, abstractive)
- [ ] Question answering (extractive, generative)
- [ ] Natural Language Inference (NLI)
- [ ] Semantic textual similarity
- [ ] Coreference resolution
- [ ] Relation extraction
- [ ] Text generation
- [ ] Dialogue systems
- [ ] Information retrieval
  - [ ] BM25
  - [ ] Dense retrieval (bi-encoder, cross-encoder)
  - [ ] Re-ranking

#### 5.1.9 NLP Tools and Libraries
- [ ] Hugging Face Transformers
- [ ] Hugging Face Datasets
- [ ] spaCy
- [ ] NLTK
- [ ] Gensim
- [ ] Stanza (Stanford NLP)
- [ ] SentenceTransformers

---

### 5.2 Computer Vision (CV)

#### 5.2.1 Image Basics
- [ ] Image representation (pixels, channels, resolution)
- [ ] Color spaces (RGB, BGR, HSV, grayscale)
- [ ] Image formats (JPEG, PNG, TIFF, WebP)
- [ ] Image loading and manipulation (PIL/Pillow, OpenCV)
- [ ] Image preprocessing (resize, crop, normalize)

#### 5.2.2 Classical Computer Vision
- [ ] Edge detection (Sobel, Canny)
- [ ] Image filtering (blur, sharpen, median)
- [ ] Morphological operations (erosion, dilation, opening, closing)
- [ ] Histogram equalization
- [ ] Template matching
- [ ] Hough transform (lines, circles)
- [ ] Feature descriptors (SIFT, SURF, ORB, HOG)
- [ ] Image thresholding (Otsu, adaptive)
- [ ] Contour detection

#### 5.2.3 Data Augmentation for Vision
- [ ] Geometric transforms (flip, rotation, scaling, cropping, shearing)
- [ ] Color jittering (brightness, contrast, saturation, hue)
- [ ] Random erasing
- [ ] Cutout
- [ ] Mixup
- [ ] CutMix
- [ ] AutoAugment / RandAugment
- [ ] AugMax
- [ ] Mosaic augmentation (YOLO style)
- [ ] Albumentations library
- [ ] torchvision.transforms

#### 5.2.4 Image Classification
- [ ] All CNN architectures (see Phase 4.3.3)
- [ ] Vision Transformers (ViT)
  - [ ] Patch embedding
  - [ ] [CLS] token
  - [ ] Position embeddings
  - [ ] Pre-training on ImageNet-21k
- [ ] DeiT (Data-efficient Image Transformers)
- [ ] Swin Transformer (shifted window attention)
  - [ ] Swin v2
- [ ] BEiT (BERT pre-training for images)
- [ ] MAE (Masked Autoencoders)
- [ ] DINO / DINOv2 (self-supervised vision)
- [ ] SigLIP
- [ ] Knowledge distillation for vision

#### 5.2.5 Object Detection

📚 **Best Resources to Learn:**
- Ultralytics YOLO docs — docs.ultralytics.com
- Detectron2 (Meta) — github.com/facebookresearch/detectron2
- CS231n object detection lecture

- [ ] **Two-stage detectors**
  - [ ] R-CNN
  - [ ] Fast R-CNN
  - [ ] Faster R-CNN
    - [ ] Region Proposal Network (RPN)
    - [ ] Anchor boxes
    - [ ] Non-Maximum Suppression (NMS)
    - [ ] RoI Pooling, RoI Align
  - [ ] Cascade R-CNN
  - [ ] Feature Pyramid Network (FPN)
- [ ] **One-stage detectors**
  - [ ] YOLO (v1 through v11 / YOLO-World)
    - [ ] Grid-based detection
    - [ ] Anchor-free YOLO (v8+)
  - [ ] SSD (Single Shot MultiBox Detector)
  - [ ] RetinaNet (focal loss)
  - [ ] EfficientDet
  - [ ] CenterNet (anchor-free)
  - [ ] FCOS (Fully Convolutional One-Stage)
- [ ] **Transformer-based detectors**
  - [ ] DETR (DEtection TRansformer)
    - [ ] Bipartite matching loss
    - [ ] Hungarian algorithm
  - [ ] Deformable DETR
  - [ ] DINO (DETR with Improved DeNoising)
  - [ ] RT-DETR (real-time)
  - [ ] Co-DETR
  - [ ] Grounding DINO (open-vocabulary detection)
- [ ] **Evaluation metrics for detection**
  - [ ] Intersection over Union (IoU)
  - [ ] mAP (mean Average Precision) @ IoU thresholds
  - [ ] COCO evaluation metrics (AP, AP50, AP75, APs, APm, APl)


🏋️ **Exercises:**
1. Train YOLOv8 on a custom dataset (label with Roboflow or CVAT)
2. Implement Non-Maximum Suppression from scratch
3. Compare YOLO vs Faster R-CNN vs DETR on COCO subset

🛠️ **CV PROJECTS:**
1. **Custom Object Detector** — collect/label images, train YOLO, deploy with Gradio
2. **Medical Image Segmentation** — U-Net on X-rays or retinal scans
3. **Real-Time Pose Estimation** — MediaPipe/HRNet with webcam feed

#### 5.2.6 Image Segmentation
- [ ] **Semantic segmentation** (pixel-level classification)
  - [ ] Fully Convolutional Network (FCN)
  - [ ] U-Net
  - [ ] SegNet
  - [ ] DeepLab (v1, v2, v3, v3+) — atrous spatial pyramid pooling (ASPP)
  - [ ] PSPNet (Pyramid Scene Parsing)
  - [ ] Mask2Former
  - [ ] SegFormer
- [ ] **Instance segmentation** (objects + masks)
  - [ ] Mask R-CNN
  - [ ] YOLACT
  - [ ] SOLOv2
- [ ] **Panoptic segmentation** (semantic + instance)
  - [ ] Panoptic FPN
  - [ ] Panoptic SegFormer
- [ ] **Segment Anything Model (SAM)**
  - [ ] SAM architecture
  - [ ] SAM 2 (video)
  - [ ] Prompt-based segmentation
- [ ] **Evaluation: mIoU, pixel accuracy, panoptic quality**

#### 5.2.7 Image Generation (see also 5.4 Generative AI)
- [ ] Neural style transfer
- [ ] Super-resolution (SRCNN, ESRGAN, Real-ESRGAN)
- [ ] Image inpainting
- [ ] Image-to-image translation (Pix2Pix, CycleGAN)

#### 5.2.8 3D Vision and Advanced Topics
- [ ] Depth estimation (monocular, stereo)
- [ ] 3D object detection
- [ ] Point cloud processing (PointNet, PointNet++)
- [ ] Neural Radiance Fields (NeRF)
- [ ] 3D Gaussian Splatting
- [ ] Optical flow estimation
- [ ] Video understanding
  - [ ] Action recognition (C3D, I3D, SlowFast)
  - [ ] Video object tracking
  - [ ] Video segmentation
- [ ] Pose estimation (OpenPose, HRNet, MediaPipe)
- [ ] Face detection and recognition
- [ ] OCR (Optical Character Recognition)
  - [ ] CRNN, TrOCR
  - [ ] Document AI

#### 5.2.9 CV Tools and Libraries
- [ ] OpenCV
- [ ] torchvision
- [ ] Detectron2
- [ ] MMDetection, MMSegmentation
- [ ] Ultralytics (YOLO)
- [ ] Hugging Face vision models
- [ ] Albumentations

---

### 5.3 Reinforcement Learning (RL)

#### 5.3.1 RL Fundamentals

📚 **Best Resources to Learn:**
- David Silver RL Course (10 lectures) — davidsilver.uk/teaching — **THE classic RL course**
- 📖 Sutton & Barto (FREE PDF) — incompleteideas.net/book/RLbook2018.pdf
- OpenAI Spinning Up — spinningup.openai.com
- Hugging Face Deep RL Course — huggingface.co/learn/deep-rl-course

- [ ] Agent, environment, state, action, reward
- [ ] Markov Decision Processes (MDPs)
  - [ ] State space, action space
  - [ ] Transition probabilities
  - [ ] Reward function
  - [ ] Discount factor (γ)
- [ ] Episodic vs continuing tasks
- [ ] Return (cumulative discounted reward)
- [ ] Value function V(s)
- [ ] Action-value function Q(s, a)
- [ ] Bellman equations
  - [ ] Bellman expectation equation
  - [ ] Bellman optimality equation
- [ ] Policy (deterministic vs stochastic)
- [ ] Optimal policy


🏋️ **Exercises:**
1. Implement Q-Learning from scratch, solve FrozenLake
2. Implement REINFORCE from scratch, solve CartPole
3. Train DQN on Atari Breakout using Stable Baselines3
4. Implement PPO from scratch (follow CleanRL)

🛠️ **RL PROJECT: Game-Playing Agent** — Train agent on Atari games, progress from DQN → PPO, visualize learning.

#### 5.3.2 Tabular Methods
- [ ] **Dynamic Programming**
  - [ ] Policy evaluation (iterative)
  - [ ] Policy improvement
  - [ ] Policy iteration
  - [ ] Value iteration
- [ ] **Monte Carlo Methods**
  - [ ] First-visit vs every-visit MC
  - [ ] MC control (on-policy, off-policy)
  - [ ] Importance sampling
- [ ] **Temporal Difference (TD) Learning**
  - [ ] TD(0)
  - [ ] SARSA (on-policy TD control)
  - [ ] Q-Learning (off-policy TD control)
  - [ ] Expected SARSA
  - [ ] Double Q-Learning
  - [ ] TD(λ) and eligibility traces
  - [ ] n-step TD methods
- [ ] **Exploration vs exploitation**
  - [ ] ε-greedy
  - [ ] Softmax / Boltzmann exploration
  - [ ] UCB (Upper Confidence Bound)
  - [ ] Thompson sampling
  - [ ] Optimistic initialization
  - [ ] Curiosity-driven exploration
  - [ ] Count-based exploration

#### 5.3.3 Deep Reinforcement Learning
- [ ] **Deep Q-Network (DQN)**
  - [ ] Experience replay
  - [ ] Target network
  - [ ] Double DQN
  - [ ] Dueling DQN
  - [ ] Prioritized experience replay
  - [ ] Rainbow (combining DQN improvements)
  - [ ] Noisy Networks
  - [ ] Distributional RL (C51, QR-DQN)
- [ ] **Policy Gradient Methods**
  - [ ] REINFORCE algorithm
  - [ ] Baseline subtraction
  - [ ] Actor-Critic methods
    - [ ] A2C (Advantage Actor-Critic)
    - [ ] A3C (Asynchronous A2C)
  - [ ] PPO (Proximal Policy Optimization)
    - [ ] Clipped surrogate objective
    - [ ] GAE (Generalized Advantage Estimation)
  - [ ] TRPO (Trust Region Policy Optimization)
  - [ ] SAC (Soft Actor-Critic)
    - [ ] Maximum entropy RL
  - [ ] DDPG (Deep Deterministic Policy Gradient)
  - [ ] TD3 (Twin Delayed DDPG)
- [ ] **Model-Based RL**
  - [ ] World models
  - [ ] Dyna architecture
  - [ ] MuZero
  - [ ] Dreamer (v1, v2, v3)
  - [ ] Model Predictive Control (MPC) with learned models
- [ ] **Multi-Agent RL (MARL)**
  - [ ] Cooperative, competitive, mixed
  - [ ] MADDPG
  - [ ] QMIX
  - [ ] Communication between agents
- [ ] **Inverse RL and Imitation Learning**
  - [ ] Behavioral cloning
  - [ ] DAgger
  - [ ] Inverse RL
  - [ ] GAIL (Generative Adversarial Imitation Learning)
- [ ] **Offline RL (batch RL)**
  - [ ] Conservative Q-Learning (CQL)
  - [ ] Decision Transformer
  - [ ] IQL (Implicit Q-Learning)
- [ ] **Hierarchical RL**
  - [ ] Options framework
  - [ ] Feudal networks
  - [ ] Goal-conditioned RL

#### 5.3.4 RL Environments and Tools
- [ ] OpenAI Gymnasium (formerly Gym)
- [ ] Stable Baselines3
- [ ] CleanRL
- [ ] RLlib (Ray)
- [ ] MuJoCo, Isaac Gym
- [ ] PettingZoo (multi-agent)
- [ ] Atari environments
- [ ] MinAtar (lightweight Atari)

---

### 5.4 Generative AI

#### 5.4.1 Variational Autoencoders (VAEs)

📚 **Best Resources to Learn:**
- Hugging Face Diffusion Models Course — huggingface.co/learn/diffusion-course
- MIT CSAIL Flow Matching & Diffusion — diffusion.csail.mit.edu/2025
- Lil'Log diffusion survey — lilianweng.github.io
- 📖 Understanding Deep Learning Ch.12–18

- [ ] Latent variable model
- [ ] Encoder (recognition model) q(z|x)
- [ ] Decoder (generative model) p(x|z)
- [ ] Reparameterization trick
- [ ] Evidence Lower Bound (ELBO)
- [ ] KL divergence regularization
- [ ] Reconstruction loss
- [ ] β-VAE (disentangled representations)
- [ ] VQ-VAE (Vector Quantized VAE)
- [ ] VQ-VAE-2
- [ ] Conditional VAE (CVAE)


🏋️ **Exercises:**
1. Build a VAE from scratch, train on MNIST, visualize latent space interpolation
2. Train DCGAN on CelebA, generate face images
3. Build a DDPM from scratch following DeepLearning.AI short course
4. Fine-tune Stable Diffusion with LoRA on custom images

🛠️ **Generative AI PROJECTS:**
1. **Build Diffusion Model from Scratch** — DDPM on CIFAR-10
2. **Custom Image Generator** — DreamBooth/LoRA fine-tune Stable Diffusion on your style
3. **Style Transfer App** — real-time neural style transfer with Gradio

#### 5.4.2 Generative Adversarial Networks (GANs)
- [ ] Generator and discriminator
- [ ] Adversarial training (minimax game)
- [ ] Nash equilibrium
- [ ] Mode collapse
- [ ] Training instability and tips
- [ ] **GAN variants**
  - [ ] DCGAN (Deep Convolutional GAN)
  - [ ] WGAN (Wasserstein GAN) — Earth mover's distance
  - [ ] WGAN-GP (gradient penalty)
  - [ ] Conditional GAN (cGAN)
  - [ ] InfoGAN
  - [ ] CycleGAN (unpaired image translation)
  - [ ] Pix2Pix (paired image translation)
  - [ ] StyleGAN (v1, v2, v3)
    - [ ] Mapping network
    - [ ] Adaptive instance normalization (AdaIN)
    - [ ] Progressive growing
  - [ ] ProGAN (Progressive Growing GAN)
  - [ ] BigGAN
  - [ ] StarGAN
- [ ] GAN evaluation metrics
  - [ ] FID (Fréchet Inception Distance)
  - [ ] IS (Inception Score)
  - [ ] KID (Kernel Inception Distance)
  - [ ] LPIPS (perceptual similarity)

#### 5.4.3 Diffusion Models
- [ ] **Forward diffusion process** (adding noise)
  - [ ] Noise schedule (linear, cosine)
  - [ ] Variance schedule
- [ ] **Reverse diffusion process** (denoising)
- [ ] **Denoising Diffusion Probabilistic Models (DDPM)**
  - [ ] Noise prediction (ε-prediction)
  - [ ] Score matching
  - [ ] Training objective (simplified MSE)
- [ ] **DDIM (Denoising Diffusion Implicit Models)**
  - [ ] Deterministic sampling
  - [ ] Fewer sampling steps
- [ ] **Score-based models (Song & Ermon)**
  - [ ] Score function ∇_x log p(x)
  - [ ] Langevin dynamics sampling
  - [ ] Noise-conditioned score networks (NCSN)
- [ ] **Latent Diffusion Models (LDM)**
  - [ ] Diffusion in latent space
  - [ ] VAE encoder/decoder
  - [ ] U-Net backbone
  - [ ] Cross-attention conditioning
- [ ] **Stable Diffusion (v1.5, 2.0, XL, 3.0)**
  - [ ] Architecture overview
  - [ ] Text conditioning (CLIP text encoder)
  - [ ] Classifier-free guidance (CFG)
  - [ ] Img2img, inpainting
- [ ] **Flux architecture**
- [ ] **ControlNet** (controllable generation)
- [ ] **LoRA for diffusion** (fine-tuning with low-rank adapters)
- [ ] **DreamBooth** (few-shot personalization)
- [ ] **Textual inversion**
- [ ] **Consistency models**
- [ ] **Distilled diffusion models** (progressive distillation, LCM)
- [ ] **Rectified Flow**

#### 5.4.4 Flow-Based Models and Flow Matching
- [ ] Normalizing flows
  - [ ] Change of variables formula
  - [ ] Invertible transformations
  - [ ] RealNVP, GLOW, Neural Spline Flows
- [ ] Continuous Normalizing Flows (CNF)
  - [ ] Neural ODEs
  - [ ] FFJORD
- [ ] Flow Matching
  - [ ] Conditional flow matching
  - [ ] Optimal transport paths
  - [ ] Relationship to diffusion models
  - [ ] Advantages over diffusion (straight paths)
- [ ] Stochastic interpolants

#### 5.4.5 Autoregressive Models for Generation
- [ ] PixelCNN, PixelRNN
- [ ] WaveNet (audio generation)
- [ ] Autoregressive image generation (ImageGPT)
- [ ] Vector-quantized autoregressive (VQ-VAE + transformer)

#### 5.4.6 Audio Generation
- [ ] WaveNet
- [ ] Bark
- [ ] MusicGen
- [ ] AudioLDM
- [ ] VALL-E (voice cloning)
- [ ] Text-to-speech (TTS) overview

---

### 5.5 Time Series Analysis and Forecasting

#### 5.5.1 Time Series Fundamentals

📚 **Best Resources to Learn:**
- Forecasting: Principles & Practice (FREE) — otexts.com/fpp3 — **THE time series textbook**
- Kaggle Time Series course (FREE) — kaggle.com/learn/time-series
- statsmodels docs — statsmodels.org

- [ ] Time series components (trend, seasonality, cyclical, irregular)
- [ ] Stationarity
  - [ ] Strict vs weak stationarity
  - [ ] Augmented Dickey-Fuller (ADF) test
  - [ ] KPSS test
- [ ] Differencing (to achieve stationarity)
- [ ] Autocorrelation (ACF) and partial autocorrelation (PACF)
- [ ] White noise
- [ ] Random walk


🏋️ **Exercises:**
1. Fit ARIMA to airline passenger data, forecast 12 months
2. Train Prophet with holidays and changepoints on daily sales
3. Build LSTM for stock prediction, compare with ARIMA
4. Use Darts library to compare 5 forecasting models

🛠️ **Time Series PROJECT:** Multi-horizon forecasting system — compare ARIMA, Prophet, LSTM, TFT; walk-forward validation; dashboard with confidence intervals.

#### 5.5.2 Classical Methods
- [ ] Moving averages (simple, weighted, exponential)
- [ ] Exponential smoothing
  - [ ] Simple exponential smoothing
  - [ ] Holt's linear trend method
  - [ ] Holt-Winters (additive, multiplicative seasonality)
- [ ] ARIMA models
  - [ ] AR (Autoregressive) model
  - [ ] MA (Moving Average) model
  - [ ] ARMA model
  - [ ] ARIMA (Autoregressive Integrated Moving Average)
  - [ ] SARIMA (seasonal ARIMA)
  - [ ] Box-Jenkins methodology
  - [ ] Model identification (ACF, PACF analysis)
  - [ ] AIC / BIC for model selection
- [ ] SARIMAX (with exogenous variables)
- [ ] VAR (Vector Autoregression) — multivariate
- [ ] GARCH (volatility modeling)
- [ ] Prophet (Facebook/Meta)
  - [ ] Trend changepoints
  - [ ] Seasonality modeling
  - [ ] Holiday effects

#### 5.5.3 Deep Learning for Time Series
- [ ] RNN/LSTM/GRU for forecasting
- [ ] Sequence-to-sequence for multi-step forecasting
- [ ] Temporal Convolutional Networks (TCN)
  - [ ] Dilated causal convolutions
  - [ ] WaveNet-style architecture
- [ ] Transformer-based models
  - [ ] Temporal Fusion Transformer (TFT)
  - [ ] Informer
  - [ ] Autoformer
  - [ ] PatchTST
  - [ ] iTransformer
  - [ ] TimesFM (Google)
- [ ] Foundation models for time series
  - [ ] TimeGPT
  - [ ] Lag-Llama
  - [ ] Chronos (Amazon)
  - [ ] Moirai
- [ ] N-BEATS (Neural Basis Expansion Analysis)
- [ ] N-HiTS
- [ ] DeepAR (Amazon)
- [ ] NeuralProphet

#### 5.5.4 Time Series Evaluation
- [ ] MAE, MSE, RMSE, MAPE, sMAPE
- [ ] MASE (Mean Absolute Scaled Error)
- [ ] Walk-forward validation
- [ ] Backtesting
- [ ] Probabilistic forecasting (prediction intervals)

#### 5.5.5 Time Series Anomaly Detection
- [ ] Statistical methods (Z-score, moving average deviation)
- [ ] Isolation Forest for time series
- [ ] LSTM-based anomaly detection
- [ ] Autoencoders for anomaly detection
- [ ] Prophet anomaly detection

---

### 5.6 Recommender Systems

#### 5.6.1 Content-Based Filtering

📚 **Best Resources to Learn:**
- Google Recommendation Systems Course (FREE) — developers.google.com/machine-learning/recommendation
- Andrew Ng ML Specialization Course 3 (RecSys)
- RecBole framework — recbole.io

- [ ] Item feature profiles
- [ ] User preference profiles
- [ ] TF-IDF for content features
- [ ] Cosine similarity matching
- [ ] Limitations (filter bubble, cold start for new items)


🏋️ **Exercises:**
1. Implement collaborative filtering (SVD) from scratch on MovieLens
2. Build a Two-Tower model with PyTorch for item retrieval
3. Implement BPR (Bayesian Personalized Ranking) from scratch

🛠️ **RecSys PROJECT:** Movie Recommendation Engine — MovieLens, CF + content-based hybrid, NDCG evaluation, Streamlit deployment.

#### 5.6.2 Collaborative Filtering
- [ ] **Memory-based (neighborhood)**
  - [ ] User-based collaborative filtering
  - [ ] Item-based collaborative filtering
  - [ ] Similarity metrics (cosine, Pearson, Jaccard)
  - [ ] K-nearest neighbors approach
- [ ] **Model-based**
  - [ ] Matrix factorization
    - [ ] SVD for recommendations
    - [ ] Alternating Least Squares (ALS)
    - [ ] Non-negative Matrix Factorization (NMF)
    - [ ] Probabilistic Matrix Factorization (PMF)
  - [ ] Factorization Machines
  - [ ] Bayesian Personalized Ranking (BPR)

#### 5.6.3 Deep Learning Recommenders
- [ ] Neural Collaborative Filtering (NCF)
- [ ] Autoencoders for collaborative filtering
- [ ] Wide & Deep Learning
- [ ] DeepFM
- [ ] Two-Tower Models (dual encoder)
  - [ ] User tower, item tower
  - [ ] Approximate nearest neighbor retrieval
- [ ] Sequence-aware recommendations (GRU4Rec, SASRec, BERT4Rec)
- [ ] Graph-based recommendations (PinSage, LightGCN)
- [ ] Multi-modal recommendations
- [ ] Cross-encoder re-ranking

#### 5.6.4 Hybrid Approaches
- [ ] Content + collaborative hybrid
- [ ] Switching, mixing, cascading, feature augmentation
- [ ] Knowledge graph-enhanced recommendations

#### 5.6.5 Advanced Topics
- [ ] Cold start problem (new users, new items)
- [ ] Implicit vs explicit feedback
- [ ] Session-based recommendations
- [ ] Context-aware recommendations
- [ ] Multi-objective recommendations (relevance, diversity, novelty)
- [ ] Exploration vs exploitation in recommendations (bandits)
- [ ] Fairness and bias in recommendations
- [ ] LLM-based recommendations

#### 5.6.6 Evaluation
- [ ] Precision@K, Recall@K, NDCG, MAP, MRR, Hit Rate
- [ ] Online A/B testing
- [ ] Offline evaluation with held-out data

---

### 5.7 Graph Neural Networks (GNNs)

#### 5.7.1 Graph Theory Basics

📚 **Best Resources to Learn:**
- Stanford CS224W (FREE lectures) — web.stanford.edu/class/cs224w — **THE GNN course**
- PyTorch Geometric tutorials — pytorch-geometric.readthedocs.io
- DGL docs — dgl.ai

- [ ] Nodes (vertices) and edges
- [ ] Directed vs undirected graphs
- [ ] Weighted graphs
- [ ] Adjacency matrix
- [ ] Degree matrix
- [ ] Laplacian matrix
- [ ] Graph properties (connectivity, diameter, clustering coefficient)
- [ ] Bipartite graphs
- [ ] Heterogeneous graphs


🏋️ **Exercises:**
1. Implement GCN from scratch using message passing
2. Node classification on Cora dataset with GAT using PyG
3. Link prediction on a social network graph

🛠️ **GNN PROJECT:** Molecular Property Prediction — OGB datasets, implement GCN + GAT, compare with non-graph baselines.

#### 5.7.2 GNN Architectures
- [ ] Message passing framework (aggregate, update)
- [ ] Graph Convolutional Network (GCN) — Kipf & Welling
- [ ] GraphSAGE (sampling and aggregating)
- [ ] Graph Attention Network (GAT)
  - [ ] Attention coefficients for neighbors
  - [ ] GATv2
- [ ] Graph Isomorphism Network (GIN)
- [ ] ChebNet (Chebyshev spectral)
- [ ] SplineCNN
- [ ] EdgeConv (DGCNN)
- [ ] PNA (Principal Neighbourhood Aggregation)

#### 5.7.3 GNN Tasks
- [ ] Node classification
- [ ] Link prediction
- [ ] Graph classification
- [ ] Graph generation
- [ ] Node clustering / community detection
- [ ] Graph regression

#### 5.7.4 Advanced GNN Topics
- [ ] Over-smoothing problem
- [ ] Over-squashing problem
- [ ] Heterogeneous graph neural networks (HGT, R-GCN)
- [ ] Temporal / dynamic graph networks (TGN, TGAT)
- [ ] Knowledge graph embeddings (TransE, RotatE, ComplEx)
- [ ] Knowledge graph completion
- [ ] Graph Transformers
- [ ] Equivariant GNNs (for molecules)
- [ ] Positional encodings for graphs (random walk, Laplacian)
- [ ] Scalability (mini-batch training, sampling)
- [ ] Self-supervised learning on graphs (GraphMAE, GraphCL)

#### 5.7.5 Applications
- [ ] Social network analysis
- [ ] Drug discovery / molecular property prediction
- [ ] Traffic prediction
- [ ] Fraud detection
- [ ] Recommendation systems
- [ ] Protein structure prediction

#### 5.7.6 GNN Tools
- [ ] PyTorch Geometric (PyG)
- [ ] Deep Graph Library (DGL)
- [ ] NetworkX (graph analysis)
- [ ] OGB (Open Graph Benchmark)

---

### 5.8 MLOps and Deployment

#### 5.8.1 ML System Design

📚 **Best Resources to Learn:**
- Made With ML (FREE) — madewithml.com — **best free MLOps course**
- MLOps Zoomcamp (FREE + cert) — github.com/DataTalksClub/mlops-zoomcamp
- Full Stack Deep Learning (FREE) — fullstackdeeplearning.com
- 💰 Designing ML Systems — Chip Huyen

- [ ] Framing ML problems
- [ ] Data pipelines (ingestion, storage, processing)
- [ ] Feature stores (Feast, Tecton)
- [ ] Training pipelines
- [ ] Serving infrastructure
- [ ] Monitoring and feedback loops
- [ ] ML system architecture patterns


🏋️ **Exercises:**
1. Wrap a model in FastAPI, test with curl and Python requests
2. Dockerize the API, run in container
3. Set up MLflow: log experiments, register model, serve
4. Set up drift detection with Evidently AI

🛠️ **MLOps PROJECT: End-to-End Pipeline** — Data → preprocess → train → evaluate → serve via FastAPI+Docker → monitor with Evidently → CI/CD with GitHub Actions.

#### 5.8.2 Experiment Tracking and Versioning
- [ ] MLflow (tracking, projects, models, registry)
- [ ] Weights & Biases (experiments, sweeps, artifacts)
- [ ] DVC (Data Version Control)
- [ ] Git LFS
- [ ] Neptune.ai, Comet ML
- [ ] Experiment reproducibility

#### 5.8.3 Model Serving and Deployment
- [ ] REST APIs for ML (Flask, FastAPI)
- [ ] gRPC
- [ ] Model serialization (pickle, joblib, ONNX, TorchScript, SavedModel)
- [ ] TorchServe
- [ ] TensorFlow Serving
- [ ] Triton Inference Server (NVIDIA)
- [ ] BentoML
- [ ] Seldon Core
- [ ] KServe
- [ ] Serverless ML (AWS Lambda, Google Cloud Functions)
- [ ] Edge deployment (TFLite, ONNX Runtime, Core ML)
- [ ] Batch inference vs real-time inference
- [ ] Model compression for deployment
  - [ ] Quantization (INT8, INT4)
  - [ ] Pruning
  - [ ] Knowledge distillation
  - [ ] Low-rank factorization

#### 5.8.4 Containerization and Orchestration
- [ ] Docker for ML
  - [ ] Dockerfile, images, containers
  - [ ] Docker Compose
  - [ ] GPU containers (NVIDIA Docker)
- [ ] Kubernetes basics
  - [ ] Pods, services, deployments
  - [ ] Kubeflow (overview)
- [ ] Container registries

#### 5.8.5 CI/CD for ML
- [ ] GitHub Actions for ML
- [ ] GitLab CI/CD
- [ ] Automated testing for ML (data tests, model tests)
- [ ] ML pipeline orchestration
  - [ ] Airflow
  - [ ] Prefect
  - [ ] Dagster
  - [ ] Kubeflow Pipelines
  - [ ] ZenML
  - [ ] Metaflow

#### 5.8.6 Monitoring and Observability
- [ ] Data drift detection
  - [ ] Concept drift
  - [ ] Feature drift
  - [ ] Label drift
- [ ] Model performance monitoring
- [ ] Alerting systems
- [ ] Evidently AI
- [ ] WhyLabs
- [ ] Prometheus + Grafana for ML metrics
- [ ] A/B testing in production
- [ ] Shadow deployment / canary deployment
- [ ] Blue-green deployment

#### 5.8.7 Cloud Platforms for ML
- [ ] AWS (SageMaker, S3, EC2, Lambda)
- [ ] Google Cloud (Vertex AI, GCS, Compute Engine)
- [ ] Azure (Azure ML, Blob Storage)
- [ ] Databricks
- [ ] Snowflake ML (overview)

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 6: CUTTING-EDGE AI — LLMs, AGENTS & FRONTIER
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 6.1 Large Language Models (LLMs)

#### 6.1.1 Transformer Architecture Deep Dive

📚 **Best Resources to Learn:**
- Karpathy "Neural Networks: Zero to Hero" (7 videos) — youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- Karpathy "Let's build GPT" — youtu.be/kCc8FmEb1nY
- 💰 Raschka "Build an LLM from Scratch" — code FREE at github.com/rasbt/LLMs-from-scratch
- Jay Alammar blogs — jalammar.github.io
- Stanford CME 295 — cme295.stanford.edu
- Hugging Face LLM Course — huggingface.co/learn/llm-course

- [ ] Full encoder-decoder architecture review
- [ ] Decoder-only architecture (GPT-style)
  - [ ] Causal / autoregressive masking
  - [ ] Next-token prediction objective
- [ ] Encoder-only architecture (BERT-style)
  - [ ] Masked language modeling
  - [ ] Bidirectional context
- [ ] Encoder-decoder (T5, BART-style)
- [ ] Architecture choices in modern LLMs
  - [ ] Pre-norm vs post-norm
  - [ ] Parallel attention + FFN (PaLM-style)
  - [ ] SwiGLU / GeGLU activation in FFN
  - [ ] Grouped-Query Attention (GQA)
  - [ ] Rotary Position Embeddings (RoPE)
  - [ ] ALiBi positional encoding
  - [ ] RMSNorm vs LayerNorm
  - [ ] Tied/untied embeddings


🏋️ **Exercises:**
1. Follow Karpathy's entire "Zero to Hero" series (all 7 videos)
2. Build nanoGPT from scratch
3. Follow Raschka's book chapters end-to-end
4. Implement RoPE positional encoding from scratch

🛠️ **PROJECT: Train Your Own Mini-LLM** — Follow Karpathy's nanochat: pre-train → SFT → RL fine-tune. OR follow Raschka's book end-to-end.

#### 6.1.2 Tokenization for LLMs
- [ ] Byte Pair Encoding (BPE) — details
- [ ] WordPiece (BERT)
- [ ] SentencePiece / Unigram
- [ ] Byte-level BPE (GPT-2, GPT-4)
- [ ] Tiktoken (OpenAI tokenizer)
- [ ] Vocabulary size tradeoffs
- [ ] Multilingual tokenization challenges
- [ ] Special tokens ([PAD], [CLS], [SEP], [MASK], [BOS], [EOS])

#### 6.1.3 Pre-training LLMs
- [ ] Pre-training objectives
  - [ ] Causal language modeling (CLM)
  - [ ] Masked language modeling (MLM)
  - [ ] Prefix language modeling
  - [ ] Span corruption (T5)
  - [ ] UL2 (Unifying Language Learning)
- [ ] Pre-training data
  - [ ] Common Crawl, C4, The Pile, RefinedWeb, RedPajama, FineWeb
  - [ ] Data quality and filtering
  - [ ] Deduplication (exact, fuzzy, MinHash)
  - [ ] Data mix ratios (web, books, code, academic)
  - [ ] Toxicity filtering
  - [ ] PII removal
- [ ] Scaling laws
  - [ ] Kaplan et al. scaling laws
  - [ ] Chinchilla optimal scaling (Hoffmann et al.)
  - [ ] Compute-optimal training
  - [ ] Over-training (training smaller models longer)
- [ ] Training infrastructure
  - [ ] Distributed training (3D parallelism)
  - [ ] Mixed precision (BF16)
  - [ ] Gradient checkpointing
  - [ ] ZeRO optimization stages
  - [ ] DeepSpeed, Megatron-LM, FSDP
  - [ ] Hardware: GPU clusters, TPU pods, custom ASICs
- [ ] Training stability
  - [ ] Loss spikes
  - [ ] Learning rate warmup
  - [ ] Gradient clipping
  - [ ] Weight decay
  - [ ] z-loss regularization

#### 6.1.4 Key LLM Families
- [ ] **GPT family** (OpenAI)
  - [ ] GPT-1, GPT-2, GPT-3, GPT-3.5, GPT-4, GPT-4o
  - [ ] Instruction tuning (InstructGPT)
  - [ ] In-context learning, few-shot, zero-shot
  - [ ] Chain-of-thought prompting
- [ ] **LLaMA / Llama family** (Meta)
  - [ ] LLaMA 1, Llama 2, Llama 3, Llama 3.1, Llama 4
  - [ ] Open weights
- [ ] **Mistral / Mixtral** (Mistral AI)
  - [ ] Mistral 7B
  - [ ] Mixtral 8x7B (Mixture of Experts)
  - [ ] Sliding window attention
- [ ] **Gemini / Gemma** (Google)
  - [ ] Gemini Ultra, Pro, Nano
  - [ ] Gemma (open models)
- [ ] **Claude** (Anthropic)
  - [ ] Constitutional AI
  - [ ] Long context (200K+ tokens)
- [ ] **Qwen** (Alibaba)
- [ ] **DeepSeek** (DeepSeek AI)
  - [ ] DeepSeek-V2, V3, R1
  - [ ] Multi-head Latent Attention (MLA)
  - [ ] Mixture of Experts
- [ ] **Phi** (Microsoft) — small models
- [ ] **Command R** (Cohere)
- [ ] **Yi** (01.AI)
- [ ] **Falcon** (TII)
- [ ] **BLOOM** (BigScience)
- [ ] **PaLM / PaLM 2** (Google)
- [ ] **T5 / Flan-T5 / UL2**
- [ ] **OLMo** (AI2) — fully open

#### 6.1.5 Inference Optimization

📚 **Best Resources to Learn:**
- vLLM docs — docs.vllm.ai
- llama.cpp — github.com/ggerganov/llama.cpp
- Hugging Face "Optimizing LLMs" guide

- [ ] KV-Cache
  - [ ] How it works
  - [ ] Memory requirements
  - [ ] Multi-query / grouped-query attention for KV reduction
- [ ] Flash Attention (1, 2, 3)
- [ ] Paged Attention (vLLM)
- [ ] Speculative decoding
- [ ] Continuous batching
- [ ] Quantization for inference
  - [ ] Post-training quantization (PTQ)
  - [ ] GPTQ (weight-only quantization)
  - [ ] AWQ (Activation-aware Weight Quantization)
  - [ ] GGUF / GGML (llama.cpp format)
  - [ ] SmoothQuant
  - [ ] FP8 quantization
  - [ ] INT4, INT8 quantization
  - [ ] Quantization-aware training (QAT)
- [ ] Pruning for LLMs
  - [ ] Unstructured pruning
  - [ ] Structured pruning
  - [ ] SparseGPT
  - [ ] Wanda
- [ ] Knowledge distillation for LLMs
- [ ] Model merging (SLERP, TIES, DARE)
- [ ] Inference engines
  - [ ] vLLM
  - [ ] TensorRT-LLM
  - [ ] llama.cpp
  - [ ] Ollama
  - [ ] SGLang
  - [ ] text-generation-inference (TGI, Hugging Face)
  - [ ] ExLlamaV2


🏋️ **Exercises:**
1. Quantize a 7B model with GPTQ, compare speed + quality with FP16
2. Run same model in vLLM vs llama.cpp vs Ollama — benchmark throughput
3. Implement KV-cache for autoregressive generation from scratch

#### 6.1.6 Context Length Extension
- [ ] Positional encoding extrapolation
- [ ] RoPE scaling (linear, NTK-aware, YaRN)
- [ ] ALiBi for length generalization
- [ ] Sliding window attention
- [ ] Ring attention
- [ ] Landmark attention
- [ ] Long-context training data

---

### 6.2 Fine-Tuning and Alignment

#### 6.2.1 Supervised Fine-Tuning (SFT)
- [ ] Instruction tuning
- [ ] Chat format / template (ChatML, Llama format, etc.)
- [ ] Dataset creation for SFT
  - [ ] Instruction-response pairs
  - [ ] Multi-turn conversations
  - [ ] Data quality > quantity
  - [ ] Open datasets: OpenAssistant, Dolly, Alpaca, WizardLM, UltraChat
- [ ] Full fine-tuning
- [ ] Catastrophic forgetting

#### 6.2.2 Parameter-Efficient Fine-Tuning (PEFT)

📚 **Best Resources to Learn:**
- Hugging Face PEFT library — huggingface.co/docs/peft
- Sebastian Raschka fine-tuning tutorials — magazine.sebastianraschka.com
- "LoRA" paper — arxiv.org/abs/2106.09685

- [ ] **LoRA (Low-Rank Adaptation)**
  - [ ] Low-rank matrices A, B
  - [ ] Rank (r) parameter
  - [ ] Alpha scaling factor
  - [ ] Which layers to apply LoRA
  - [ ] Merging LoRA weights
- [ ] **QLoRA (Quantized LoRA)**
  - [ ] 4-bit NormalFloat quantization
  - [ ] Double quantization
  - [ ] Paged optimizers
- [ ] **DoRA (Weight-Decomposed LRA)**
- [ ] **AdaLoRA (Adaptive LoRA)**
- [ ] **Prefix tuning**
- [ ] **Prompt tuning** (soft prompts)
- [ ] **P-Tuning (v1, v2)**
- [ ] **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)**
- [ ] **Adapter layers** (Houlsby et al.)
- [ ] **LongLoRA** (for context length extension)
- [ ] Comparison of PEFT methods
- [ ] Hugging Face PEFT library


🏋️ **Exercises:**
1. LoRA fine-tune Llama-3/Mistral on custom instruction data using HF PEFT
2. QLoRA (4-bit) fine-tune on single GPU, compare quality with full LoRA
3. Compare LoRA ranks (4, 8, 16, 64): accuracy vs training time
4. Merge LoRA weights into base model, export for deployment

🛠️ **PROJECT: Domain-Specific LLM** — Curate domain data, QLoRA fine-tune 7B model, evaluate on domain benchmarks, deploy with Ollama/vLLM.

#### 6.2.3 Reinforcement Learning from Human Feedback (RLHF)

📚 **Best Resources to Learn:**
- Nathan Lambert "RLHF Book" (FREE) — rlhfbook.com — **THE comprehensive RLHF resource**
- Hugging Face TRL library — huggingface.co/docs/trl
- Hugging Face Alignment Handbook — github.com/huggingface/alignment-handbook
- DeepLearning.AI RLHF short course

- [ ] **The RLHF pipeline**
  - [ ] Step 1: Supervised Fine-Tuning (SFT)
  - [ ] Step 2: Reward Model training
    - [ ] Pairwise comparisons
    - [ ] Bradley-Terry model
    - [ ] Reward model architecture
  - [ ] Step 3: PPO optimization
    - [ ] KL divergence penalty (from reference model)
    - [ ] Value function estimation
    - [ ] Clipping in PPO
    - [ ] Reward hacking / reward gaming
- [ ] **InstructGPT** (the original RLHF paper)
- [ ] Proximal Policy Optimization (PPO) for LLMs
- [ ] REINFORCE Leave-One-Out (RLOO)
- [ ] Rejection sampling
- [ ] Best-of-N sampling


🏋️ **Exercises:**
1. Train a reward model on preference data using TRL
2. Implement DPO training using TRL library
3. Evaluate fine-tuned model on MT-Bench using LLM-as-judge
4. Compare DPO vs PPO on same preference dataset

#### 6.2.4 Direct Alignment Methods
- [ ] **DPO (Direct Preference Optimization)**
  - [ ] Implicit reward model
  - [ ] Pairwise preference data
  - [ ] Relationship to RLHF
  - [ ] Advantages: no reward model needed, stable training
- [ ] **IPO (Identity Preference Optimization)**
- [ ] **KTO (Kahneman-Tversky Optimization)** — unpaired preferences
- [ ] **ORPO (Odds Ratio Preference Optimization)**
- [ ] **SimPO (Simple Preference Optimization)**
- [ ] **CPO (Contrastive Preference Optimization)**
- [ ] **SPIN (Self-Play Fine-Tuning)**
- [ ] **Constitutional AI (CAI)**
  - [ ] Principles-based feedback
  - [ ] RLAIF (RL from AI Feedback)
- [ ] **Self-Rewarding Language Models**
- [ ] **RLVR (Reinforcement Learning with Verifiable Rewards)**
- [ ] **GRPO (Group Relative Policy Optimization)** — DeepSeek R1

#### 6.2.5 LLM Evaluation
- [ ] **Benchmarks**
  - [ ] MMLU (Massive Multitask Language Understanding)
  - [ ] HellaSwag
  - [ ] ARC (AI2 Reasoning Challenge)
  - [ ] TruthfulQA
  - [ ] WinoGrande
  - [ ] GSM8K (math reasoning)
  - [ ] HumanEval / MBPP (code generation)
  - [ ] MATH benchmark
  - [ ] BBH (Big-Bench Hard)
  - [ ] MT-Bench (multi-turn)
  - [ ] AlpacaEval
  - [ ] Chatbot Arena (LMSYS)
  - [ ] HELM (Holistic Evaluation)
  - [ ] Open LLM Leaderboard (Hugging Face)
  - [ ] LiveBench
  - [ ] GPQA (graduate-level QA)
  - [ ] IFEval (instruction following)
  - [ ] SWE-bench (software engineering)
  - [ ] Aider polyglot (coding)
- [ ] **Evaluation approaches**
  - [ ] Automated metrics (BLEU, ROUGE, BERTScore, METEOR)
  - [ ] LLM-as-judge
  - [ ] Human evaluation
  - [ ] Elo ratings
  - [ ] Win rates
- [ ] **Safety evaluation**
  - [ ] Red teaming
  - [ ] Jailbreak resistance
  - [ ] Toxicity benchmarks
  - [ ] Bias evaluation

---

### 6.3 AI Agents and Tool Use

#### 6.3.1 Agent Architectures
- [ ] **ReAct (Reason + Act)**
  - [ ] Thought-action-observation loop
- [ ] **Reflexion** (self-reflection for improvement)
- [ ] **LATS (Language Agent Tree Search)**
- [ ] **Plan-and-Execute agents**
- [ ] **Cognitive architectures** (CoALA)
- [ ] **Multi-agent systems**
  - [ ] Agent-to-Agent communication
  - [ ] Role-based agents
  - [ ] Debate / discussion agents
  - [ ] Hierarchical agent systems
  - [ ] AutoGen, CrewAI, LangGraph

#### 6.3.2 Tool Use and Function Calling
- [ ] Function calling / tool use in LLMs
- [ ] API integration
- [ ] Code execution tools
- [ ] Web browsing tools
- [ ] File manipulation tools
- [ ] Database query tools
- [ ] Calculator / math tools
- [ ] Tool creation by agents

#### 6.3.3 Retrieval-Augmented Generation (RAG)

📚 **Best Resources to Learn:**
- LangChain docs — python.langchain.com/docs
- LlamaIndex docs — docs.llamaindex.ai
- Lil'Log "LLM-Powered Agents" — lilianweng.github.io
- Hugging Face AI Agents Course — huggingface.co/learn/agents-course

- [ ] **RAG pipeline overview**
  - [ ] Indexing (chunking, embedding, storing)
  - [ ] Retrieval (query, search, re-rank)
  - [ ] Generation (context injection, answer)
- [ ] **Chunking strategies**
  - [ ] Fixed-size chunking
  - [ ] Semantic chunking
  - [ ] Recursive character splitting
  - [ ] Document-structure-aware chunking
  - [ ] Sliding window with overlap
- [ ] **Embedding models**
  - [ ] Sentence Transformers
  - [ ] OpenAI embeddings
  - [ ] Cohere embeddings
  - [ ] BGE, GTE, E5 embeddings
  - [ ] Nomic embeddings
  - [ ] Multilingual embeddings
- [ ] **Vector databases**
  - [ ] Pinecone
  - [ ] Weaviate
  - [ ] ChromaDB
  - [ ] Milvus
  - [ ] Qdrant
  - [ ] FAISS (Facebook AI Similarity Search)
  - [ ] pgvector (PostgreSQL)
  - [ ] LanceDB
- [ ] **Similarity search**
  - [ ] Cosine similarity
  - [ ] Euclidean distance
  - [ ] Maximum inner product search (MIPS)
  - [ ] Approximate nearest neighbors (ANN)
    - [ ] HNSW (Hierarchical Navigable Small World)
    - [ ] IVF (Inverted File Index)
    - [ ] Product quantization
- [ ] **Advanced RAG techniques**
  - [ ] Hybrid search (dense + sparse / BM25)
  - [ ] Re-ranking (cross-encoders, Cohere re-rank)
  - [ ] Query transformation (HyDE, multi-query, sub-query)
  - [ ] Parent-child document retrieval
  - [ ] Recursive retrieval
  - [ ] Self-RAG
  - [ ] Corrective RAG (CRAG)
  - [ ] Adaptive RAG
  - [ ] GraphRAG (knowledge graph + RAG)
  - [ ] Multi-modal RAG
  - [ ] Agentic RAG (agents that decide when/how to retrieve)
- [ ] **RAG evaluation**
  - [ ] Faithfulness
  - [ ] Answer relevancy
  - [ ] Context precision / recall
  - [ ] RAGAS framework


🏋️ **Exercises:**
1. Build RAG chatbot: LangChain + ChromaDB + OpenAI embeddings on 10 PDF documents
2. Implement hybrid search (dense + BM25) with re-ranking
3. Implement HyDE (Hypothetical Document Embeddings) query transformation
4. Build GraphRAG: extract entities, build knowledge graph, answer multi-hop questions
5. Evaluate RAG with RAGAS framework (faithfulness, relevancy, precision)

🛠️ **AGENT PROJECTS:**
1. **RAG Knowledge Base** — Upload PDFs, ask questions, get cited answers
2. **Multi-Agent Research Bot** — searcher + summarizer + fact-checker agents
3. **Personal AI Assistant** — long-term memory, tool use, calendar/email integration

#### 6.3.4 Memory for Agents
- [ ] Short-term memory (conversation context)
- [ ] Long-term memory (vector store, knowledge base)
- [ ] Episodic memory
- [ ] Semantic memory
- [ ] Procedural memory
- [ ] Memory management (summarization, forgetting)
- [ ] MemGPT / Letta

#### 6.3.5 Protocols and Standards
- [ ] MCP (Model Context Protocol) — Anthropic
- [ ] A2A (Agent-to-Agent) protocol — Google
- [ ] OpenAI function calling API
- [ ] Tool use standards

#### 6.3.6 Agent Frameworks
- [ ] LangChain
- [ ] LangGraph
- [ ] LlamaIndex
- [ ] AutoGen (Microsoft)
- [ ] CrewAI
- [ ] Semantic Kernel (Microsoft)
- [ ] Haystack (deepset)
- [ ] smolagents (Hugging Face)
- [ ] DSPy (declarative language model programming)
- [ ] Pydantic AI

#### 6.3.7 Prompt Engineering
- [ ] Zero-shot prompting
- [ ] Few-shot prompting (in-context learning)
- [ ] Chain-of-Thought (CoT) prompting
- [ ] Tree-of-Thought prompting
- [ ] Self-consistency (multiple reasoning paths)
- [ ] ReAct prompting
- [ ] Structured output prompting (JSON mode)
- [ ] System prompts
- [ ] Role prompting
- [ ] Prompt chaining
- [ ] Prompt optimization (DSPy, automatic prompt engineering)
- [ ] Jailbreak awareness and defenses

---

### 6.4 Multimodal AI

#### 6.4.1 Vision-Language Models

📚 **Best Resources to Learn:**
- CMU 11-777 Multimodal ML — cmu-mmml.github.io
- Hugging Face multimodal docs
- CLIP paper — arxiv.org/abs/2103.00020

- [ ] **CLIP (Contrastive Language-Image Pre-training)**
  - [ ] Dual encoder (image + text)
  - [ ] Contrastive learning objective
  - [ ] Zero-shot classification
- [ ] **SigLIP** (Sigmoid loss for language-image pre-training)
- [ ] **BLIP / BLIP-2** (Bootstrapping Language-Image Pre-training)
- [ ] **LLaVA** (Large Language and Vision Assistant)
  - [ ] Visual instruction tuning
  - [ ] LLaVA-1.5, LLaVA-NeXT
- [ ] **GPT-4V / GPT-4o** (multimodal GPT)
- [ ] **Gemini** (natively multimodal)
- [ ] **Claude vision** (multimodal Claude)
- [ ] **Qwen-VL**
- [ ] **InternVL**
- [ ] **PaliGemma**
- [ ] **Florence-2**
- [ ] **CogVLM**
- [ ] **Pixtral**


🏋️ **Exercises:**
1. Use CLIP for zero-shot image classification on a custom dataset
2. Build a multimodal search engine: embed images and text, search across modalities
3. Fine-tune LLaVA on a visual instruction dataset

🛠️ **Multimodal PROJECT: Document Q&A** — Upload PDFs with images/tables, ColPali embeddings, natural language queries.

#### 6.4.2 Text-to-Image Generation
- [ ] Stable Diffusion (v1.5, 2.0, XL, 3.0)
- [ ] DALL-E (1, 2, 3)
- [ ] Midjourney
- [ ] Flux
- [ ] Imagen (Google)
- [ ] SDXL Turbo / LCM
- [ ] Playground v2
- [ ] ControlNet (conditional generation)
- [ ] IP-Adapter (image prompt adapter)
- [ ] T2I-Adapter

#### 6.4.3 Text-to-Video
- [ ] Sora (OpenAI)
- [ ] Gen-2, Gen-3 (Runway)
- [ ] Kling (Kuaishou)
- [ ] CogVideo
- [ ] Open-Sora
- [ ] Temporal consistency challenges
- [ ] Video diffusion models

#### 6.4.4 Text-to-Audio / Speech
- [ ] Text-to-Speech (TTS)
  - [ ] Tacotron, WaveNet, VITS
  - [ ] Bark
  - [ ] XTTS / Coqui
- [ ] Voice cloning (VALL-E)
- [ ] Music generation (MusicGen, Suno, Udio)
- [ ] Audio understanding (Whisper for transcription)
- [ ] Speech-to-text (ASR)

#### 6.4.5 Multimodal Embeddings and Retrieval
- [ ] Shared embedding spaces
- [ ] Cross-modal retrieval (text-to-image, image-to-text)
- [ ] ImageBind (Meta) — multiple modalities
- [ ] Multimodal search systems

#### 6.4.6 Document Understanding
- [ ] Document OCR + LLM
- [ ] Layout-aware models (LayoutLM, LayoutLMv3)
- [ ] Table extraction
- [ ] Chart understanding
- [ ] DocVQA (Document Visual Question Answering)
- [ ] ColPali / ColQwen (late-interaction document retrieval)

---

### 6.5 Frontier Research Topics

#### 6.5.1 State Space Models

📚 **Best Resources to Learn:**
- Maarten Grootendorst "Visual Guide to Mamba" — maartengrootendorst.com/blog/mamba
- Mamba paper — arxiv.org/abs/2312.00752
- Albert Gu S4 lectures — youtube.com

- [ ] Linear state space layers (S4)
- [ ] **Mamba** (selective state space model)
  - [ ] Input-dependent selection mechanism
  - [ ] Hardware-aware algorithm
  - [ ] Linear-time sequence modeling
- [ ] Mamba-2
- [ ] Jamba (hybrid Mamba + Transformer)
- [ ] Comparison to Transformers (subquadratic complexity)
- [ ] Hybrid architectures (Mamba + Attention)
- [ ] RWKV (linear attention RNN)
- [ ] xLSTM (extended LSTM)
- [ ] Griffin (recurrent + attention hybrid)


🏋️ **Exercises:**
1. Read and summarize the Mamba paper using 3-pass method
2. Run Mamba vs Transformer on a sequence modeling benchmark, compare speed and quality

#### 6.5.2 Mixture of Experts (MoE)
- [ ] Sparse MoE architecture
- [ ] Router / gating mechanism
  - [ ] Top-k routing
  - [ ] Expert load balancing
- [ ] Expert parallelism
- [ ] MoE training stability
- [ ] Examples: Mixtral, DeepSeek-V2/V3, Switch Transformer, GShard
- [ ] Shared experts vs routed experts
- [ ] Fine-grained MoE

#### 6.5.3 Reasoning and Test-Time Compute
- [ ] Chain-of-thought reasoning
- [ ] Tree-of-thought reasoning
- [ ] Self-consistency decoding
- [ ] o1-style reasoning models (extended thinking)
- [ ] DeepSeek-R1 (RL for reasoning)
- [ ] Process Reward Models (PRM)
- [ ] Outcome Reward Models (ORM)
- [ ] Monte Carlo Tree Search (MCTS) for reasoning
- [ ] Test-time compute scaling (thinking longer = better)
- [ ] Verification and self-correction
- [ ] Math reasoning (chain-of-thought, tool use)
- [ ] Code reasoning (generate-and-test)

#### 6.5.4 Mechanistic Interpretability

📚 **Best Resources to Learn:**
- Anthropic interpretability research blog — transformer-circuits.pub
- Neel Nanda TransformerLens — neelnanda-io.github.io/TransformerLens
- "Toy Models of Superposition" paper — arxiv.org

- [ ] Feature visualization
- [ ] Activation patching
- [ ] Probing classifiers
- [ ] Attention head analysis
- [ ] Sparse autoencoders for feature discovery
- [ ] Circuit analysis
- [ ] Logit lens / tuned lens
- [ ] Representation engineering
- [ ] Superposition hypothesis
- [ ] Polysemanticity


🏋️ **Exercises:**
1. Use TransformerLens to analyze attention patterns in GPT-2
2. Train a sparse autoencoder on GPT-2 activations, find interpretable features
3. Implement activation patching to identify important components for a task

🛠️ **RESEARCH PROJECT: Paper Reproduction** — Choose a paper from NeurIPS/ICML/ICLR (< 1 year old), reproduce key results, write blog post, open-source code.

#### 6.5.5 AI Safety and Alignment

📚 **Best Resources to Learn:**
- AI Safety Fundamentals (BlueDot Impact, FREE course)
- Anthropic research blog
- ARC (Alignment Research Center)
- Nathan Lambert RLHF Book

- [ ] Alignment problem definition
- [ ] Scalable oversight
- [ ] Constitutional AI
- [ ] Debate as alignment method
- [ ] Iterated Distillation and Amplification (IDA)
- [ ] Recursive reward modeling
- [ ] Cooperative AI
- [ ] AI governance and policy
- [ ] Red teaming and adversarial testing
- [ ] Specification gaming / reward hacking
- [ ] Deceptive alignment (overview)
- [ ] Power-seeking (overview)
- [ ] Corrigibility
- [ ] AI Safety Fundamentals course (BlueDot Impact)
- [ ] Anthropic interpretability research
- [ ] ARC (Alignment Research Center)


🏋️ **Exercises:**
1. Complete AI Safety Fundamentals course
2. Red-team a language model: find 10 failure modes
3. Implement a simple Constitutional AI pipeline: model critiques its own outputs

#### 6.5.6 Synthetic Data and Self-Improvement
- [ ] Synthetic data generation for training
- [ ] Self-play for improvement
- [ ] Evol-Instruct (WizardLM)
- [ ] Self-Instruct
- [ ] Constitutional AI as self-improvement
- [ ] Rejection sampling for data quality
- [ ] Data curation with LLMs
- [ ] Textbooks Are All You Need (Phi)

#### 6.5.7 Neurosymbolic AI
- [ ] Combining neural networks with symbolic reasoning
- [ ] Neural program synthesis
- [ ] Differentiable programming
- [ ] LLM + formal verification
- [ ] Knowledge graphs + neural networks

#### 6.5.8 Embodied AI and Robotics
- [ ] Vision-language-action models
- [ ] RT-1, RT-2 (Robotics Transformer)
- [ ] Sim-to-real transfer
- [ ] Foundation models for robotics
- [ ] World models for physical reasoning

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PHASE 7: PRACTICAL SKILLS & CAREER BUILDING
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 7.1 Kaggle and Competitions
- [ ] Kaggle account setup and profile
- [ ] Kaggle Notebooks (free GPU/TPU)
- [ ] Beginner competitions (Titanic, House Prices, Digit Recognizer)
- [ ] Tabular competitions (structured data)
- [ ] Computer vision competitions
- [ ] NLP competitions
- [ ] Time series competitions
- [ ] Understanding competition rubrics and evaluation
- [ ] Feature engineering for competitions
- [ ] Ensemble strategies for competitions
- [ ] Post-competition analysis (reading winning solutions)
- [ ] Kaggle ranking system (Novice → Grandmaster)
- [ ] Other platforms (DrivenData, Zindi, AIcrowd)

### 7.2 Portfolio Building
- [ ] 5–8 polished GitHub repositories
- [ ] Clear README files with problem statement, approach, results
- [ ] Code quality (documentation, modularization, testing)
- [ ] Technical blog writing
  - [ ] Medium, Substack, or personal site
  - [ ] Tutorial posts
  - [ ] Paper reproduction posts
  - [ ] Project write-ups
- [ ] Open-source contributions
  - [ ] Contributing to Hugging Face, PyTorch, scikit-learn
  - [ ] Bug fixes, documentation improvements
  - [ ] Feature implementations
- [ ] Paper reproduction projects
- [ ] Demo applications (Gradio, Streamlit, Hugging Face Spaces)

### 7.3 Research Skills
- [ ] How to read research papers (3-pass method)
- [ ] How to find relevant papers (arXiv, Google Scholar, Semantic Scholar)
- [ ] Paper discussion and study groups
- [ ] Reproducing paper results
- [ ] Writing research papers
- [ ] LaTeX basics
- [ ] Experiment design and ablation studies
- [ ] Statistical significance in experiments

### 7.4 Interview Preparation
- [ ] ML fundamentals interview questions
- [ ] Statistics and probability questions
- [ ] Coding interviews (LeetCode, data structures, algorithms)
- [ ] ML system design interviews
  - [ ] Designing recommendation systems
  - [ ] Designing search engines
  - [ ] Designing fraud detection systems
  - [ ] Designing content moderation systems
  - [ ] Designing ad ranking systems
- [ ] Behavioral interviews
- [ ] Take-home assignments
- [ ] Case studies and business problem framing

### 7.5 Compute Resources
- [ ] Google Colab (free and Pro)
- [ ] Kaggle Notebooks (free GPU)
- [ ] Lightning AI
- [ ] Amazon SageMaker Studio Lab
- [ ] Google TPU Research Cloud
- [ ] Lambda Labs, RunPod, Vast.ai (paid GPU clouds)
- [ ] University/institutional compute access
- [ ] Efficient training strategies for limited compute

### 7.6 Staying Current
- [ ] Daily arXiv monitoring (cs.LG, cs.CL, cs.CV, cs.AI)
- [ ] Hugging Face Daily Papers
- [ ] Papers With Code
- [ ] Twitter/X ML community
- [ ] Newsletters: The Batch, Import AI, TLDR AI, The Rundown
- [ ] YouTube: 3Blue1Brown, Andrej Karpathy, Yannic Kilcher, Two Minute Papers, StatQuest, AI Explained
- [ ] Podcasts: Lex Fridman, Latent Space, TWIML AI, Gradient Dissent, Dwarkesh
- [ ] Blogs: Lil'Log, Jay Alammar, Colah, Sebastian Raschka
- [ ] Reddit: r/MachineLearning, r/LocalLLaMA, r/learnmachinelearning
- [ ] Discord communities: Hugging Face, EleutherAI, fast.ai
- [ ] Conference proceedings: NeurIPS, ICML, ICLR, AAAI, CVPR, ACL, EMNLP

### 7.7 Ethics and Responsible AI
- [ ] Bias in ML models
  - [ ] Data bias (historical, representation, measurement)
  - [ ] Algorithmic bias
  - [ ] Evaluation bias
- [ ] Fairness metrics and definitions
  - [ ] Demographic parity
  - [ ] Equalized odds
  - [ ] Individual fairness
- [ ] Privacy in ML
  - [ ] Differential privacy
  - [ ] Federated learning
  - [ ] Membership inference attacks
  - [ ] Model inversion attacks
- [ ] Transparency and explainability (see SHAP, LIME above)
- [ ] Environmental impact of AI training
- [ ] Copyright and intellectual property
- [ ] Deepfakes and misinformation
- [ ] AI regulation and governance (EU AI Act, etc.)
- [ ] Responsible AI deployment principles

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PROGRESS TRACKER
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Phase | Section | Est. Topics | Completed | % |
|-------|---------|-------------|-----------|---|
| 1 | Python Fundamentals | ~180 | ___/180 | __% |
| 1 | Python for Data Science | ~150 | ___/150 | __% |
| 2 | Linear Algebra | ~100 | ___/100 | __% |
| 2 | Calculus | ~75 | ___/75 | __% |
| 2 | Probability & Statistics | ~200 | ___/200 | __% |
| 3 | Core Machine Learning | ~300 | ___/300 | __% |
| 4 | Deep Learning | ~250 | ___/250 | __% |
| 5A | NLP | ~120 | ___/120 | __% |
| 5B | Computer Vision | ~140 | ___/140 | __% |
| 5C | Reinforcement Learning | ~120 | ___/120 | __% |
| 5D | Generative AI | ~100 | ___/100 | __% |
| 5E | Time Series | ~80 | ___/80 | __% |
| 5F | Recommender Systems | ~70 | ___/70 | __% |
| 5G | Graph Neural Networks | ~60 | ___/60 | __% |
| 5H | MLOps | ~90 | ___/90 | __% |
| 6 | Cutting-Edge AI | ~400 | ___/400 | __% |
| 7 | Career & Practical | ~80 | ___/80 | __% |
| **TOTAL** | | **~2,515** | ___/2515 | __% |

---

> **Note:** This checklist contains approximately **2,500+ individual topics and sub-topics**. Not every item needs to be mastered to the same depth. Items in Phases 1–4 should be thoroughly understood. Specialization tracks (Phase 5) should be deep in your chosen 2–3 tracks and at overview level for others. Phase 6 topics at the frontier change rapidly — focus on understanding the principles rather than memorizing every model name.

> **Last updated:** March 2026
