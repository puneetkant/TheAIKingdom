from pathlib import Path
import re

root = Path('ML-DL Mastery')

KEYWORD_MAP = [
    (['setup', 'environment'], ('Environment Setup Project', "Check your Python environment and install packages.")),
    (['basic syntax', 'data types'], ('Python Syntax Practice', "Practice variables, strings, lists, and type conversions.")),
    (['operators'], ('Operators Explorer', "Experiment with arithmetic, comparison, logical, and bitwise operators.")),
    (['control flow'], ('Control Flow Exercises', "Build loops and conditionals to solve small problems.")),
    (['data structures'], ('Data Structures Workshop', "Work with lists, tuples, dicts, sets, and comprehensions.")),
    (['functions'], ('Functions and Modules', "Define functions, use args/kwargs, and document them.")),
    (['modules', 'packages'], ('Python Modules Project', "Create and import a custom Python module and package.")),
    (['file i', 'file i/o'], ('File I/O Project', "Read, write, and process text or JSON files.")),
    (['error', 'exception'], ('Exceptions and Error Handling', "Handle runtime errors and use try/except blocks.")),
    (['object-oriented', 'oop'], ('OOP Practice', "Build classes with methods, inheritance, and dunder methods.")),
    (['advanced python'], ('Advanced Python Concepts', "Use iterators, generators, decorators, and context managers.")),
    (['testing'], ('Testing Project', "Write unit tests for your functions with simple asserts.")),
    (['algorithms', 'complexity'], ('Algorithms Practice', "Implement sorting, searching, and complexity analysis.")),
    (['numpy'], ('NumPy Starter', "Use NumPy arrays and operations for numeric computing.")),
    (['pandas'], ('Pandas Data Project', "Load data into DataFrames and perform basic analysis.")),
    (['data visualization'], ('Visualization Project', "Create charts and plots for data exploration.")),
    (['exploratory data analysis', 'eda'], ('EDA Project', "Inspect, clean, and summarize a dataset.")),
    (['sql'], ('SQL Practice', "Use SQLite to create tables and query data.")),
    (['version control', 'git'], ('Git Learning Notes', "Track and manage code changes with Git.")),
    (['command line', 'shell'], ('Command Line Basics', "Write scripts and use command-line tools.")),
    (['web scraping', 'api'], ('Web Scraping & APIs', "Fetch data from the web and parse it.")),
    (['vectors'], ('Vectors Project', "Implement vector operations and visualizations.")),
    (['matrices'], ('Matrices Project', "Work with matrix math and transformations.")),
    (['systems of linear equations'], ('Linear System Solver', "Solve linear equation systems programmatically.")),
    (['determinants'], ('Determinants and Matrix Algebra', "Compute determinants and matrix properties.")),
    (['vector spaces'], ('Vector Space Concepts', "Explore linear combinations and spans.")),
    (['eigenvalues', 'eigenvectors'], ('Eigen Decomposition', "Compute eigenvalues and eigenvectors.")),
    (['svd', 'singular value decomposition'], ('SVD Demo', "Decompose matrices with SVD and inspect components.")),
    (['inner product', 'inner product spaces'], ('Inner Product Project', "Explore dot products and orthogonality.")),
    (['numerical linear algebra'], ('Numerical Linear Algebra', "Use numerical methods in linear algebra.")),
    (['calculus'], ('Calculus Project', "Implement derivatives, integrals, and optimization examples.")),
    (['probability'], ('Probability Project', "Compute probabilities, expectations, and distributions.")),
    (['statistics'], ('Statistics Project', "Analyze sample data and calculate summary statistics.")),
    (['bayesian'], ('Bayesian Statistics', "Work with prior/posterior reasoning.")),
    (['machine learning fundamentals', 'learning theory'], ('ML Fundamentals', "Implement basic ML concepts and theory examples.")),
    (['supervised learning'], ('Supervised Learning Project', "Train simple predictor models on sample data.")),
    (['unsupervised learning'], ('Unsupervised Learning Project', "Cluster or reduce dimensionality of data.")),
    (['data preprocessing', 'feature engineering'], ('Feature Engineering Project', "Clean and transform data for modeling.")),
    (['model evaluation'], ('Model Evaluation', "Calculate metrics and compare model performance.")),
    (['neural network', 'neural networks'], ('Neural Network Basics', "Build and inspect a simple neural network structure.")),
    (['cnn', 'convolutional neural'], ('CNN Project', "Explore convolutional operations for image data.")),
    (['rnn', 'recurrent neural'], ('RNN Project', "Work with sequence data and recurrent models.")),
    (['attention', 'transformers'], ('Transformer Concepts', "Inspect attention and transformer architecture.")),
    (['autoencoder'], ('Autoencoder Demo', "Build an autoencoder-style encoder/decoder flow.")),
    (['pytorch'], ('PyTorch Starter', "Write code using PyTorch tensors and models.")),
    (['tensorflow', 'keras'], ('TensorFlow/Keras Starter', "Build a simple model with Keras APIs.")),
    (['experiment tracking'], ('Experiment Tracking', "Log model training and metrics.")),
    (['natural language processing', 'nlp'], ('NLP Project', "Process text and build simple NLP pipelines.")),
    (['computer vision', 'cv'], ('Computer Vision Project', "Process images and build CV examples.")),
    (['reinforcement learning', 'rl'], ('Reinforcement Learning', "Implement a basic RL agent or environment.")),
    (['generative ai', 'generative'], ('Generative AI Demo', "Create or explore generative model results.")),
    (['time series'], ('Time Series Project', "Forecast or analyze temporal data.")),
    (['recommender'], ('Recommender System', "Build a simple recommendation engine.")),
    (['graph neural', 'gnn'], ('Graph Neural Networks', "Work with graphs and GNN concepts.")),
    (['mlops'], ('MLOps Starter', "Prepare model deployment and monitoring code.")),
    (['large language models', 'llm'], ('LLM Exploration', "Inspect tokenization and sampling.")),
    (['fine-tuning'], ('Fine-Tuning Project', "Adapt a model to a custom dataset.")),
    (['agent'], ('AI Agents', "Design a simple agent architecture.")),
    (['multimodal'], ('Multimodal Project', "Combine text and image data pipelines.")),
    (['state space', 'mamba', 'rwkv'], ('State Space Models', "Explore sequence modeling beyond transformers.")),
    (['mixture of experts', 'moe'], ('Mixture of Experts', "Create a router/expert-style pattern.")),
    (['mechanistic interpretability'], ('Interpretability Project', "Inspect model internals and activations.")),
    (['safety', 'alignment'], ('AI Safety Study', "Explore alignment methods and safe ML practices.")),
    (['synthetic data'], ('Synthetic Data Project', "Generate training examples automatically.")),
    (['neurosymbolic'], ('Neurosymbolic AI', "Mix symbolic reasoning with neural models.")),
    (['embodied', 'robotics'], ('Embodied AI Project', "Work with action, perception, and robotics concepts.")),
    (['kaggle'], ('Kaggle Project', "Create a Kaggle-ready notebook and dataset pipeline.")),
    (['portfolio'], ('Portfolio Project', "Build a polished code project with README and results.")),
    (['research skills'], ('Research Project', "Read papers and reproduce results.")),
    (['interview'], ('Interview Prep', "Practice ML and coding interview questions.")),
    (['compute resources'], ('Compute Resource Guide', "Track compute options and experiment setups.")),
    (['staying current'], ('Learning Plan', "Follow papers, blogs, and community resources.")),
    (['ethics', 'responsible'], ('Responsible AI Study', "Explore bias, fairness, and AI governance.")),
]

DEFAULT_TITLE = 'Learning Project'
DEFAULT_DESCRIPTION = 'Explore this topic with a small practical project or coding exercise.'


def pick_topic_info(name: str):
    key = name.lower()
    for keywords, info in KEYWORD_MAP:
        if any(k in key for k in keywords):
            return info
    return (DEFAULT_TITLE, DEFAULT_DESCRIPTION)


def sanitize_filename(name: str):
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    return name.replace(' ', '_').replace('--', '-').lower()


def make_readme(path: Path, title: str, description: str, rel_name: str):
    content = f"""# {rel_name}

**Project:** {title}

{description}

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
"""
    path.write_text(content, encoding='utf-8')


def make_project_py(path: Path, title: str, rel_name: str):
    body = f'"""Starter code for {rel_name}.\n\nProject: {title}\n"""\n\n'
    if 'numpy' in title.lower() or 'vector' in rel_name.lower() or 'matrix' in rel_name.lower():
        body += "try:\n    import numpy as np\nexcept ImportError:\n    np = None\n\n"
        body += "def example():\n    if np is None:\n        print('Install numpy to run this example.')\n        return\n    a = np.array([1, 2, 3])\n    b = np.array([4, 5, 6])\n    print('a + b =', a + b)\n\n"
    elif 'pandas' in title.lower() or 'dataframe' in rel_name.lower():
        body += "try:\n    import pandas as pd\nexcept ImportError:\n    pd = None\n\n"
        body += "def example():\n    if pd is None:\n        print('Install pandas to run this example.')\n        return\n    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n    print(df)\n\n"
    elif 'visualization' in title.lower() or 'plot' in rel_name.lower():
        body += "try:\n    import matplotlib.pyplot as plt\nexcept ImportError:\n    plt = None\n\n"
        body += "def example():\n    if plt is None:\n        print('Install matplotlib to run this example.')\n        return\n    x = [1, 2, 3, 4]\n    y = [1, 4, 9, 16]\n    plt.plot(x, y, marker='o')\n    plt.title('Sample Plot')\n    plt.xlabel('x')\n    plt.ylabel('y')\n    plt.show()\n\n"
    elif 'sql' in title.lower():
        body += "import sqlite3\n\n"
        body += "def example():\n    conn = sqlite3.connect(':memory:')\n    c = conn.cursor()\n    c.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, value REAL)')\n    c.execute('INSERT INTO items (name, value) VALUES (?, ?)', ('apple', 2.5))\n    conn.commit()\n    for row in c.execute('SELECT * FROM items'):\n        print(row)\n    conn.close()\n\n"
    elif 'web scraping' in title.lower() or 'api' in title.lower():
        body += "def example():\n    print('Use requests and BeautifulSoup or the requests library to fetch APIs.')\n\n"
    elif 'git' in title.lower() or 'command line' in title.lower() or 'shell' in title.lower():
        body += "def example():\n    print('This topic is best practiced with shell commands and Git workflows.')\n\n"
    elif 'object-oriented' in title.lower() or 'oop' in title.lower():
        body += "class ExampleClass:\n    def __init__(self, name):\n        self.name = name\n\n    def greet(self):\n        print(f'Hello, {self.name}!')\n\n\ndef example():\n    obj = ExampleClass('Learner')\n    obj.greet()\n\n"
    elif 'testing' in title.lower():
        body += "def add(a, b):\n    return a + b\n\n\ndef example():\n    assert add(2, 3) == 5\n    print('Basic test passed.')\n\n"
    else:
        body += "def example():\n    print('Implement the starter project for this topic.')\n\n"
    body += "if __name__ == '__main__':\n    example()\n"
    path.write_text(body, encoding='utf-8')


for folder in sorted(root.rglob('*')):
    if not folder.is_dir():
        continue
    if any(child.is_dir() for child in folder.iterdir()):
        continue
    readme_path = folder / 'README.md'
    project_path = folder / 'project.py'
    title, description = pick_topic_info(folder.name)
    rel_name = str(folder.relative_to(root))
    if not readme_path.exists():
        make_readme(readme_path, title, description, rel_name)
    if not project_path.exists():
        make_project_py(project_path, title, rel_name)
print('Generated starter files in', len([p for p in root.rglob('README.md') if p.parent != root]))
