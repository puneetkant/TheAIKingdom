# 1.2.7 Command Line — Shell Basics

Unix shell concepts in Python: `find`/`grep`/`cut`/`head` as generators, pushd/popd, env vars, Makefile task runner.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Shell basics: file ops, glob, subprocess |
| `working_example2.py` | Python-native shell: grep/cut/head pipes, pushd/popd, task runner, process info |
| `working_example.ipynb` | Interactive: find, grep pipeline, pushd/popd, env vars |

## Run

```bash
python working_example.py
python working_example2.py
jupyter lab working_example.ipynb
```

## Shell → Python Translation

| Shell | Python |
|-------|--------|
| `pwd` | `Path.cwd()` |
| `ls` | `Path('.').iterdir()` |
| `find . -name '*.py'` | `Path('.').rglob('*.py')` |
| `grep pattern file` | `[l for l in file.splitlines() if pattern in l]` |
| `wc -l file` | `len(file.read_text().splitlines())` |
| `cut -d, -f2` | `line.split(',')[1]` |
| `head -5` | `itertools.islice(lines, 5)` |
| `pushd dir` | `os.chdir(dir)` |
| `export VAR=val` | `os.environ['VAR'] = val` |
| `make task` | Custom task runner class |

## Pipe Pattern

```python
def grep(lines, pattern):
    for line in lines: 
        if pattern in line: yield line

def cut(lines, field, sep=','):
    for line in lines:
        yield line.strip().split(sep)[field]

# cat file | grep female | cut -f2 | head -5
result = list(itertools.islice(cut(grep(open('data.csv'), 'female'), 1), 5))
```

## Learning Resources
- [The Linux Command Line (free book)](https://linuxcommand.org/tlcl.php)
- [Bash scripting cheatsheet](https://devhints.io/bash)
- [Python os module](https://docs.python.org/3/library/os.html)
- [Python pathlib docs](https://docs.python.org/3/library/pathlib.html)
- [Real Python: Python subprocess](https://realpython.com/python-subprocess/)

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
