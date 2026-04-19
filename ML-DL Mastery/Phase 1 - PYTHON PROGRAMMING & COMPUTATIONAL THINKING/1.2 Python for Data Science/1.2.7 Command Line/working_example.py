"""
Working Example: Command Line / Shell Basics
Covers subprocess, os, pathlib, argparse, shutil,
environment variables, and practical CLI scripting patterns.
"""
import subprocess
import os
import sys
import shutil
import tempfile
import argparse
from pathlib import Path


# -- 1. Running shell commands with subprocess ---------------------------------
def subprocess_demo():
    print("=== 1. subprocess — Running Shell Commands ===")

    # run() — simple, recommended
    result = subprocess.run(
        ["python", "--version"],
        capture_output=True, text=True
    )
    print(f"  python --version: {result.stdout.strip() or result.stderr.strip()}")

    # shell=True (platform-specific, avoid in production security-critical code)
    result = subprocess.run("echo Hello from shell", shell=True,
                            capture_output=True, text=True)
    print(f"  echo: {result.stdout.strip()}")

    # Capture and process output
    result = subprocess.run(
        [sys.executable, "-c", "for i in range(5): print(i**2)"],
        capture_output=True, text=True
    )
    squares = [int(x) for x in result.stdout.split()]
    print(f"  squares via subprocess: {squares}")

    # check=True raises CalledProcessError on non-zero exit
    try:
        subprocess.run(["python", "-c", "exit(1)"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  CalledProcessError caught: returncode={e.returncode}")

    # Popen for streaming output
    print("  streaming output from Popen:")
    with subprocess.Popen(
        [sys.executable, "-c",
         "import time\nfor i in range(3):\n  print(f'line {i}',flush=True)"],
        stdout=subprocess.PIPE, text=True
    ) as proc:
        for line in proc.stdout:
            print(f"    got: {line.rstrip()}")


# -- 2. os module — environment and process ------------------------------------
def os_demo():
    print("\n=== 2. os Module ===")
    print(f"  cwd           : {os.getcwd()}")
    print(f"  os.name       : {os.name}")
    print(f"  cpu count     : {os.cpu_count()}")
    print(f"  pid           : {os.getpid()}")

    # Environment variables
    os.environ["MY_APP_ENV"] = "production"
    print(f"  MY_APP_ENV    : {os.environ.get('MY_APP_ENV', 'not set')}")
    path_val = os.environ.get("PATH", "")
    print(f"  PATH entries  : {len(path_val.split(os.pathsep))}")

    # os.walk — traverse directory tree
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "sub").mkdir()
    (Path(tmp) / "a.txt").write_text("a")
    (Path(tmp) / "sub" / "b.txt").write_text("b")
    print(f"\n  os.walk tree ({tmp}):")
    for dirpath, dirnames, filenames in os.walk(tmp):
        rel = os.path.relpath(dirpath, tmp) or "."
        print(f"    {rel}/  dirs={dirnames}  files={filenames}")
    shutil.rmtree(tmp)


# -- 3. pathlib ----------------------------------------------------------------
def pathlib_demo():
    print("\n=== 3. pathlib ===")
    p = Path.cwd()
    print(f"  cwd           : {p}")
    print(f"  home          : {Path.home()}")

    example = Path("/usr/local/lib/python3.12/site-packages/numpy")
    print(f"  parts         : {example.parts[-4:]}")
    print(f"  stem          : {example.stem}")
    print(f"  suffix        : {example.suffix}")

    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / "raw").mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "test.csv", "val.csv"]:
        (tmp / "data" / "raw" / name).write_text("col1,col2\n1,2")

    print(f"\n  glob *.csv:")
    for f in sorted((tmp / "data").rglob("*.csv")):
        print(f"    {f.relative_to(tmp)}")

    shutil.rmtree(tmp)


# -- 4. shutil — file & directory operations -----------------------------------
def shutil_demo():
    print("\n=== 4. shutil ===")
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "source.txt"
    src.write_text("Important data.\n")

    # copy
    dst = tmp / "backup.txt"
    shutil.copy2(str(src), str(dst))
    print(f"  copy2 -> {dst.name} exists: {dst.exists()}")

    # copytree
    src_dir = tmp / "src_dir"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("hello")
    dst_dir = tmp / "dst_dir"
    shutil.copytree(str(src_dir), str(dst_dir))
    print(f"  copytree files: {[f.name for f in dst_dir.iterdir()]}")

    # move
    shutil.move(str(dst), str(tmp / "moved.txt"))
    print(f"  moved backup.txt -> moved.txt: {(tmp / 'moved.txt').exists()}")

    # disk usage
    usage = shutil.disk_usage(tmp)
    print(f"  disk_usage: total={usage.total//1e9:.0f}GB  free={usage.free//1e9:.0f}GB")

    # which
    python_path = shutil.which("python")
    print(f"  which('python'): {python_path}")

    shutil.rmtree(tmp)


# -- 5. argparse — building CLI tools -----------------------------------------
def argparse_demo():
    print("\n=== 5. argparse (CLI argument parsing) ===")
    parser = argparse.ArgumentParser(
        prog="data_tool",
        description="Example data processing CLI"
    )
    parser.add_argument("input_file",      help="Path to input CSV")
    parser.add_argument("-o", "--output",  default="output.csv",
                        help="Output file (default: output.csv)")
    parser.add_argument("-n", "--rows",    type=int, default=100,
                        help="Number of rows to process")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--format",        choices=["csv","json","parquet"],
                        default="csv", help="Output format")

    # Simulate parsing (don't actually parse sys.argv in demo)
    args = parser.parse_args(["data.csv", "-n", "50", "--verbose", "--format", "json"])
    print(f"  input_file : {args.input_file}")
    print(f"  output     : {args.output}")
    print(f"  rows       : {args.rows}")
    print(f"  verbose    : {args.verbose}")
    print(f"  format     : {args.format}")

    print("\n  Help text:")
    parser.print_help()


# -- 6. Useful shell commands mapped to Python ---------------------------------
def command_mapping():
    print("\n=== 6. Shell -> Python Equivalents ===")
    mapping = [
        ("ls / dir",         "Path('.').iterdir()  or  os.listdir('.')"),
        ("mkdir -p",         "Path('a/b/c').mkdir(parents=True)"),
        ("cp",               "shutil.copy2(src, dst)"),
        ("mv",               "shutil.move(src, dst)"),
        ("rm -rf",           "shutil.rmtree(path)"),
        ("cat file",         "Path('file').read_text()"),
        ("grep pattern",     "re.findall(pattern, text)"),
        ("echo $VAR",        "os.environ.get('VAR')"),
        ("which cmd",        "shutil.which('cmd')"),
        ("pwd",              "Path.cwd()  or  os.getcwd()"),
        ("find . -name",     "Path('.').rglob('pattern')"),
        ("wc -l",            "len(Path('f').read_text().splitlines())"),
        ("curl / wget",      "urllib.request.urlretrieve(url, dst)"),
        ("python script.py", "subprocess.run(['python','script.py'])"),
    ]
    for shell, python in mapping:
        print(f"  {shell:<25} <->  {python}")


if __name__ == "__main__":
    subprocess_demo()
    os_demo()
    pathlib_demo()
    shutil_demo()
    argparse_demo()
    command_mapping()
