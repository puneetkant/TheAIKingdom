"""
Working Example: Command Line / Shell Basics (Shell-focused)
Covers common shell patterns, pipes, redirects, environment,
process management — illustrated via Python subprocess.
"""
import subprocess
import sys
import os
import tempfile
from pathlib import Path


def run(cmd, **kwargs):
    """Helper: run shell command, print stdout."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return (r.stdout + r.stderr).strip()


def environment_vars():
    print("=== Environment Variables ===")
    # Reading
    home   = os.environ.get("HOME") or os.environ.get("USERPROFILE", "unknown")
    path_n = len(os.environ.get("PATH","").split(os.pathsep))
    print(f"  HOME / USERPROFILE : {home}")
    print(f"  PATH entries       : {path_n}")
    print(f"  PYTHONPATH         : {os.environ.get('PYTHONPATH', '(not set)')}")

    # Setting and using
    os.environ["MY_TOKEN"] = "secret123"
    print(f"  MY_TOKEN           : {os.environ['MY_TOKEN']}")

    # Passing custom env to subprocess
    env = os.environ.copy()
    env["GREETING"] = "Hello from env"
    out = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ['GREETING'])"],
        capture_output=True, text=True, env=env
    ).stdout.strip()
    print(f"  subprocess env var : {out}")


def pipes_and_redirection():
    print("\n=== Pipes & Redirection (via subprocess) ===")
    tmp = Path(tempfile.mkdtemp())
    data_file = tmp / "numbers.txt"
    data_file.write_text("\n".join(str(i) for i in [5,2,8,1,3,7,4,6]))

    # Equivalent of: cat numbers.txt | sort -n | tail -3
    cat  = subprocess.Popen(["python","-c",
                             f"print(open(r'{data_file}').read(),end='')"],
                            stdout=subprocess.PIPE)
    sort = subprocess.Popen([sys.executable,"-c",
                             "import sys; lines=sys.stdin.read().split(); "
                             "[print(x) for x in sorted(lines, key=int)]"],
                            stdin=cat.stdout, stdout=subprocess.PIPE, text=True)
    cat.stdout.close()
    out, _ = sort.communicate()
    top3 = out.strip().split("\n")[-3:]
    print(f"  top 3 sorted numbers: {top3}")

    # Redirect stdout to file
    with open(tmp / "output.txt", "w") as f:
        subprocess.run([sys.executable, "-c",
                        "for i in range(5): print(f'item {i}')"],
                       stdout=f)
    print(f"  redirected output:\n    {(tmp/'output.txt').read_text().strip()}")

    import shutil; shutil.rmtree(tmp)


def stdin_stdin_demo():
    print("\n=== stdin / stdout Interaction ===")
    # Send data to a script via stdin
    script = (
        "import sys\n"
        "for line in sys.stdin:\n"
        "    print(line.strip().upper())"
    )
    input_data = "hello world\npython is great\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=input_data, capture_output=True, text=True
    )
    print(f"  stdin→uppercase:\n    {result.stdout.strip()}")


def exit_codes():
    print("\n=== Exit Codes ===")
    codes = {
        0:   "success",
        1:   "general error",
        2:   "misuse of shell command",
        126: "command cannot execute",
        127: "command not found",
        130: "terminated by Ctrl+C",
    }
    for code, meaning in codes.items():
        print(f"  exit {code:<4} → {meaning}")

    # Check return code
    r = subprocess.run([sys.executable, "-c", "raise SystemExit(42)"],
                       capture_output=True)
    print(f"\n  script exit 42 → returncode={r.returncode}")


def shell_patterns():
    print("\n=== Common Shell Patterns in Python ===")
    tmp = Path(tempfile.mkdtemp())

    # Create files
    for i in range(5):
        (tmp / f"file_{i}.log").write_text(f"entry {i}\nerror {i}\n")
    (tmp / "data.csv").write_text("a,b\n1,2\n")

    # Glob (find *.log)
    logs = sorted(tmp.glob("*.log"))
    print(f"  glob *.log  : {[f.name for f in logs]}")

    # Word count (lines)
    total_lines = sum(len(f.read_text().splitlines()) for f in logs)
    print(f"  total lines : {total_lines}")

    # grep-style (find 'error' in all log files)
    matches = []
    for f in logs:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "error" in line.lower():
                matches.append(f"{f.name}:{i}: {line}")
    print(f"  grep 'error': {len(matches)} matches")
    for m in matches[:3]:
        print(f"    {m}")

    # Sort filenames
    all_files = sorted(tmp.iterdir(), key=lambda p: p.name)
    print(f"  sorted ls   : {[f.name for f in all_files]}")

    import shutil; shutil.rmtree(tmp)


def shebang_and_script_guide():
    print("\n=== Shebang & Script Packaging ===")
    template = '''\
#!/usr/bin/env python3
"""My CLI script."""
import argparse, sys

def main():
    parser = argparse.ArgumentParser(description="My tool")
    parser.add_argument("name", help="Your name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        print(f"Hello, {args.name}! (verbose mode)")
    else:
        print(f"Hi {args.name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    print("  Recommended Python CLI script template:")
    for line in template.strip().splitlines():
        print(f"    {line}")

    print("\n  Make executable on Unix:")
    print("    chmod +x script.py")
    print("    ./script.py Alice --verbose")


if __name__ == "__main__":
    environment_vars()
    pipes_and_redirection()
    stdin_stdin_demo()
    exit_codes()
    shell_patterns()
    shebang_and_script_guide()
