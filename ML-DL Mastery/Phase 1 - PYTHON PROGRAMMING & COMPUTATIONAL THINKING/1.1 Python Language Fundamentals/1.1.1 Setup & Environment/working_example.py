"""
Working Example: Setup & Environment
Demonstrates Python version check, package inspection, venv usage guidance,
pip management, and environment introspection.
"""
import sys
import platform
import subprocess
import os


def check_python_version():
    print("=== Python Version ===")
    print(f"Version      : {sys.version}")
    print(f"Version Info : {sys.version_info}")
    print(f"Platform     : {platform.platform()}")
    print(f"Executable   : {sys.executable}")
    required = (3, 10)
    if sys.version_info >= required:
        print(f"✓ Python {required[0]}.{required[1]}+ requirement met")
    else:
        print(f"✗ Please upgrade to Python {required[0]}.{required[1]}+")


def show_sys_path():
    print("\n=== sys.path (module search paths) ===")
    for i, p in enumerate(sys.path):
        print(f"  [{i}] {p}")


def list_installed_packages():
    print("\n=== Installed Packages (pip list) ===")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().splitlines()
        # Show header + first 15 packages
        for line in lines[:17]:
            print(" ", line)
        if len(lines) > 17:
            print(f"  ... and {len(lines) - 17} more packages")
    except Exception as e:
        print(f"  Could not list packages: {e}")


def show_environment_variables():
    print("\n=== Relevant Environment Variables ===")
    keys = ["PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PATH"]
    for key in keys:
        val = os.environ.get(key, "(not set)")
        if key == "PATH":
            # Only show first entry of PATH
            val = val.split(os.pathsep)[0] + " ..."
        print(f"  {key}: {val}")


def venv_guidance():
    print("\n=== Virtual Environment Guidance ===")
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print(f"  ✓ Running inside a virtual environment: {sys.prefix}")
    else:
        print("  ✗ Not inside a virtual environment.")
        print("  To create one:")
        print("    python -m venv .venv")
        print("  To activate (Windows):")
        print("    .venv\\Scripts\\activate")
        print("  To activate (Linux/macOS):")
        print("    source .venv/bin/activate")


def requirements_example():
    print("\n=== requirements.txt Workflow ===")
    print("  # Freeze current packages:")
    print("  pip freeze > requirements.txt")
    print()
    print("  # Install from requirements.txt:")
    print("  pip install -r requirements.txt")
    print()
    print("  # Install specific package:")
    print("  pip install numpy==1.26.0")
    print()
    print("  # Upgrade a package:")
    print("  pip install --upgrade numpy")


if __name__ == "__main__":
    check_python_version()
    show_sys_path()
    list_installed_packages()
    show_environment_variables()
    venv_guidance()
    requirements_example()
