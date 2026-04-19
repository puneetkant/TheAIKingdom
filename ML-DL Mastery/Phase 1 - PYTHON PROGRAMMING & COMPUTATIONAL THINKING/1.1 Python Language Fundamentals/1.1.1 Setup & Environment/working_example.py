"""
Working Example: Setup & Environment
Demonstrates Python version check, package inspection, venv usage guidance,
pip management, environment introspection, dependency auditing, and
modern tooling (pyproject.toml, uv, conda).
"""
import sys
import platform
import subprocess
import os
import importlib
import json
import hashlib


def check_python_version():
    print("=== Python Version ===")
    print(f"Version      : {sys.version}")
    print(f"Version Info : {sys.version_info}")
    print(f"Platform     : {platform.platform()}")
    print(f"Executable   : {sys.executable}")
    required = (3, 10)
    if sys.version_info >= required:
        print(f"[OK] Python {required[0]}.{required[1]}+ requirement met")
    else:
        print(f"[X] Please upgrade to Python {required[0]}.{required[1]}+")


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
        print(f"  [OK] Running inside a virtual environment: {sys.prefix}")
    else:
        print("  [X] Not inside a virtual environment.")
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


def check_key_ml_packages():
    """Check that essential ML packages are available and show their versions."""
    print("\n=== Key ML Package Availability ===")
    packages = [
        ("numpy",      "np"),
        ("pandas",     "pd"),
        ("matplotlib", "matplotlib"),
        ("sklearn",    "sklearn"),
        ("scipy",      "scipy"),
        ("torch",      "torch"),
        ("tensorflow", "tensorflow"),
        ("transformers","transformers"),
        ("datasets",   "datasets"),
    ]
    print(f"  {'Package':<16} {'Status':<10} {'Version'}")
    print(f"  {'-'*50}")
    for pkg, _ in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  {pkg:<16} {'[OK] installed':<10} {version}")
        except ImportError:
            print(f"  {pkg:<16} {'[X] missing':<10} run: pip install {pkg}")


def detect_gpu():
    """Detect available GPU hardware."""
    print("\n=== GPU Detection ===")
    # PyTorch
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"  PyTorch CUDA available : {cuda_avail}")
        if cuda_avail:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name}  {props.total_memory // 1024**2} MB")
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            print("  Apple MPS (Metal) available")
    except ImportError:
        print("  PyTorch not installed — cannot check CUDA")
    # TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  TensorFlow GPUs        : {len(gpus)}")
        for g in gpus:
            print(f"    {g.name}")
    except ImportError:
        print("  TensorFlow not installed — cannot check TF GPUs")


def pyproject_toml_example():
    """Show a modern pyproject.toml template."""
    print("\n=== Modern Project Setup: pyproject.toml ===")
    template = '''
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.0",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "torch>=2.2",
    "transformers>=4.40",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "pre-commit"]
notebooks = ["jupyterlab", "ipywidgets"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]
'''
    print(template)


def uv_workflow():
    """Demonstrate the modern uv package manager workflow."""
    print("\n=== Modern Package Manager: uv ===")
    print("  uv is a fast Python package/project manager (written in Rust)")
    print()
    steps = [
        ("Install uv",        "pip install uv          # or: curl -Lsf https://astral.sh/uv/install.sh | sh"),
        ("Create project",    "uv init my-ml-project   # creates pyproject.toml, .python-version"),
        ("Add dependency",    "uv add numpy pandas scikit-learn torch"),
        ("Run script",        "uv run python train.py  # auto-syncs venv first"),
        ("Sync from lock",    "uv sync                 # deterministic install from uv.lock"),
        ("Virtual env",       "uv venv && source .venv/bin/activate"),
        ("Run jupyter",       "uv run jupyter lab"),
    ]
    for step, cmd in steps:
        print(f"  {step:<20}  {cmd}")


def conda_workflow():
    """Show conda environment workflow for data science."""
    print("\n=== Conda Environment Workflow ===")
    conda_yaml = '''
# environment.yml
name: ml-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - pytorch::pytorch
  - pytorch::torchvision
  - pip:
    - transformers
    - datasets
    - wandb
'''
    print(conda_yaml)
    print("  Commands:")
    cmds = [
        "conda env create -f environment.yml",
        "conda activate ml-env",
        "conda env export > environment.yml",
        "conda env update -f environment.yml --prune",
    ]
    for c in cmds:
        print(f"  $ {c}")


def environment_fingerprint():
    """Create a reproducibility fingerprint of the current environment."""
    print("\n=== Environment Reproducibility Fingerprint ===")
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    # Try to capture package list
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=15
        )
        pkgs = json.loads(result.stdout)
        info["package_count"] = len(pkgs)
        # Hash the package list for reproducibility checks
        pkg_str = json.dumps(sorted([(p["name"], p["version"]) for p in pkgs]))
        info["package_hash_sha256"] = hashlib.sha256(pkg_str.encode()).hexdigest()[:16]
    except Exception as e:
        info["package_error"] = str(e)

    for k, v in info.items():
        print(f"  {k:<24} {v}")
    print()
    print("  Tip: store this fingerprint alongside model checkpoints for reproducibility.")


if __name__ == "__main__":
    check_python_version()
    show_sys_path()
    list_installed_packages()
    show_environment_variables()
    venv_guidance()
    requirements_example()
    check_key_ml_packages()
    detect_gpu()
    pyproject_toml_example()
    uv_workflow()
    conda_workflow()
    environment_fingerprint()
