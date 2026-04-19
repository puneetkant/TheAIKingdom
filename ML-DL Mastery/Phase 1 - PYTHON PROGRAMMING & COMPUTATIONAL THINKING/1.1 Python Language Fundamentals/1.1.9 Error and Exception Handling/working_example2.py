"""
Working Example 2: Error and Exception Handling — Production Patterns
=====================================================================
Demonstrates robust error handling used in real ML pipelines:
  - Custom exception hierarchy for a data pipeline
  - Context managers for resource safety (with statement)
  - Retry decorator with exponential back-off for network calls
  - Graceful degradation: try HF download, fallback to synthetic data
  - logging module instead of bare print for errors
  - Exception groups (Python 3.11+) awareness

Run:  python working_example2.py
"""
import csv
import json
import logging
import time
import urllib.request
import urllib.error
import functools
import random
from pathlib import Path
from typing import Any, Callable

# -- Configure logging (not bare print) ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_pipeline")

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)


# -- 1. Custom Exception Hierarchy ---------------------------------------------
class PipelineError(Exception):
    """Base class for all data-pipeline errors."""

class DataDownloadError(PipelineError):
    """Raised when a dataset cannot be fetched."""
    def __init__(self, url: str, cause: Exception):
        super().__init__(f"Failed to download {url}: {cause}")
        self.url   = url
        self.cause = cause

class DataValidationError(PipelineError):
    """Raised when loaded data doesn't meet schema requirements."""
    def __init__(self, field: str, msg: str):
        super().__init__(f"Validation failed for '{field}': {msg}")
        self.field = field

class ModelTrainingError(PipelineError):
    """Raised when model training encounters a numerical issue."""


# -- 2. Retry decorator with exponential back-off ------------------------------
def retry(max_attempts: int = 3, base_delay: float = 1.0, exceptions=(Exception,)):
    """Decorator: retry a function up to max_attempts with exponential back-off."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s…"
                    )
                    time.sleep(min(delay, 0.5))   # cap to 0.5s in demo
            raise last_exc
        return wrapper
    return decorator


# -- 3. Data download with graceful fallback -----------------------------------
@retry(max_attempts=2, base_delay=0.5, exceptions=(urllib.error.URLError, OSError))
def _attempt_download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)


def download_dataset(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.info(f"Using cached {dest.name}")
        return dest
    try:
        logger.info(f"Downloading {url} …")
        _attempt_download(url, dest)
        logger.info(f"Saved to {dest}")
    except Exception as exc:
        logger.warning(f"Download failed; generating synthetic fallback. ({exc})")
        dest.write_text(
            "PassengerId,Survived,Pclass,Name,Sex,Age,Fare\n"
            + "\n".join(
                f"{i},{i%2},{(i%3)+1},Person {i},{'male' if i%2 else 'female'},{20+i%40},{10+i%100}"
                for i in range(1, 51)
            )
        )
    return dest


# -- 4. Data validation with custom exceptions ---------------------------------
REQUIRED_COLUMNS = {"PassengerId", "Survived", "Pclass", "Age", "Fare"}

def validate_row(row: dict, row_num: int) -> dict:
    """Validate and coerce one CSV row; raise DataValidationError if invalid."""
    for col in REQUIRED_COLUMNS:
        if col not in row:
            raise DataValidationError(col, f"Missing column in row {row_num}")

    for numeric in ("Age", "Fare"):
        val = row.get(numeric, "").strip()
        if val and not val.replace(".", "", 1).replace("-", "", 1).isdigit():
            raise DataValidationError(numeric, f"Non-numeric value '{val}' in row {row_num}")

    return {
        "id":       int(row["PassengerId"]),
        "survived": int(row.get("Survived") or 0),
        "class":    int(row.get("Pclass")   or 0),
        "age":      float(row.get("Age")    or 0),
        "fare":     float(row.get("Fare")   or 0),
    }


def load_and_validate(path: Path) -> tuple[list[dict], list[str]]:
    records = []
    errors  = []
    with open(path, newline="", encoding="utf-8") as f:
        for row_num, row in enumerate(csv.DictReader(f), start=1):
            try:
                records.append(validate_row(row, row_num))
            except DataValidationError as e:
                errors.append(str(e))
    return records, errors


def demo_validation(path: Path) -> list[dict]:
    print("\n=== Data Validation with Custom Exceptions ===")
    records, errors = load_and_validate(path)
    print(f"  Valid rows  : {len(records)}")
    print(f"  Error count : {len(errors)}")
    if errors[:3]:
        print(f"  First errors: {errors[:3]}")
    return records


# -- 5. Context manager for pipeline resources ----------------------------------
class PipelineSession:
    """Context manager: tracks pipeline state, ensures cleanup on error."""
    def __init__(self, name: str):
        self.name    = name
        self.start   = None
        self.metrics: dict[str, Any] = {}

    def __enter__(self):
        self.start = time.perf_counter()
        logger.info(f"Pipeline '{self.name}' started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        if exc_type:
            logger.error(f"Pipeline '{self.name}' FAILED after {elapsed:.2f}s: {exc_val}")
            # Save error report even if pipeline failed
            (DATA / "error_report.json").write_text(
                json.dumps({"name": self.name, "error": str(exc_val), "elapsed": elapsed})
            )
            return False   # re-raise the exception
        logger.info(f"Pipeline '{self.name}' completed in {elapsed:.2f}s. Metrics: {self.metrics}")
        return False       # don't suppress normal flow


def demo_context_manager(records: list[dict]) -> None:
    print("\n=== Context Manager: PipelineSession ===")
    with PipelineSession("titanic_analysis") as session:
        survived    = [r for r in records if r["survived"] == 1]
        session.metrics["total"]        = len(records)
        session.metrics["survived"]     = len(survived)
        session.metrics["survival_pct"] = round(100 * len(survived) / len(records), 1)
        session.metrics["avg_fare"]     = round(sum(r["fare"] for r in records) / len(records), 2)

    # Demo: pipeline with error
    print("\n  Demo: pipeline that raises inside with-block …")
    try:
        with PipelineSession("failing_pipeline") as session:
            session.metrics["step"] = "feature_engineering"
            raise ModelTrainingError("NaN detected in gradient — check learning rate")
    except PipelineError as e:
        print(f"  Caught PipelineError: {e}")


if __name__ == "__main__":
    url  = "https://huggingface.co/datasets/phihung/titanic/resolve/main/train.csv"
    path = download_dataset(url, DATA / "titanic.csv")
    records = demo_validation(path)
    demo_context_manager(records)
