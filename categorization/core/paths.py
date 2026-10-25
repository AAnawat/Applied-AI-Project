"""Locations of data/ and results/ inside categorization/."""

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXAMS_DIR = DATA_DIR / "exams"
ANSWER_KEYS_DIR = DATA_DIR / "answer_keys"
RESULTS_DIR = ROOT / "results"


def run_dir(name, model=None):
    """A new results/<name>_<model>_<timestamp>/ folder, so earlier runs are kept."""
    parts = [name] + ([model.replace(":", "-").replace("/", "-")] if model else [])
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = RESULTS_DIR / "_".join(parts)
    path.mkdir(parents=True, exist_ok=False)
    return path
