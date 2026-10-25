"""Load an exam and its answer key, and pair their rows by id.

    data/exams/<name>.csv        id, feedback   (what the model sees)
    data/answer_keys/<name>.csv  id, category   (human-assigned codes)
"""

import pandas as pd

from categorization.core.examples import FEW_SHOT_EXAMPLES
from categorization.core.paths import ANSWER_KEYS_DIR, EXAMS_DIR
from categorization.core.taxonomy import CODES


def _read(path, columns):
    if not path.exists():
        raise ValueError(f"Not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {', '.join(sorted(missing))}")

    frame = frame[columns].apply(lambda col: col.str.strip())
    if frame.isna().any().any() or frame.eq("").any().any():
        raise ValueError(f"{path.name}: every row needs {' and '.join(columns)}.")
    duplicates = frame["id"][frame["id"].duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"{path.name}: duplicate ids {', '.join(duplicates[:5])}")
    return frame


def load_exam(path):
    exam = _read(path, ["id", "feedback"])
    duplicates = exam["feedback"].duplicated().sum()
    if duplicates:
        raise ValueError(f"{path.name}: {duplicates} duplicate feedback texts; deduplicate before evaluation.")
    return exam


def load_answer_key(path):
    key = _read(path, ["id", "category"])
    unknown = sorted(set(key["category"]) - set(CODES))
    if unknown:
        raise ValueError(f"{path.name}: unknown category codes {', '.join(unknown)}. Use: {', '.join(CODES)}")
    return key


def join(exam, key):
    """Pair each exam row with its answer by id. Every id must appear in both files."""
    no_answer = sorted(set(exam["id"]) - set(key["id"]))
    no_question = sorted(set(key["id"]) - set(exam["id"]))
    if no_answer or no_question:
        problems = []
        if no_answer:
            problems.append(f"ids with no answer: {', '.join(no_answer[:5])}")
        if no_question:
            problems.append(f"answers with no exam row: {', '.join(no_question[:5])}")
        raise ValueError("Exam and answer key do not match (" + "; ".join(problems) + ").")
    return exam.merge(key, on="id", validate="one_to_one")


def load_exam_set(name):
    """The joined id, feedback, category rows for data/exams/<name>.csv and data/answer_keys/<name>.csv."""
    return join(load_exam(EXAMS_DIR / f"{name}.csv"), load_answer_key(ANSWER_KEYS_DIR / f"{name}.csv"))


def few_shot_overlap(feedback):
    examples = {text.strip() for text, _ in FEW_SHOT_EXAMPLES}
    return sorted(set(feedback) & examples)
