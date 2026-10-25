"""Validate NEW human-labelled CSV and create grouped train/validation/test splits."""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from categorization.finetuning.common import (
    CODES, DEFAULT_DATA, FEW_SHOT_EXAMPLES, ROOT, normalize, prompt_hash, sha256, training_example,
)


def protected_feedback():
    exam = ROOT / "data" / "exams" / "climatelens_105.csv"
    with exam.open(encoding="utf-8-sig", newline="") as stream:
        protected = {normalize(row["feedback"]) for row in csv.DictReader(stream)}
    return protected | {normalize(text) for text, _ in FEW_SHOT_EXAMPLES}


def load_labelled(path, protected=None):
    protected = protected_feedback() if protected is None else protected
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "feedback", "category", "group_id"} <= set(reader.fieldnames or []):
            raise ValueError("CSV needs id,feedback,category,group_id. Use the blank template.")
        review_fields = {"review_status", "reviewed_category"}
        review_mode = bool(review_fields & set(reader.fieldnames or []))
        if review_mode and not review_fields <= set(reader.fieldnames or []):
            raise ValueError("Review CSV needs both review_status and reviewed_category.")
        rows = []
        for number, source_row in enumerate(reader, 2):
            row = {key: (source_row.get(key) or "").strip()
                   for key in ("id", "feedback", "category", "group_id")}
            if review_mode:
                status = (source_row.get("review_status") or "").strip().lower()
                reviewed = (source_row.get("reviewed_category") or "").strip()
                if status != "approved" or not reviewed:
                    raise ValueError(
                        f"CSV row {number}: human review is not complete. Set reviewed_category "
                        "and review_status=approved only after a person checks it."
                    )
                row["category"] = reviewed
            rows.append(row)
    if not rows:
        raise ValueError("No labelled rows. The template is intentionally empty; add reviewed data.")
    seen_ids, seen_text = set(), set()
    for number, row in enumerate(rows, 2):
        if not all(row.values()):
            raise ValueError(f"CSV row {number}: all four fields must be non-empty.")
        if row["category"] not in CODES:
            raise ValueError(f"CSV row {number}: category must be one of {CODES}.")
        text = normalize(row["feedback"])
        if row["id"] in seen_ids or text in seen_text:
            raise ValueError(f"CSV row {number}: duplicate ID or normalized feedback.")
        if text in protected:
            raise ValueError(f"CSV row {number}: overlaps development exam/few-shot examples.")
        seen_ids.add(row["id"])
        seen_text.add(text)
    return rows


def split_rows(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_category = defaultdict(list)
    for group, members in groups.items():
        labels = {row["category"] for row in members}
        if len(labels) != 1:
            raise ValueError(f"Group {group}: conflicting labels. Review this sentence family first.")
        by_category[next(iter(labels))].append(group)
    missing = set(CODES) - set(by_category)
    if missing:
        raise ValueError(f"Add reviewed examples for missing categories: {sorted(missing)}.")
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for category in CODES:
        ids = sorted(by_category[category])
        if len(ids) < 3:
            raise ValueError(f"{category}: at least 3 independent groups needed to split. This is a technical minimum, not evidence of sufficient training data.")
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * .1))
        n_val = max(1, round(len(ids) * .1))
        assignments = {"test": ids[:n_test], "validation": ids[n_test:n_test+n_val],
                       "train": ids[n_test+n_val:]}
        for name, selected in assignments.items():
            for group in selected:
                splits[name].extend(groups[group])
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def prepare(source, output, seed=42):
    source, output = Path(source), Path(output)
    rows = load_labelled(source)
    splits = split_rows(rows, seed)
    # Never silently replace a held-out split.
    output.mkdir(parents=True, exist_ok=False)
    for name, members in splits.items():
        payload = "".join(json.dumps(training_example(row), ensure_ascii=False) + "\n" for row in members)
        (output / f"{name}.jsonl").write_text(payload, encoding="utf-8")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    origins = sorted({(row.get("source") or "unspecified").strip() for row in source_rows})
    manifest = {
        "data_origin": origins,
        "source_sha256": sha256(source), "seed": seed, "prompt_sha256": prompt_hash(),
        "categories": CODES,
        "split_strategy": "grouped stratified approximately 80/10/10; at least one group/class/split",
        "counts": {name: dict(Counter(row["category"] for row in members)) for name, members in splits.items()},
        "file_sha256": {f"{name}.jsonl": sha256(output / f"{name}.jsonl") for name in splits},
        "warnings": [
            "Exact normalized overlap is blocked; human reviewers must assign the same group_id to paraphrases/template variants.",
            "Small split sizes do not establish generalization; expand reviewed data before drawing conclusions.",
            "climatelens_105 is development data and is not copied into these splits.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(args.input, args.output, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
