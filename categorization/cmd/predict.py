"""Add a predicted topic category to each row of a feedback CSV.

    python -m categorization.cmd.predict --input feedback.csv --few-shot
"""

import argparse
from pathlib import Path

import pandas as pd

from categorization.core.llm import DEFAULT_HOST, DEFAULT_MODEL, categorize
from categorization.core.paths import run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 CSV with a feedback column")
    parser.add_argument("--output", type=Path,
                        help="CSV to write (default: results/<input>_<model>_<timestamp>/categorized.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--few-shot", action="store_true", help="Include the built-in few-shot examples")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    if "feedback" not in frame.columns:
        parser.error("CSV must contain a feedback column.")
    if {"category", "category_confidence"} & set(frame.columns):
        parser.error("Input already contains category/category_confidence; rename or remove them first.")

    try:
        result = categorize(frame["feedback"], model=args.model, host=args.host,
                            few_shot=args.few_shot, timeout=args.timeout)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    # Invalid outputs stay empty so no category_predictions row is saved for them
    frame["category"] = result["category"]
    output = args.output or run_dir(args.input.stem, args.model) / "categorized.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    invalid = int((~result["category_valid"]).sum())
    print(f"Categorized {len(frame)} responses ({invalid} invalid outputs left blank) -> {output}")


if __name__ == "__main__":
    main()
