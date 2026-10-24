import argparse
from pathlib import Path

import pandas as pd

from .model import DEFAULT_MODEL_PATH, load_model, predict_sentiment
from .storage import save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One Thai feedback message")
    source.add_argument("--input", type=Path, help="UTF-8 CSV with a feedback column")
    parser.add_argument("--dry-run", action="store_true", help="Preview without connecting to Supabase")
    parser.add_argument("--batch-name", help="Survey upload name; defaults to the input filename")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    args = parser.parse_args()

    frame = pd.DataFrame({"feedback": [args.text]}) if args.text is not None else pd.read_csv(
        args.input, encoding="utf-8-sig",
    )

    if "feedback" not in frame.columns:
        parser.error("CSV must contain a feedback column.")

    if {"sentiment", "confidence"} & set(frame.columns):
        parser.error("Input already contains sentiment/confidence; rename or remove them first.")

    try:
        result = frame.join(predict_sentiment(frame["feedback"], load_model(args.model)))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(result.to_string(index=False))
        return

    if result.empty:
        parser.error("No feedback rows to save.")
    batch_name = args.batch_name if args.batch_name is not None else (args.input.stem if args.input else "Single feedback")
    print(f"Saving survey: {batch_name}", flush=True)

    try:
        batch_id, count = save_predictions(result, batch_name=batch_name)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Saved {count} responses and sentiment predictions to Supabase (survey batch ID: {batch_id})")


if __name__ == "__main__":
    main()
