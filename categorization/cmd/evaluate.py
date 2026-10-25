"""Score categorizers on an exam against its answer key.

Rows of data/exams/<name>.csv and data/answer_keys/<name>.csv are matched by id.
Results go to results/<name>_<model>_<timestamp>/.

    python -m categorization.cmd.evaluate --exam sample --model qwen3:1.7b
"""

import argparse
import json
from datetime import datetime

import pandas as pd

from categorization.core.exam import few_shot_overlap, load_exam_set
from categorization.core.llm import DEFAULT_HOST, DEFAULT_MODEL, categorize, loaded_model_memory
from categorization.core.paths import run_dir
from categorization.core.scoring import INVALID, score

METHODS = ["zero-shot", "few-shot"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exam", required=True, help="Name shared by data/exams/<name>.csv and data/answer_keys/<name>.csv")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, help="Score only the first N rows (quick checks)")
    args = parser.parse_args()

    try:
        data = load_exam_set(args.exam)
    except ValueError as error:
        parser.error(str(error))
    if args.limit is not None:
        data = data.head(args.limit)

    if "few-shot" in args.methods:
        overlap = few_shot_overlap(data["feedback"])
        if overlap:
            parser.error(f"{len(overlap)} exam rows are few-shot examples; remove them first.")

    out_dir = run_dir(args.exam, args.model)
    predictions = data.copy()
    summary = {
        "exam": args.exam,
        "model": args.model,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "methods": {},
    }

    for method in args.methods:
        print(f"Running {method} on {len(data)} responses...", flush=True)
        try:
            result = categorize(data["feedback"], model=args.model, host=args.host,
                                few_shot=method == "few-shot", timeout=args.timeout)
        except (ValueError, RuntimeError) as error:
            parser.error(f"{method}: {error}")

        metrics = score(data["category"], result["category"], result["category_seconds"])
        metrics["ollama_memory"] = loaded_model_memory(args.model, args.host)
        summary["methods"][method] = metrics

        predictions[f"{method}_category"] = result["category"].fillna(INVALID)
        predictions[f"{method}_raw"] = result["raw_output"]
        pd.crosstab(data["category"], predictions[f"{method}_category"],
                    rownames=["answer"], colnames=["predicted"]).to_csv(
            out_dir / f"confusion_{method}.csv", encoding="utf-8-sig")

    predictions.rename(columns={"category": "answer"}).to_csv(
        out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    table = pd.DataFrame({
        method: {key: metrics[key] for key in
                 ("accuracy", "macro_f1", "weighted_f1", "invalid_rate", "mean_seconds", "p95_seconds")}
        for method, metrics in summary["methods"].items()
    }).T
    print(table.round(3).to_string())
    print(f"Saved predictions, confusion matrices and metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
