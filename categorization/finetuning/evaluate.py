"""Compare frozen base and adapter under identical Transformers/NF4 conditions."""
import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path

from categorization.finetuning.common import (
    DEFAULT_DATA, FEW_SHOT_EXAMPLES, CODES, read_jsonl, sha256, verify_manifest,
    normalize, prompt_hash, training_example,
)
from categorization.finetuning.runtime import cuda_runtime, load_base, prompt_ids
from categorization.core.llm import build_messages


def parse_output(text):
    try:
        value = json.loads(text).get("category")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value in CODES else None


def load_external_csv(path):
    """Read proposed/reviewed labels without treating AI review as human approval."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        label = "reviewed_category" if "reviewed_category" in fields else "category"
        if not {"id", "feedback", label} <= fields:
            raise ValueError("CSV needs id, feedback and reviewed_category (or category).")
        rows, statuses, ids, texts = [], set(), set(), set()
        for line, row in enumerate(reader, 2):
            item_id = (row.get("id") or "").strip()
            text = (row.get("feedback") or "").strip()
            category = (row.get(label) or "").strip()
            if not item_id or not text or category not in CODES:
                raise ValueError(f"CSV row {line}: blank ID/text or invalid category.")
            if item_id in ids or normalize(text) in texts:
                raise ValueError(f"CSV row {line}: duplicate ID or normalized feedback.")
            ids.add(item_id)
            texts.add(normalize(text))
            statuses.add(row.get("review_status") or "unspecified")
            rows.append(training_example({
                "id": item_id, "feedback": text, "category": category,
                "group_id": row.get("group_id") or item_id,
            }))
    if not rows:
        raise ValueError("External CSV is empty.")
    return rows, sorted(statuses)


def evaluate(args):
    external = getattr(args, "input_csv", None)
    metadata = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    if external:
        if args.split == "test" or args.confirm_final_test:
            raise ValueError("External CSV is development-only; do not combine it with final-test flags.")
        if metadata["prompt_sha256"] != prompt_hash():
            raise ValueError("The shared taxonomy/prompt changed since training.")
        rows, statuses = load_external_csv(external)
        data_info = {"input_csv_sha256": sha256(external), "label_review_statuses": statuses}
    else:
        verify_manifest(args.data_dir)
        if metadata["data_manifest_sha256"] != sha256(args.data_dir / "manifest.json"):
            raise ValueError("Adapter and evaluation data do not share the same manifest.")
        data_info = {"data_manifest_sha256": sha256(args.data_dir / "manifest.json")}
    if args.split == "test" and not args.confirm_final_test:
        raise ValueError("Use --confirm-final-test only after freezing model/prompt/hyperparameters.")
    if args.split == "test" and metadata["smoke_only"]:
        raise ValueError("Smoke-test adapters cannot be used for final test claims.")
    if args.output.exists():
        raise ValueError("Output already exists. Keep each evaluation run separate.")
    if not external:
        rows = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    if not rows:
        raise ValueError("Evaluation data is empty.")
    torch, _ = cuda_runtime()
    import pandas as pd
    from peft import PeftModel
    from categorization.core.scoring import score
    model, tokenizer, revision = load_base(metadata["base_revision"])
    model = PeftModel.from_pretrained(model, args.run / "adapter", is_trainable=False)
    model.eval()
    model.config.use_cache = True
    examples = FEW_SHOT_EXAMPLES if args.method == "few-shot" else ()

    def generate(text):
        ids = prompt_ids(build_messages(text, examples), tokenizer)
        if len(ids) + args.max_new_tokens > args.max_context:
            raise ValueError("Evaluation prompt exceeds max-context; no prompt was truncated.")
        tokens = torch.tensor([ids], device=model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=tokens, attention_mask=torch.ones_like(tokens),
                max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return tokenizer.decode(output[0][len(ids):], skip_special_tokens=True).strip(), elapsed

    results, summary = [], {}
    for variant in ("base", "adapter"):
        # Warm each path separately; do not include model loading in warm latency.
        with model.disable_adapter() if variant == "base" else nullcontext():
            generate("ข้อความทดสอบระบบ")
            predictions, seconds = [], []
            for row in rows:
                text = row["messages"][-2]["content"]
                raw, elapsed = generate(text)
                predicted = parse_output(raw)
                predictions.append(predicted)
                seconds.append(elapsed)
                results.append({"variant": variant, "id": row["id"],
                                **({"feedback": text} if external else {}),
                                "answer": row["category"], "predicted": predicted,
                                "seconds": elapsed, "raw_output": raw})
        summary[variant] = score(pd.Series([r["category"] for r in rows]),
                                 pd.Series(predictions), pd.Series(seconds))
    args.output.mkdir(parents=True)
    pd.DataFrame(results).to_csv(args.output / "predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "split": "external_development" if external else args.split,
        "method": args.method, "categories": CODES,
        **data_info,
        "base_revision": revision, "gpu": torch.cuda.get_device_name(0),
        "runtime": "Transformers, same frozen 4-bit NF4 base; LoRA disabled/enabled",
        "generation": {"do_sample": False, "max_context": args.max_context,
                       "max_new_tokens": args.max_new_tokens, "json_schema_enforced": False},
        "latency": "warm per-response wall time; model loading excluded",
        "note": "Do not directly compare these NF4/unconstrained scores with the original Ollama Q4_K_M/schema-constrained run.",
        "metrics": summary,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Training output containing run.json and adapter/")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--input-csv", type=Path, help="External development CSV; no re-splitting or training")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--method", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_context <= args.max_new_tokens:
        parser.error("Invalid generation limits.")
    try:
        evaluate(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
