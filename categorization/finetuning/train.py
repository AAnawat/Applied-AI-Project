"""Train a Typhoon QLoRA adapter on NEW reviewed data, separate from the web app."""
import argparse
import json
import platform
from pathlib import Path

from categorization.finetuning.common import BASE_MODEL, DEFAULT_DATA, read_jsonl, sha256, verify_manifest
from categorization.finetuning.runtime import cuda_runtime, encode_example, load_base


def train(args):
    manifest = verify_manifest(args.data_dir)
    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    val_rows = read_jsonl(args.data_dir / "validation.jsonl")
    if not train_rows or not val_rows:
        raise ValueError("Train and validation must both be non-empty.")
    if args.output.exists():
        raise ValueError("Output already exists. Use a new run directory.")
    torch, dtype = cuda_runtime()
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    model, tokenizer, revision = load_base(args.revision)
    train_data = [encode_example(row, tokenizer, args.max_length) for row in train_rows]
    val_data = [encode_example(row, tokenizer, args.max_length) for row in val_rows]
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    model.print_trainable_parameters()
    args.output.mkdir(parents=True)
    metadata = {
        "base_model": BASE_MODEL, "base_revision": revision,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
        "prompt_sha256": manifest["prompt_sha256"],
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training": "4-bit NF4 frozen base + LoRA; assistant JSON and EOS loss only",
        "smoke_only": args.max_steps > 0,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.learning_rate, per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            optim="adamw_torch", warmup_ratio=.05, weight_decay=.01,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=2,
            logging_steps=1 if args.max_steps > 0 else 10,
            report_to="none", seed=args.seed, data_seed=args.seed,
        ),
        train_dataset=train_data, eval_dataset=val_data, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=None, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "validation_loss.json").write_text(
        json.dumps(trainer.evaluate(), indent=2), encoding="utf-8"
    )
    (args.output / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    print(f"Saved adapter to {args.output / 'adapter'}; no held-out test evaluation was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", help="Optional exact Hugging Face revision; resolved SHA is recorded")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value = smoke test only")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.learning_rate < 1:
        parser.error("learning-rate must be greater than 0 and less than 1.")
    if args.epochs <= 0 or args.max_length < 32 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("epochs/max-length must be positive; max-steps is -1 or a positive smoke-test count.")
    try:
        train(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
