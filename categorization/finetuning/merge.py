"""Optional export: merge a selected adapter into original non-quantized weights."""
import argparse
import json
from pathlib import Path

from categorization.finetuning.runtime import cuda_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory.")
    metadata = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    if metadata["smoke_only"]:
        parser.error("Do not export a smoke-test adapter as the deployed model.")
    # Export needs more VRAM than 4-bit training; never promise a 16GB GPU will fit.
    torch, dtype = cuda_runtime()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(
        metadata["base_model"], revision=metadata["base_revision"],
        torch_dtype=dtype, device_map={"": 0}, trust_remote_code=False, use_safetensors=True,
    )
    model = PeftModel.from_pretrained(base, args.run / "adapter").merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["base_model"], revision=metadata["base_revision"], trust_remote_code=False,
    )
    tokenizer.save_pretrained(args.output)
    (args.output / "source_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved merged Safetensors to {args.output}. Re-evaluate after deployment conversion.")


if __name__ == "__main__":
    main()
