"""Shared experiment contracts; importable without PyTorch or model downloads."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from categorization.core.examples import FEW_SHOT_EXAMPLES
from categorization.core.llm import build_messages, system_prompt
from categorization.core.taxonomy import CODES

BASE_MODEL = "typhoon-ai/typhoon2.5-qwen3-4b"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "finetuning"


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_hash():
    return hashlib.sha256(system_prompt().encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def training_example(row):
    return {
        "id": row["id"], "group_id": row["group_id"], "category": row["category"],
        "messages": build_messages(row["feedback"]) + [
            {"role": "assistant", "content": json.dumps({"category": row["category"]})}
        ],
    }


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_sha256"] != prompt_hash():
        raise ValueError("The shared taxonomy/prompt changed. Prepare a new dataset and experiment.")
    for name, expected in manifest["file_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Prepared dataset was edited: {name}. Prepare it again.")
    return manifest
