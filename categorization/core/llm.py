"""Single-label topic categorization with a local model served by Ollama."""

import json
import time
import urllib.error
import urllib.request

import pandas as pd

from categorization.core.examples import FEW_SHOT_EXAMPLES
from categorization.core.feedback import clean_feedback
from categorization.core.taxonomy import CATEGORIES, CODES, RULES

# Not "localhost": on Windows it tries IPv6 first and adds about 2 s per request
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "hf.co/typhoon-ai/typhoon2.5-qwen3-4b-gguf:Q4_K_M"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": CODES}},
    "required": ["category"],
}


def system_prompt():
    lines = [
        "You categorize Thai employee survey feedback into exactly one organizational topic.",
        "Topics (code: name - Thai description):",
    ]
    lines += [f"- {code}: {info['name']} - {info['description']}" for code, info in CATEGORIES.items()]
    lines += ["Rules:"] + [f"- {rule}" for rule in RULES]
    lines.append('Answer with JSON only, for example {"category": "workload"}.')
    return "\n".join(lines)


def build_messages(text, examples=()):
    messages = [{"role": "system", "content": system_prompt()}]
    for example, code in examples:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": json.dumps({"category": code})})
    messages.append({"role": "user", "content": text})
    return messages


def _post(host, path, payload, timeout):
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from None
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"Cannot reach Ollama at {host} ({error}). Start Ollama and check --host."
        ) from None
    except json.JSONDecodeError:
        raise RuntimeError("Ollama returned an invalid JSON response.") from None


def parse_category(content):
    try:
        category = json.loads(content).get("category")
    except (json.JSONDecodeError, AttributeError):
        return None
    return category if isinstance(category, str) and category in CATEGORIES else None


def categorize_one(text, *, model=DEFAULT_MODEL, host=DEFAULT_HOST, examples=(), timeout=120):
    payload = {
        "model": model,
        "messages": build_messages(text, examples),
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 2048},
    }
    start = time.perf_counter()
    response = _post(host, "/api/chat", payload, timeout)
    message = response.get("message") if isinstance(response, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Ollama returned an invalid chat response.")

    content = message["content"]

    seconds = time.perf_counter() - start

    category = parse_category(content)
    return {"category": category, "category_valid": category is not None, "category_seconds": seconds,
            "raw_output": content}


def categorize(texts, *, model=DEFAULT_MODEL, host=DEFAULT_HOST, few_shot=False, timeout=120):
    series = clean_feedback(texts)
    examples = FEW_SHOT_EXAMPLES if few_shot else ()
    rows = [categorize_one(text, model=model, host=host, examples=examples, timeout=timeout) for text in series]
    return pd.DataFrame(rows, index=series.index,
                        columns=["category", "category_valid", "category_seconds", "raw_output"])


def loaded_model_memory(model, host=DEFAULT_HOST):
    """Return Ollama's reported total and VRAM bytes for a loaded model, or None."""
    request = urllib.request.Request(host.rstrip("/") + "/api/ps")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            running = json.loads(response.read().decode("utf-8")).get("models", [])
    except (urllib.error.URLError, json.JSONDecodeError):
        return None
    for entry in running:
        if entry.get("name") == model or entry.get("model") == model:
            return {"size_bytes": entry.get("size"), "vram_bytes": entry.get("size_vram")}
    return None
