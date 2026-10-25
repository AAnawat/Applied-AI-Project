"""CUDA/model helpers are lazy: importing dataset tools does not load a model."""
from categorization.finetuning.common import BASE_MODEL


def cuda_runtime():
    try:
        import torch
    except ImportError:
        raise RuntimeError("Use a CUDA Colab runtime with PyTorch and requirements-gpu.txt installed.") from None
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise RuntimeError("Use a CUDA runtime with PyTorch >= 2.6; the dashboard environment is not used.")
    if not torch.cuda.is_available():
        raise RuntimeError("This QLoRA experiment requires an NVIDIA CUDA GPU. No model was downloaded.")
    return torch, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base(revision=None):
    torch, dtype = cuda_runtime()
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    revision = HfApi().model_info(BASE_MODEL, revision=revision or "main").sha
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=revision, trust_remote_code=False, use_safetensors=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        ),
        torch_dtype=dtype, device_map={"": 0},
    )
    return model, tokenizer, revision


def prompt_ids(messages, tokenizer):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def encode_example(row, tokenizer, max_length):
    prefix = prompt_ids(row["messages"][:-1], tokenizer)
    answer = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
    answer = answer + [tokenizer.eos_token_id]
    ids = prefix + answer
    if len(ids) > max_length:
        raise ValueError(
            f"Example {row['id']} needs {len(ids)} tokens, exceeding {max_length}. "
            "Increase max-length within the GPU budget; examples are never silently truncated."
        )
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + answer}
