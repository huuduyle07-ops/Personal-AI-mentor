"""Pinned, local TinyLlama loading and explicit completion-only tokenization."""

from .contract import dumps, messages


def load_tokenizer(config, source=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source or config["model"], revision=config["revision"], local_files_only=True,
        use_fast=True, trust_remote_code=False,
    )
    if not tokenizer.is_fast or tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise RuntimeError("Expected TinyLlama fast tokenizer, EOS and native chat template")
    return tokenizer


def load_base(config):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and a visible NVIDIA GPU are required")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True,
        trust_remote_code=False, quantization_config=quantization,
        dtype=torch.float16, device_map={"": 0},
    )
    if not model.is_loaded_in_4bit:
        raise RuntimeError("Expected the model to be loaded in 4-bit")
    return model


def generation_text(tokenizer, row):
    return tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)


def encode_completion(tokenizer, row, sequence_length):
    """Verify the template boundary and supervise only JSON content and EOS."""
    answer = dumps(row["gold"])
    prompt = generation_text(tokenizer, row)
    full = tokenizer.apply_chat_template(
        messages(row) + [{"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False,
    )
    start = full.rfind(answer)
    if start < 0 or full[:start].rstrip() != prompt.rstrip():
        raise ValueError("Chat template assistant boundary differs from generation prefix")
    if full[start + len(answer):].strip() != tokenizer.eos_token:
        raise ValueError("Expected one assistant EOS immediately after the response")

    # Exclude trailing template formatting after EOS from the objective.
    end = start + len(answer) + len(tokenizer.eos_token)
    encoded = tokenizer(full[:end], add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > sequence_length:
        raise ValueError(f"{row['id']} uses {len(ids)} tokens (limit {sequence_length}); no truncation applied")
    first = next(index for index, (_, right) in enumerate(encoded["offset_mapping"]) if right > start)
    left = encoded["offset_mapping"][first][0]
    if full[left:start].strip():
        raise ValueError("First supervised token overlaps the assistant header")
    decoded = tokenizer.decode(ids[first:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if first == 0 or decoded.strip() != answer or ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f"Cannot verify exact assistant completion masking for {row['id']}")
    return {"input_ids": ids, "labels": [-100] * first + ids[first:]}


def generate(model, tokenizer, row, config):
    import torch

    inputs = tokenizer(generation_text(tokenizer, row), add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
