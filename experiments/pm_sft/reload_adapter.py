"""Verify saved LoRA state and its effect on a freshly loaded quantized model."""

import argparse
from pathlib import Path


def verify_adapter(adapter, config):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    from .model import generation_text, load_base, load_tokenizer

    tokenizer = load_tokenizer(config, source=adapter)
    model = PeftModel.from_pretrained(load_base(config), adapter, local_files_only=True)
    model.eval()
    saved = load_file(adapter / "adapter_model.safetensors")
    actual = get_peft_model_state_dict(model)
    if not saved or actual.keys() != saved.keys() or any(
        not torch.equal(value.cpu(), saved[key]) for key, value in actual.items()
    ):
        raise RuntimeError("Reloaded LoRA tensors differ from saved tensors")
    probe = {"text": "Today I did 12 minutes.", "pending": None}
    inputs = tokenizer(generation_text(tokenizer, probe), add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        adapted = model(**inputs, use_cache=False).logits[0, -1].float()
        with model.disable_adapter():
            original = model(**inputs, use_cache=False).logits[0, -1].float()
        max_delta = (adapted - original).abs().max().item()
    if not torch.isfinite(adapted).all().item() or not (0 < max_delta < float("inf")):
        raise RuntimeError("Reloaded adapter did not make a finite change to probe logits")
    return {"verified": True, "tensor_count": len(saved), "probe_logit_max_delta": max_delta}


def main():
    from .dataset import CONFIG, read_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    print(verify_adapter(args.adapter, read_json(CONFIG)))


if __name__ == "__main__":
    main()
