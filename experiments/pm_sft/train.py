"""QLoRA SFT with verified completion labels and a fixed optimizer-step budget."""

import argparse
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import random
from time import perf_counter

from .dataset import ARTIFACTS, CONFIG, load_split, read_json, write_json
from .evaluate import evaluate
from .model import encode_completion, generate, load_base, load_tokenizer
from .reload_adapter import verify_adapter

PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate",
            "huggingface-hub", "safetensors", "tokenizers", "numpy")


def versions():
    result = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def cuda_memory():
    import torch

    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def train_loss(model, example):
    import torch

    inputs = {key: torch.tensor([value], device=model.device) for key, value in example.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return model(**inputs, use_cache=False).loss


def validate_model(model, tokenizer, rows, encoded, config):
    import torch

    model.eval()
    try:
        with torch.inference_mode():
            losses = [train_loss(model, example).item() for example in encoded]
        if not all(map(math.isfinite, losses)):
            raise RuntimeError("Non-finite validation loss")
        scores = evaluate(rows, lambda row: generate(model, tokenizer, row, config))
        return {"loss": sum(losses) / len(losses), "task": scores["overall"], "by_language": scores["by_language"]}
    finally:
        model.train()


def shuffled_examples(examples, seed):
    rng = random.Random(seed)
    while True:
        epoch = list(examples)
        rng.shuffle(epoch)
        yield from epoch


def fit(mode, output, train, validation, config, metrics, optimizer_name):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    metrics["cuda"] = {
        "device": torch.cuda.get_device_name(),
        "runtime": torch.version.cuda,
        "initial_free_mib": torch.cuda.mem_get_info()[0] / 2**20,
    }
    tokenizer = load_tokenizer(config)
    encoded = [encode_completion(tokenizer, row, config["sequence_length"]) for row in train]
    validation_encoded = [encode_completion(tokenizer, row, config["sequence_length"]) for row in validation]
    examples = encoded + validation_encoded
    metrics["masking"] = {
        "verified_examples": len(examples),
        "max_sequence_tokens": max(len(example["input_ids"]) for example in examples),
        "min_supervised_tokens": min(sum(label != -100 for label in example["labels"]) for example in examples),
    }

    model = prepare_model_for_kbit_training(
        load_base(config), use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config["rank"], lora_alpha=config["alpha"],
        lora_dropout=config["dropout"], target_modules=config["targets"], bias="none",
    ))
    model.train()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    metrics["trainable_parameters"] = sum(parameter.numel() for parameter in params)
    if not params or any(parameter.dtype != torch.float32 for parameter in params):
        raise RuntimeError("Expected FP32 LoRA trainable parameters under FP16 autocast")

    if optimizer_name == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer = AdamW8bit(params, lr=config["learning_rate"])
    else:
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    steps = config[f"{mode}_steps"]
    stream = shuffled_examples(encoded, config["seed"])
    accumulation = config["gradient_accumulation"]
    optimizer.zero_grad(set_to_none=True)
    training_started = perf_counter()
    for step in range(1, steps + 1):
        begin = perf_counter()
        losses = []
        for _ in range(accumulation):
            loss = train_loss(model, next(stream))
            loss_value = loss.detach().item()
            losses.append(loss_value)
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss at optimizer step {step}")
            scaler.scale(loss / accumulation).backward()

        scaler.unscale_(optimizer)
        with_grad = [parameter for parameter in params if parameter.grad is not None]
        gradient = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        metrics["last_gradient"] = {"step": step, "norm": gradient if math.isfinite(gradient) else None,
                                    "finite": math.isfinite(gradient), "tensors": len(with_grad)}
        if len(with_grad) != len(params) or not math.isfinite(gradient) or gradient <= 0:
            raise RuntimeError(f"Missing, zero or non-finite gradients at optimizer step {step}")
        strongest = max(with_grad, key=lambda parameter: parameter.grad.abs().max().item()) if step == 1 else None
        before = strongest.detach().clone() if strongest is not None else None
        scaler.step(optimizer)
        scaler.update()
        if strongest is not None:
            delta = (strongest.detach() - before).abs().max().item()
            if not math.isfinite(delta) or delta <= 0:
                raise RuntimeError("First optimizer step did not make a finite parameter update")
            metrics["first_update_max_delta"] = delta
            del before, strongest
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        record = {"step": step, "loss": sum(losses) / len(losses), "grad_norm": gradient,
                  "optimizer_step_seconds": perf_counter() - begin, **cuda_memory()}
        metrics["steps"].append(record)
        print(json.dumps(record), flush=True)
        write_json(output / "metrics.json", metrics)
        if mode == "sft" and (step % config["validate_every"] == 0 or step == steps):
            metrics["validation"].append({
                "step": step, **validate_model(model, tokenizer, validation, validation_encoded, config),
            })
            print(f"Step {step}: validation loss {metrics['validation'][-1]['loss']:.4f}", flush=True)
            write_json(output / "metrics.json", metrics)

    metrics["fit_seconds"] = perf_counter() - training_started
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter)
    return adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "sft"))
    parser.add_argument("--dataset", type=Path, default=ARTIFACTS / "dataset")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=ARTIFACTS / "baseline.json")
    parser.add_argument("--smoke-metrics", type=Path, default=ARTIFACTS / "smoke" / "metrics.json")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit")
    args = parser.parse_args()
    output = args.output or ARTIFACTS / args.mode
    settings = read_json(CONFIG)
    if args.sequence_length is not None:
        if not 1 <= args.sequence_length <= settings["sequence_length"]:
            parser.error("Sequence length must be positive and no greater than the preset")
        settings["sequence_length"] = args.sequence_length
    if settings["microbatch_size"] != 1:
        parser.error("This implementation requires microbatch size one")

    manifest = read_json(args.dataset / "manifest.json")
    baseline = read_json(args.baseline)
    if baseline.get("split") != "validation" or baseline.get("dataset") != manifest or "base" not in baseline:
        parser.error("Run the original model on this sealed validation split first")
    if args.mode == "sft":
        smoke = read_json(args.smoke_metrics)
        if (smoke.get("status") != "complete" or smoke.get("mode") != "smoke"
                or smoke.get("dataset") != manifest or smoke.get("settings") != settings
                or smoke.get("optimizer") != args.optimizer or not smoke.get("reload", {}).get("verified")):
            parser.error("Complete the matching dataset and configuration smoke run first")

    train = load_split(args.dataset, "train")
    validation = load_split(args.dataset, "validation")
    if args.mode == "smoke":
        # Exercise all three tokenizers paths on authored examples while keeping the smoke run small.
        train = train[:24]
        validation = validation[:3]
    output.mkdir(parents=True, exist_ok=False)
    metrics = {"mode": args.mode, "status": "running", "environment": versions(),
               "settings": settings, "optimizer": args.optimizer, "steps": [], "validation": [],
               "dataset": manifest, "training_examples": len(train), "validation_examples": len(validation)}
    start = perf_counter()
    try:
        adapter = fit(args.mode, output, train, validation, settings, metrics, args.optimizer)
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        metrics["reload"] = verify_adapter(adapter, settings)
        metrics["status"] = "complete"
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if "cuda" in metrics:
            metrics["cuda"].update(cuda_memory())
        metrics["elapsed_seconds"] = perf_counter() - start
        metrics["optimizer_steps_completed"] = len(metrics["steps"])
        write_json(output / "metrics.json", metrics)


if __name__ == "__main__":
    main()
