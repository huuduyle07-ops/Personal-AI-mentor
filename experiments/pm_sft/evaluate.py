"""Measure validation and evaluate a sealed holdout with a predeclared paired comparison."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .contract import INTENTS, SLOTS, messages, parse, proposal_violation, validate
from .dataset import ARTIFACTS, CONFIG, load_split, read_json, write_json

ABSENT = object()
WARMUP = {"text": "Today I practiced for 11 minutes.", "pending": None}


def score_one(row, raw, elapsed_ms):
    value = None
    parsed = valid = False
    try:
        value = parse(raw)
        parsed = True
        validate(value)
        valid = True
    except (ValueError, RecursionError):
        pass

    gold = row["gold"]
    predicted = value if valid else {"i": [], "s": {}, "m": []}
    expected = gold["s"]
    actual = predicted["s"]
    slot_keys = (expected.keys() | actual.keys()) if valid else expected.keys()
    slots = {key: valid and actual.get(key, ABSENT) == expected.get(key, ABSENT) for key in slot_keys}
    intent_exact = valid and set(predicted["i"]) == set(gold["i"])
    missing_exact = valid and set(predicted["m"]) == set(gold["m"])
    whole = intent_exact and missing_exact and actual == expected
    return {
        "id": row["id"], "language": row["language"], "gold_intents": gold["i"],
        "predicted_intents": predicted["i"], "intent_exact": intent_exact,
        "slots": slots, "missing_exact": missing_exact, "whole_json": whole,
        "parse_valid": parsed, "schema_valid": valid,
        "policy_violation": proposal_violation(value, row),
        "latency_ms": elapsed_ms, "raw": raw,
    }


def _ratio(num, denom):
    return num / denom if denom else None


def summarize(examples):
    n = len(examples)
    tp, fp, fn = Counter(), Counter(), Counter()
    for result in examples:
        expected, actual = set(result["gold_intents"]), set(result["predicted_intents"])
        tp.update(expected & actual)
        fp.update(actual - expected)
        fn.update(expected - actual)
    per_intent = {
        intent: {
            "support": tp[intent] + fn[intent], "tp": tp[intent], "fp": fp[intent], "fn": fn[intent],
            "precision": _ratio(tp[intent], tp[intent] + fp[intent]),
            "recall": _ratio(tp[intent], tp[intent] + fn[intent]),
            "f1": _ratio(2 * tp[intent], 2 * tp[intent] + fp[intent] + fn[intent]),
        } for intent in INTENTS
    }
    per_slot = {}
    for key in SLOTS:
        values = [result["slots"][key] for result in examples if key in result["slots"]]
        per_slot[key] = {"count": len(values), "accuracy": _ratio(sum(values), len(values))}
    slot_values = [correct for result in examples for correct in result["slots"].values()]
    latency = sorted(result["latency_ms"] for result in examples)
    return {
        "examples": n,
        **{key: _ratio(sum(result[key] for result in examples), n) for key in (
            "intent_exact", "missing_exact", "whole_json", "parse_valid", "schema_valid", "policy_violation",
        )},
        "slot_accuracy": _ratio(sum(slot_values), len(slot_values)), "slot_count": len(slot_values),
        "per_slot": per_slot, "per_intent": per_intent,
        "latency_ms": {"median": median(latency), "p95": latency[math.ceil(n * .95) - 1]} if n else {},
    }


def evaluate(rows, predict):
    examples = []
    for row in rows:
        begin = perf_counter()
        raw = predict(row)
        examples.append(score_one(row, raw, (perf_counter() - begin) * 1000))
    return {
        "overall": summarize(examples),
        "by_language": {
            language: summarize([result for result in examples if result["language"] == language])
            for language in sorted({row["language"] for row in rows})
        },
        "examples": examples,
    }


def tiny_evaluate(rows, config, adapter=None):
    from .model import generate, load_base, load_tokenizer

    tokenizer = load_tokenizer(config, source=adapter)
    model = load_base(config)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    generate(model, tokenizer, WARMUP, config)  # Exclude warm-up and load from timed requests.
    return evaluate(rows, lambda row: generate(model, tokenizer, row, config))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def ollama_evaluate(rows, url, name, config):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("Ollama URL must be a plain loopback HTTP origin")
    opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        query = Request(url.rstrip("/") + path, data, {"Content-Type": "application/json"})
        with opener.open(query, timeout=120) as response:
            raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("Ollama response exceeds 1 MiB")
            return json.loads(raw)

    models = request("/api/tags")["models"]
    installed = next((model for model in models if model["name"] == name), None)
    if installed is None:
        raise ValueError(f"Ollama model {name} is absent from local /api/tags")
    version = request("/api/version")["version"]
    already_loaded = any(model["name"] == name for model in request("/api/ps")["models"])

    def predict(row):
        response = request("/api/chat", {
            "model": name, "messages": messages(row), "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": config["max_new_tokens"], "seed": config["seed"]},
        })
        return response["message"]["content"]

    try:
        predict(WARMUP)
        return {"server_version": version, "model": name, "model_digest": installed["digest"],
                **evaluate(rows, predict)}
    finally:
        # The ordinary server may have unrelated sessions. This releases only this model.
        if not already_loaded:
            try:
                request("/api/generate", {"model": name, "keep_alive": 0, "stream": False})
            except (OSError, ValueError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "ollama", "compare"))
    parser.add_argument("--dataset", type=Path, default=ARTIFACTS / "dataset")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter", type=Path, default=ARTIFACTS / "sft" / "adapter")
    parser.add_argument("--ollama-model", default="qwen3:0.6b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    output = args.output or ARTIFACTS / f"{args.mode}.json"
    if output.exists():
        parser.error(f"Output already exists: {output}")
    manifest = read_json(args.dataset / "manifest.json")
    split = "holdout" if args.mode == "compare" else "validation"
    if args.mode == "compare":
        run = read_json(args.adapter.parent / "metrics.json")
        if (run.get("status") != "complete" or run.get("mode") != "sft"
                or run.get("dataset") != manifest or not run.get("reload", {}).get("verified")):
            parser.error("A completed and verified SFT run for this sealed dataset is required")
    rows = load_split(args.dataset, split)
    config = read_json(CONFIG)
    result = {"split": split, "dataset": manifest, "settings": config}
    if args.mode == "ollama":
        try:
            result["ollama"] = ollama_evaluate(rows, args.ollama_url, args.ollama_model, config)
        except (URLError, ValueError, OSError) as exc:
            result["ollama"] = {"status": "unavailable", "reason": str(exc)}
    else:
        result["base"] = tiny_evaluate(rows, config)
        if args.mode == "compare":
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            result["qlora"] = tiny_evaluate(rows, config, adapter=args.adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f"Wrote {split} metrics to {output}")


if __name__ == "__main__":
    main()
