"""Score original outputs and export side-by-side multilingual review records."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

from experiments.pm_sft.contract import INTENTS, dumps

from .annotation import validate
from .dataset import LANGUAGES, load_reference
from .run import write_json


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "f1": divide(2 * tp, 2 * tp + fp + fn)}


def compare_sets(gold, predicted):
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def timed(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "p95": None}
    return {"n": len(values), "median": median(values), "p95": values[math.ceil(0.95 * len(values)) - 1]}


def score(row, record):
    envelope = record.get("response", {})
    raw = envelope.get("response")
    checked = validate(raw if isinstance(raw, str) else None, row)
    usable = record["status"] == "ok" and checked["schema"] and checked["contract"]
    prediction = checked["value"] if usable else {"i": [], "s": {}, "m": [], "a": False}
    gold = row["gold"]
    exact_intent = usable and set(prediction["i"]) == set(gold["i"])
    missing = usable and set(prediction["m"]) == set(gold["m"])
    ambiguity = usable and prediction["a"] == gold["a"]
    semantics = exact_intent and prediction["s"] == gold["s"] and missing and ambiguity
    whole = semantics and not checked["errors"] and not checked["policy_flags"]
    metrics = {
        "exact_intent": exact_intent, "missing_exact": missing, "ambiguity_exact": ambiguity,
        "semantics_exact": semantics, "whole_example": whole,
        "intent_counts": {name: compare_sets({name} & set(gold["i"]), {name} & set(prediction["i"]))
                          for name in INTENTS},
        "slot_counts": compare_sets(set(gold["s"].items()), set(prediction["s"].items())),
        "missing_counts": compare_sets(set(gold["m"]), set(prediction["m"])),
        "ambiguity_counts": compare_sets({True} if gold["a"] else set(),
                                        {True} if prediction["a"] else set()),
    }
    return {"id": row["id"], "language": row["language"], "status": record["status"],
            "latency_s": record["latency_s"], "done_reason": envelope.get("done_reason"),
            "generation_count": envelope.get("eval_count"), "generation_ns": envelope.get("eval_duration"),
            "load_ns": envelope.get("load_duration"), "thinking_emitted": bool(envelope.get("thinking")),
            "checks": checked, "metrics": metrics}


def sum_counts(values):
    totals = [sum(counts[index] for counts in values) for index in range(3)]
    return prf(*totals)


def summarize(scores, expected):
    count = len(scores)
    by_intent = {intent: sum_counts([s["metrics"]["intent_counts"][intent] for s in scores])
                 for intent in INTENTS}
    observed_f1 = [v["f1"] for v in by_intent.values() if v["f1"] is not None]

    def count_rate(events):
        return {"count": events, "rate": divide(events, count)}

    summary = {"expected": expected, "attempted": count, "intent_by_label": by_intent,
               "macro_intent_f1": divide(sum(observed_f1), len(observed_f1)),
               "macro_intent_labels": len(observed_f1)}
    for field in ("parsed", "schema", "contract", "evidence", "numbers", "dates"):
        summary[field] = count_rate(sum(s["status"] == "ok" and s["checks"][field] is True for s in scores))
    for field in ("exact_intent", "missing_exact", "ambiguity_exact", "semantics_exact", "whole_example"):
        summary[field] = count_rate(sum(s["metrics"][field] for s in scores))
    for field in ("slot", "missing", "ambiguity"):
        summary[field] = sum_counts([s["metrics"][field + "_counts"] for s in scores])
    summary["policy_flagged"] = count_rate(sum(bool(s["checks"]["policy_flags"]) for s in scores))
    summary["policy_flags_by_type"] = dict(Counter(flag for s in scores for flag in s["checks"]["policy_flags"]))
    summary["timeouts"] = count_rate(sum(s["status"] == "timeout" for s in scores))
    summary["transport_failures"] = count_rate(sum(s["status"] != "ok" for s in scores))
    summary["output_limit"] = count_rate(sum(s["done_reason"] == "length" for s in scores))
    summary["thinking_emitted"] = count_rate(sum(s["thinking_emitted"] for s in scores))
    summary["latency_s"] = timed(s["latency_s"] for s in scores if s["status"] == "ok")
    throughput = [s["generation_count"] * 1e9 / s["generation_ns"] for s in scores
                  if s["status"] == "ok" and s["generation_count"] and s["generation_ns"]]
    summary["decode_tokens_s"] = timed(throughput)
    return summary


def resource_summary(path, result):
    samples = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    pre = result.get("preload", {})
    min_free = min((s["free_ram_mib"] for s in samples), default=None)
    max_reads = max((s["page_reads_mib_s"] for s in samples if s["page_reads_mib_s"] is not None), default=None)
    vram = {}
    for sample in samples:
        for gpu in sample["gpus"]:
            used = gpu["used_vram_mib"]
            if used is not None:
                vram[gpu["name"]] = max(used, vram.get(gpu["name"], 0))
    warmup = result.get("warmup", {})
    loaded = result.get("loaded", [])
    allocator = loaded[0] if loaded else {}
    return {
        "samples": len(samples), "preload_available_ram_mib": pre.get("free_ram_mib"),
        "min_available_ram_mib": min_free,
        "peak_system_ram_used_mib": round(pre["total_ram_mib"] - min_free, 1) if min_free is not None else None,
        "peak_system_ram_increase_mib": round(pre["free_ram_mib"] - min_free, 1) if min_free is not None else None,
        "peak_device_vram_mib": vram, "max_hard_page_reads_mib_s": max_reads,
        "minimum_commit_headroom_mib": min((s["free_commit_mib"] for s in samples), default=None),
        "cold_request_latency_s": warmup.get("latency_s"),
        "cold_load_s": divide(warmup.get("response", {}).get("load_duration", 0), 1e9)
        if "load_duration" in warmup.get("response", {}) else None,
        "ollama_reported_size_mib": divide(allocator["size"], 2**20) if "size" in allocator else None,
        "ollama_reported_vram_mib": divide(allocator["size_vram"], 2**20) if "size_vram" in allocator else None,
        "ollama_gpu_fraction": divide(allocator["size_vram"], allocator["size"])
        if "size" in allocator and "size_vram" in allocator else None,
    }


def score_run(directory, rows):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if sha256(dumps(rows).encode()).hexdigest() != manifest["reference_sha256"]:
        raise ValueError("The current reference differs from the reference used for inference")
    by_id = {row["id"]: row for row in rows}
    report = {}
    raw_by_model = {}
    scores_by_model = {}
    for model in manifest["config"]["models"]:
        run_dir = directory / model.replace(":", "-")
        path = run_dir / "raw.jsonl"
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        result = next((entry for entry in manifest["models"] if entry["model"] == model),
                      {"model": model, "status": "incomplete", "reason": "not_started"})
        if [item["id"] for item in raw] != list(by_id)[:len(raw)] or len(raw) != result.get("attempted", 0):
            raise ValueError(f"Missing, duplicate or out-of-order artifacts for {model}")
        scores = [score(by_id[item["id"]], item) for item in raw]
        raw_by_model[model] = {item["id"]: item for item in raw}
        scores_by_model[model] = {item["id"]: item for item in scores}
        report[model] = {"status": result["status"], "reason": result.get("reason"),
                         "overall": summarize(scores, len(rows)),
                         "languages": {lang: summarize([s for s in scores if s["language"] == lang],
                                                       sum(row["language"] == lang for row in rows))
                                       for lang in LANGUAGES},
                         "resources": resource_summary(run_dir / "resources.json", result)}
        if run_dir.exists():
            write_json(run_dir / "validation.json", scores)
    write_json(directory / "metrics.json", report)
    return raw_by_model, scores_by_model


def interpretation(value):
    if type(value) is not dict:
        return value
    return {key: sorted(value[key]) if key in {"i", "m"} and isinstance(value[key], list)
            and all(isinstance(part, str) for part in value[key]) else value[key]
            for key in value if key != "e"}


def export(path, rows, raw_by_model, scores_by_model):
    path.parent.mkdir(parents=True, exist_ok=True)
    models = list(raw_by_model)
    fields = ["id", "family", "language", "text", "anchor_day", "anchor_zone", "pending", "review_status",
              "reference_json", "model_disagreement"]
    for model in models:
        fields.extend(f"{model}_{field}" for field in ("status", "raw", "validation"))
    fields.extend(["reviewer_reference_json", "reviewer_notes"])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {key: row[key] for key in fields if key in row}
            item["reference_json"] = dumps(row["gold"])
            interpreted = []
            for model in models:
                raw = raw_by_model[model].get(row["id"], {})
                checked = scores_by_model[model].get(row["id"], {}).get("checks")
                item[f"{model}_raw"] = raw.get("response", {}).get("response")
                item[f"{model}_status"] = raw.get("status", "unattempted")
                item[f"{model}_validation"] = dumps(checked) if checked is not None else ""
                if checked is not None:
                    interpreted.append(interpretation(checked["value"]))
            if len(interpreted) == 2:
                item["model_disagreement"] = interpreted[0] != interpreted[1]
            writer.writerow(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Score a completed or partial run directory")
    parser.add_argument("--review", type=Path, required=True, help="CSV with provisional labels and outputs")
    args = parser.parse_args()
    rows = load_reference()
    raw, scores = score_run(args.run, rows) if args.run else ({}, {})
    export(args.review, rows, raw, scores)


if __name__ == "__main__":
    main()
