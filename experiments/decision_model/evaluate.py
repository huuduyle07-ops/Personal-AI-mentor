"""One-time sealed test opening and shadow probability evaluation; no actions."""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psutil
import sklearn
import torch

from experiments.decision_model.data import load_partition, sha
from experiments.decision_model.features import FeatureEncoder, ablate, drop_history_channels, raw_arrays
from experiments.decision_model.metrics import evaluate_predictions
from experiments.decision_model.models import LSTM, SeparateLSTM
from experiments.decision_model.train import arrays, code_hashes, MemoryMonitor, setup_torch

HEADS = ("A_report", "B_completion_given_report")


def load_network(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTM(checkpoint["hidden_size"], checkpoint["heads"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_model(folder, entry):
    if entry["kind"] in {"statistics", "logistic"}:
        # Only the local, hash-verified artifacts produced by this run are loaded.
        return joblib.load(folder / entry["file"])
    if entry["kind"] == "lstm_joint":
        return load_network(folder / entry["file"])
    return SeparateLSTM([load_network(folder / file) for file in entry["files"]])


def prediction(model, entry, values, masks, encoder, ids, config):
    if entry["kind"] == "statistics":
        return model.predict_raw(values, masks)
    sequence = encoder.transform(values, masks)
    sequence = ablate(sequence, entry["mode"], ids, config["shuffle_seed"])
    return model.predict_proba(sequence)


def paired_analysis(rows, probabilities):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group_id"], {})[row["pair_branch"]] = index
    gaps, oracle_gaps = [], []
    for pair in groups.values():
        high, low = pair["high"], pair["low"]
        gaps.append(probabilities[high] - probabilities[low])
        oracle_gaps.append(np.asarray(rows[high]["oracle"]) - rows[low]["oracle"])
    gaps, oracle_gaps = np.stack(gaps), np.stack(oracle_gaps)
    return {head: {"pairs": len(groups), "mean_predicted_high_minus_low": float(gaps[:, index].mean()),
                   "mean_oracle_high_minus_low": float(oracle_gaps[:, index].mean()),
                   "correct_direction_fraction": float((gaps[:, index] > 1e-7).mean()),
                   "oracle_direction_positive_fraction": float((oracle_gaps[:, index] > 0).mean())}
            for index, head in enumerate(HEADS)}


def timing(model, entry, values, masks, encoder, ids, config):
    count = min(64, len(values))
    for index in range(5):
        prediction(model, entry, values[index:index + 1], masks[index:index + 1], encoder, ids[index:index + 1], config)
    elapsed = []
    for index in range(count):
        start = time.perf_counter_ns()
        prediction(model, entry, values[index:index + 1], masks[index:index + 1], encoder, ids[index:index + 1], config)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"single_window_median_ms": statistics.median(elapsed),
            "single_window_p95_ms": float(np.percentile(elapsed, 95)),
            "scope": "encoding + aggregation/network; excludes JSON/event parsing, I/O, and cold startup", "n": count}


def summaries(results):
    groups = {}
    for name, item in results.items():
        group = name.rsplit("_s", 1)[0] if item["kind"].startswith("lstm") else name
        groups.setdefault(group, []).append(item)
    summary = {}
    for group, items in groups.items():
        summary[group] = {"runs": len(items), "conditions": {}}
        for condition in ("native", "drop_20", "drop_50", "shuffle_at_inference"):
            if not all(condition in item for item in items):
                continue
            summary[group]["conditions"][condition] = {}
            for head in HEADS:
                entry = {}
                for metric in ("brier", "log_loss", "pr_auc_ap", "ece"):
                    values = [item[condition][head][metric] for item in items if item[condition][head][metric] is not None]
                    entry[metric] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0}
                summary[group]["conditions"][condition][head] = entry
        summary[group]["model_bytes_mean"] = statistics.mean(item["artifact_bytes"] for item in items)
        summary[group]["inference_median_ms_mean"] = statistics.mean(item["timing"]["single_window_median_ms"] for item in items)
    return summary


def evaluate(data, models, output, replay=False):
    data, models, output = Path(data), Path(models), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = models / "test_opened.json"
    if marker.exists() and not replay:
        raise ValueError("Test was already opened. Use --replay only to reproduce unchanged frozen models")
    selection = json.loads((models / "selection.json").read_text())
    protocol = selection["protocol"]
    if sha(data / "manifest.json") != protocol["dataset_manifest_sha256"] or code_hashes() != protocol["source_hashes"]:
        raise ValueError("Source or dataset changed since validation selection; cannot claim a sealed evaluation")
    for file, expected in selection["artifact_sha256"].items():
        if sha(models / file) != expected:
            raise ValueError("Model/preprocessing artifact changed after selection")
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["selection_sha256"] != sha(models / "selection.json"):
            raise ValueError("Selection changed after test opening")
    else:
        marker.write_text(json.dumps({"opened_at": datetime.now(timezone.utc).isoformat(),
                                      "selection_sha256": sha(models / "selection.json")}, indent=2), encoding="utf-8")
    config = protocol["config"]
    setup_torch(config, config["model_seeds"][0])
    rows, pairs = load_partition(data, "test"), load_partition(data, "test", paired=True)
    values, masks = raw_arrays(rows)
    pair_values, pair_masks = raw_arrays(pairs)
    y, target_mask = arrays(rows)
    encoder = FeatureEncoder.load(models / "encoder.json")
    ids, pair_ids = [row["id"] for row in rows], [row["id"] for row in pairs]
    robustness = {f"drop_{int(rate * 100)}": drop_history_channels(values, masks, ids, config["robustness_seed"], rate)
                  for rate in (.2, .5)}
    entries = {}
    with MemoryMonitor() as monitor:
        for name, entry in selection["models"].items():
            model = load_model(models, entry)
            probabilities = prediction(model, entry, values, masks, encoder, ids, config)
            file_names = entry.get("files", [entry.get("file")])
            result = {"kind": entry["kind"], "native": evaluate_predictions(y, target_mask, probabilities),
                      "artifact_bytes": sum((models / file).stat().st_size for file in file_names),
                      "timing": timing(model, entry, values, masks, encoder, ids, config),
                      "paired": paired_analysis(pairs, prediction(model, entry, pair_values, pair_masks, encoder, pair_ids, config)),
                      "per_family": {}}
            for condition, (damaged_values, damaged_masks) in robustness.items():
                damaged = prediction(model, entry, damaged_values, damaged_masks, encoder, ids, config)
                result[condition] = evaluate_predictions(y, target_mask, damaged)
            if entry.get("mode") == "full" and entry["kind"] != "statistics":
                sequence = encoder.transform(values, masks)
                shuffled = ablate(sequence, "shuffled", ids, config["shuffle_seed"])
                result["shuffle_at_inference"] = evaluate_predictions(y, target_mask, model.predict_proba(shuffled))
            for family in sorted({row["family"] for row in rows}):
                chosen = np.asarray([row["family"] == family for row in rows])
                result["per_family"][family] = evaluate_predictions(y[chosen], target_mask[chosen], probabilities[chosen])
            entries[name] = result
    manifest = json.loads((data / "manifest.json").read_text())
    results = {"version": "0.1", "shadow_mode": True, "protocol": protocol,
               "dataset_counts": manifest["counts"], "models": entries, "summary": summaries(entries),
               "test_prevalence": {"A": float(y[:, 0].mean()), "B_given_A": float(y[target_mask[:, 1].astype(bool), 1].mean())},
               "test_observed_B": int(target_mask[:, 1].sum()), "memory": monitor.metrics(),
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                               "sklearn": sklearn.__version__, "platform": platform.platform(), "device": "cpu"},
               "limitations": ["Synthetic shared simulator; not real-user benefit", "Only three held-out scenario mechanisms",
                               "Overlapping windows within grouped personas; row counts are not independent sample size",
                               "B is conditional on valid in-window reports, not missing-goal failure",
                               "CPU RSS includes interpreter, datasets, libraries and allocator caching",
                               "Three model seeds on one split are not independent dataset replicates"]}
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, output / "REPORT.md")
    return results


def write_report(results, path):
    text = "# Temporal outcome benchmark v0.1 results\n\nLower Brier/log loss is better. B is scored only when A=1.\n\n"
    text += "| Model | A Brier | B Brier | A log loss | B log loss | A AP | B AP |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name, item in results["summary"].items():
        native = item["conditions"]["native"]
        a, b = [native[head] for head in HEADS]
        metrics = [a["brier"], b["brier"], a["log_loss"], b["log_loss"], a["pr_auc_ap"], b["pr_auc_ap"]]
        text += f"| {name} | " + " | ".join(f"{metric['mean']:.4f}" for metric in metrics) + " |\n"
    text += "\nLSTM rows average three initialization seeds on one grouped split; full reliability bins, per-family scores, robustness, paired diagnostics, sizes, latency and RSS are in results.json.\n"
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("experiments/decision_model/artifacts/data"))
    parser.add_argument("--models", type=Path, default=Path("experiments/decision_model/artifacts/models"))
    parser.add_argument("--output", type=Path, default=Path("experiments/decision_model/artifacts/evaluation"))
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.data, args.models, args.output, args.replay)
    print(json.dumps({name: {head: scores["brier"]["mean"] for head, scores in item["conditions"]["native"].items()}
                      for name, item in result["summary"].items()}, indent=2))


if __name__ == "__main__":
    main()
