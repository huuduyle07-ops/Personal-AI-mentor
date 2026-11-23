"""One sealed test opening; aggregate reports never export learner records."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import load_split, make_windows
from .features import context_of
from .metrics import learner_bootstrap_delta, scores
from .models import FittedPredictor
from .train import seed_everything
from .utils import (DEFAULT_ARTIFACTS, MemoryMonitor, digest, read_json, source_hashes, write_json)


def latency(predictor, histories):
    samples = []
    for learner in histories:
        events = learner["events"]
        for i in np.linspace(1, len(events)-1, min(10, len(events)-1), dtype=int):
            samples.append((events[:i], context_of(events[i])))
    for history, context in samples[:5]:
        predictor.predict(history, context)
    timings = []
    for history, context in samples:
        start = time.perf_counter_ns()
        predictor.predict(history, context)
        timings.append((time.perf_counter_ns()-start)/1e6)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, 0.95)),
            "queries": len(timings), "scope": "single CPU query, raw-history encoding + inference; no I/O/startup"}


def plot_results(result, selection, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(result["main"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind, record in result["main"].items():
        bins = record["metrics"]["calibration"]["bins"]
        axes[0].plot([b["mean_probability"] for b in bins], [b["observed_rate"] for b in bins], "o-", label=kind)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed correctness", title="Test calibration (10 fixed bins)")
    axes[0].legend(fontsize=8)
    axes[1].bar(names, [result["main"][n]["metrics"]["brier"] for n in names])
    axes[1].set(ylabel="Brier score (lower is better)", title="Unseen learners, selected models")
    fig.tight_layout()
    fig.savefig(output / "calibration_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2*len(names), 3.3), squeeze=False)
    for ax, name in zip(axes[0], names):
        matrix = np.asarray(result["main"][name]["metrics"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Observed", title=name)
    fig.suptitle("Test confusion matrices; threshold = 0.5")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lookup = {r["id"]: r for r in selection["candidates"]}
    for name in ("gru", "lstm", "mlp"):
        curve = lookup[selection["selected"][name]]["curves"]
        axes[0].plot([c["epoch"] for c in curve], [c["train_log_loss"] for c in curve], "o-", label=name)
        axes[1].plot([c["epoch"] for c in curve], [c["validation_brier"] for c in curve], "o-", label=name)
    axes[0].set(xlabel="Epoch", ylabel="Training log loss", title="Selected-model learning curves")
    axes[1].set(xlabel="Epoch", ylabel="Validation Brier", title="Early-stopping diagnostics")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind in ("statistical", "logistic", "gru", "lstm"):
        rows = sorted([r for r in result["length_models"] if r["kind"] == kind and r["length"] > 0], key=lambda r: r["length"])
        axes[0].plot([r["length"] for r in rows], [r["metrics"]["brier"] for r in rows], "o-", label=kind)
    axes[0].set(xlabel="Maximum historical window", ylabel="Test Brier", xticks=[16, 32, 64], title="Retrained history-length models")
    modes = ["full", "last", "shuffled", "missing20", "missing50"]
    for kind in ("statistical", "logistic", "gru", "lstm"):
        axes[1].plot(modes, [result["perturbations"][kind][mode]["brier"] for mode in modes], "o-", label=kind)
    axes[1].set(ylabel="Test Brier", title="Inference-only history perturbations")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "history_ablations.png", dpi=150)
    plt.close(fig)


def evaluate(data, models, output, replay=False):
    data, models, output = Path(data), Path(models), Path(output)
    selection = read_json(models / "selection.json")
    if digest(data / "manifest.json") != selection["data_manifest_sha256"]:
        raise ValueError("Data manifest changed after model selection")
    if source_hashes() != selection["source_sha256"]:
        raise ValueError("Source changed after model selection; use a fresh run")
    for name, expected in selection["artifact_sha256"].items():
        if digest(models / name) != expected:
            raise ValueError(f"Artifact changed after selection: {name}")
    seal = {"selection_sha256": digest(models / "selection.json"),
            "test_sha256": read_json(data / "manifest.json")["files"]["test.jsonl"]}
    marker = models / "test_opened.json"
    if marker.exists():
        if not replay or read_json(marker) != seal:
            raise ValueError("Test already opened; unchanged-artifact replay requires --replay")
    else:
        write_json(marker, seal)
    config = selection["config"]
    seed_everything(config["seed"], config["threads"])
    output.mkdir(parents=True, exist_ok=True)
    result = {"version": "0.1", "seal": seal, "main": {}, "length_models": [],
              "retrained_controls": [], "perturbations": {}, "subgroups": {}, "comparisons": {},
              "capacity_comparisons": {}}
    records = selection["candidates"] + selection["controls"]
    lookup = {r["id"]: r for r in records}
    with MemoryMonitor() as memory:
        histories = load_split(data, "test")
        main_probabilities = {}
        for kind, identifier in selection["selected"].items():
            predictor = FittedPredictor.load(models / identifier)
            windows = make_windows(histories, predictor.encoder, predictor.length, seed=config["seed"])
            with MemoryMonitor() as model_memory:
                p = predictor.predict_windows(windows)
                timing = latency(predictor, histories)
            main_probabilities[kind] = p
            result["main"][kind] = {"id": identifier, "length": predictor.length,
                                    "metrics": scores(windows.y, p, windows.groups, config["threshold"]),
                                    "training_seconds": lookup[identifier]["training_seconds"],
                                    "parameters": lookup[identifier]["parameters"],
                                    "artifact_bytes": sum(f.stat().st_size for f in (models / identifier).iterdir()),
                                    "latency": timing, "memory": model_memory.result}
            subgroups = {}
            masks = {"short_prior_1_to_15": windows.history_counts < 16,
                     "medium_prior_16_to_63": (windows.history_counts >= 16) & (windows.history_counts < 64),
                     "long_prior_64_plus": windows.history_counts >= 64,
                     **{f"format_{f}": np.asarray(windows.formats) == f for f in sorted(set(windows.formats))}}
            for name, mask in masks.items():
                if mask.any():
                    subgroups[name] = scores(windows.y[mask], p[mask], windows.groups[mask], config["threshold"])
            subgroups["confident_errors"] = {"wrong_probability_ge_0.9": int(((p >= .9) & (windows.y == 0)).sum()),
                                                "correct_probability_le_0.1": int(((p <= .1) & (windows.y == 1)).sum())}
            result["subgroups"][kind] = subgroups
            result["perturbations"][kind] = {"full": result["main"][kind]["metrics"]}
            del windows
            for mode in ("last", "shuffled", "missing20", "missing50"):
                perturbed = make_windows(histories, predictor.encoder, predictor.length, mode, config["seed"])
                result["perturbations"][kind][mode] = scores(perturbed.y, predictor.predict_windows(perturbed),
                                                             perturbed.groups, config["threshold"])
                del perturbed
        # Evaluate each predeclared length with its validation-selected smoothing/C.
        best_length = {}
        for record in selection["candidates"]:
            key = record["kind"], record["length"]
            if key not in best_length or record["validation"]["brier"] < best_length[key]["validation"]["brier"]:
                best_length[key] = record
        for record in list(best_length.values()) + selection["controls"]:
            predictor = FittedPredictor.load(models / record["id"])
            windows = make_windows(histories, predictor.encoder, predictor.length, predictor.mode, config["seed"])
            control_probabilities = predictor.predict_windows(windows)
            row = {"id": record["id"], "kind": record["kind"], "length": predictor.length,
                   "mode": predictor.mode, "validation_brier": record["validation"]["brier"],
                   "metrics": scores(windows.y, control_probabilities, windows.groups, config["threshold"])}
            if predictor.mode != "full":
                row["ordered_minus_control"] = learner_bootstrap_delta(
                    windows.y, main_probabilities[record["kind"]], control_probabilities,
                    windows.groups, config["seed"])
            destination = "length_models" if predictor.mode == "full" else "retrained_controls"
            result[destination].append(row)
        baseline = min(("statistical", "logistic"), key=lambda kind: lookup[selection["selected"][kind]]["validation"]["brier"])
        result["reference_baseline_selected_on_validation"] = baseline
        for kind in ("gru", "lstm", "mlp"):
            result["comparisons"][kind] = learner_bootstrap_delta(windows.y, main_probabilities[kind],
                                                                  main_probabilities[baseline], windows.groups, config["seed"])
        for kind in ("gru", "lstm"):
            result["capacity_comparisons"][kind] = learner_bootstrap_delta(
                windows.y, main_probabilities[kind], main_probabilities["mlp"], windows.groups, config["seed"])
    result["resources"] = memory.result
    # Aggregate-only snapshot: no IDs, records, predictions, or model weights.
    manifest = read_json(data / "manifest.json")
    result["dataset"] = {"splits": manifest["splits"], "provenance": manifest["provenance"]}
    result["training"] = {k: v for k, v in selection.items() if k != "artifact_sha256"}
    write_json(output / "results.json", result)
    plot_results(result, selection, output)
    print(json.dumps({kind: r["metrics"] for kind, r in result["main"].items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_ARTIFACTS / "data")
    parser.add_argument("--models", type=Path, default=DEFAULT_ARTIFACTS / "models")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "evaluation")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    evaluate(args.data, args.models, args.output, args.replay)


if __name__ == "__main__":
    main()
