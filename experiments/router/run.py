"""Run with: uv run --extra research python -m experiments.router.run"""
import argparse
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from experiments.router.baseline import Baseline
from experiments.router.dataset import generate
from router import RoutingInput, RuleRouter

LABELS = ["RULE", "CLARIFY", "LLM", "DEFER"]


def evaluate(predict, records):
    expected = [row["label"] for row in records]
    events = [row["input"] for row in records]
    actual = predict(events)
    latencies = []
    # Warm-up is excluded. Each measurement is one item, not a batch average.
    predict(events[:1])
    for event in events:
        start = time.perf_counter_ns()
        predict([event])
        latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(latencies)
    return {"accuracy": accuracy_score(expected, actual),
            "macro_f1": f1_score(expected, actual, labels=LABELS, average="macro", zero_division=0),
            "classification_report": classification_report(expected, actual, labels=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(expected, actual, labels=LABELS).tolist(),
            "latency_ms": {"median": statistics.median(latencies), "p95": ordered[int((len(ordered) - 1) * .95)]}}


def run(output, seed=42, samples_per_family=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = generate(seed, samples_per_family)
    for split, records in splits.items():
        (output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    families = {key: sorted({row["family"] for row in records}) for key, records in splits.items()}
    assert not (set(families["train"]) & set(families["test"]))
    assert not (set(families["validation"]) & set(families["test"]))
    tracemalloc.start()
    start = time.perf_counter()
    candidates = []
    for regularization in (0.5, 2.0, 8.0):
        model = Baseline(seed=seed, regularization=regularization).fit(splits["train"])
        predicted = model.predict([row["input"] for row in splits["validation"]])
        score = f1_score([row["label"] for row in splits["validation"]], predicted, average="macro", zero_division=0)
        candidates.append((score, regularization, model))
    # Tie-breaking prefers the smaller C. Test labels never select hyperparameters.
    score, regularization, best = max(candidates, key=lambda item: (item[0], -item[1]))
    train_seconds = time.perf_counter() - start
    _, peak_python_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    artifact = output / "baseline.joblib"
    best.save(artifact)
    loaded = Baseline.load(artifact)
    test_inputs = [row["input"] for row in splits["test"]]
    assert best.predict(test_inputs) == loaded.predict(test_inputs)
    rule = RuleRouter()

    def rule_predict(events):
        return [rule.route(RoutingInput(**event)).route.value for event in events]

    metrics = {
        "seed": seed, "samples_per_family": samples_per_family, "labels": LABELS,
        "split_sizes": {key: len(value) for key, value in splits.items()}, "split_families": families,
        "selected_C": regularization, "validation_macro_f1": score,
        "training_seconds_including_selection": train_seconds,
        "peak_python_allocated_bytes": peak_python_memory,
        "memory_limitation": "tracemalloc excludes native allocations; not total process or GPU memory",
        "artifact_bytes": artifact.stat().st_size,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "platform": platform.platform()},
        "rule": evaluate(rule_predict, splits["test"]), "baseline": evaluate(loaded.predict, splits["test"]),
        "limitations": ["Synthetic inputs only; no real-user efficacy claims", "Metadata contains strong routing cues",
                        "No LSTM, Mamba, distillation, or critic has been trained or evaluated"],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-family", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("experiments/router/artifacts"))
    args = parser.parse_args()
    result = run(args.output, args.seed, args.samples_per_family)
    print(json.dumps({key: result[key] for key in ("seed", "split_sizes", "selected_C", "rule", "baseline")}, indent=2))


if __name__ == "__main__":
    main()
