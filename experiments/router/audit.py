"""Frozen offline representation audit; never reads personal state or tunes on test.

Run: uv run --locked --extra research python -m experiments.router.audit
"""
import argparse
import hashlib
import json
import platform
import re
import statistics
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from experiments.router.baseline import Baseline
from experiments.router.dataset import FAMILIES, encode, generate
from experiments.router.mixed_baseline import MixedBaseline, columns
from experiments.router.run import LABELS
from router import RoutingInput, RuleRouter

# Fixed before the alternative's test results were inspected. Do not choose seeds
# or change representations based on the results emitted by this audit.
SEEDS = (0, 1, 2, 3, 4, 42, 73, 101, 2026, 2027)
C_GRID = (0.5, 2.0, 8.0)
ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    files = [*ROOT.glob("*.py"), ROOT / "pyproject.toml", ROOT / "uv.lock",
             *Path(__file__).parent.glob("*.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def score(records, predictions):
    expected = [row["label"] for row in records]
    return {"accuracy": accuracy_score(expected, predictions),
            "macro_f1": f1_score(expected, predictions, labels=LABELS, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(expected, predictions, labels=LABELS).tolist(),
            "per_class": classification_report(expected, predictions, labels=LABELS, output_dict=True, zero_division=0)}


def check_dataset(splits):
    """Audit all pairs, canonical templates, explicit labels, and routing contract."""
    family_labels = {family: label for label, families in FAMILIES.items() for family, _, _ in families}
    if len(family_labels) != sum(map(len, FAMILIES.values())):
        raise ValueError("Non-unique family identifiers")
    groups, texts, canonical, encoded = {}, {}, {}, {}
    mismatches, inconsistent_families = [], []
    router = RuleRouter()
    for split, records in splits.items():
        groups[split] = {row["family"] for row in records}
        texts[split] = {row["input"]["text"] for row in records}
        canonical[split] = {re.sub(r"\d+", "<number>", value) for value in texts[split]}
        encoded[split] = {encode(row["input"]) for row in records}
        for row in records:
            if family_labels[row["family"]] != row["label"]:
                inconsistent_families.append(row["family"])
            if router.route(RoutingInput(**row["input"])).route.value != row["label"]:
                mismatches.append(row["family"])
    overlaps = {}
    for left, right in combinations(splits, 2):
        overlaps[f"{left}/{right}"] = {
            "families": sorted(groups[left] & groups[right]),
            "exact_text_count": len(texts[left] & texts[right]),
            "numeric_normalized_template_count": len(canonical[left] & canonical[right]),
            "encoded_input_count": len(encoded[left] & encoded[right]),
        }
    if inconsistent_families or mismatches or any(any(value.values()) for value in overlaps.values()):
        raise ValueError("Split overlap or label-contract inconsistency detected")
    return {"family_counts": {key: len(value) for key, value in groups.items()},
            "row_counts": {key: len(value) for key, value in splits.items()},
            "families": {key: sorted(value) for key, value in groups.items()},
            "class_counts": {key: dict(Counter(row["label"] for row in value)) for key, value in splits.items()},
            "pairwise_overlap": overlaps, "label_mismatches": mismatches,
            "note": "Rule agreement is a contract check, not independent human adjudication of labels"}


def select(model_type, seed, train, validation):
    """The selection boundary deliberately has no test parameter."""
    candidates = []
    for regularization in C_GRID:
        model = model_type(seed=seed, regularization=regularization).fit(train)
        predictions = model.predict([row["input"] for row in validation])
        validation_metrics = score(validation, predictions)
        candidates.append((validation_metrics["macro_f1"], regularization, model, validation_metrics))
    selected = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
    return selected[2], {"selected_C": selected[1], "validation": selected[3],
                         "candidates": [{"C": c, "validation": result} for _, c, _, result in candidates]}


def family_results(model, records):
    results = []
    for family in sorted({row["family"] for row in records}):
        rows = [row for row in records if row["family"] == family]
        predictions = model.predict([row["input"] for row in rows])
        results.append({"family": family, "true_label": rows[0]["label"], "rows": len(rows),
                        "prediction_counts": dict(Counter(predictions)), "example": rows[0]["input"]})
    return results


def baseline_evidence(model, records):
    """Exact linear true-minus-CLARIFY score attribution by ngram occurrence.

    Shared ngrams are allocated proportionally to their raw occurrence counts
    in the metadata prefix versus free text, after full TF-IDF normalization.
    This is descriptive attribution, not a causal explanation.
    """
    vectorizer, classifier = model.pipeline.steps[0][1], model.pipeline.steps[1][1]
    analyzer = vectorizer.build_analyzer()
    names = vectorizer.get_feature_names_out()
    class_index = {label: index for index, label in enumerate(classifier.classes_)}
    output = []
    for result in family_results(model, records):
        event, true = result["example"], result["true_label"]
        matrix = vectorizer.transform([encode(event)])
        scores = classifier.decision_function(matrix)[0]
        probabilities = classifier.predict_proba(matrix)[0]
        details = {**result, "example_probabilities": dict(zip(classifier.classes_, probabilities.tolist()))}
        if true != "CLARIFY":
            delta = classifier.coef_[class_index[true]] - classifier.coef_[class_index["CLARIFY"]]
            intercept = classifier.intercept_[class_index[true]] - classifier.intercept_[class_index["CLARIFY"]]
            prefix_counts = Counter(analyzer(encode({**event, "text": ""})))
            text_counts = Counter(analyzer(event["text"]))
            prefix_contribution, text_contribution, effects = 0.0, 0.0, []
            for index, value in zip(matrix.indices, matrix.data):
                feature = names[index]
                n_prefix, n_text = prefix_counts[feature], text_counts[feature]
                contribution = float(value * delta[index])
                prefix_contribution += contribution * n_prefix / (n_prefix + n_text)
                text_contribution += contribution * n_text / (n_prefix + n_text)
                effects.append((contribution, feature))
            margin = float(scores[class_index[true]] - scores[class_index["CLARIFY"]])
            assert np.isclose(intercept + prefix_contribution + text_contribution, margin)
            details["true_minus_clarify_score"] = {"margin": margin, "intercept": float(intercept),
                                                    "metadata": prefix_contribution, "text": text_contribution}
            details["strongest_clarify_ngram_contributions"] = sorted(effects)[:8]
        output.append(details)
    return output


def encoding_diagnostics(splits, model):
    train = splits["train"]
    columns_transformer = model.pipeline.steps[0][1]
    categories = columns_transformer.named_transformers_["categories"].categories_
    unknown = {}
    for split in ("validation", "test"):
        values = columns([row["input"] for row in splits[split]])
        unknown[split] = {name: sorted(set(values[:, index + 1]) - set(categories[index]))
                          for index, name in enumerate(("command", "non_urgent", "priority"))}
    metadata_signatures = {
        (row["input"]["command"], row["input"]["minutes"], row["input"]["non_urgent"], row["input"]["priority"],
         len(row["input"]["text"].split()) < 3)
        for records in splits.values() for row in records
    }
    report_rows = [row for records in splits.values() for row in records if row["family"] == "report_total"]
    contradictory = sum(int(re.search(r"\d+", row["input"]["text"])[0]) != row["input"]["minutes"] for row in report_rows)
    return {"train_categories": {name: values.tolist() for name, values in zip(("command", "non_urgent", "priority"), categories)},
            "unknown_categories": unknown, "distinct_routing_signatures": len(metadata_signatures),
            "numeric_text_vs_structured_minutes_disagreements": contradictory,
            "report_total_rows": len(report_rows),
            "train_minutes_counts": dict(Counter(str(row["input"]["minutes"]) for row in train)),
            "priority_values": sorted({row["input"]["priority"] for records in splits.values() for row in records})}


def aggregate(runs, model_name):
    output = {}
    for split in ("validation", "test"):
        for metric in ("accuracy", "macro_f1"):
            values = [run[model_name][split][metric] for run in runs]
            output[f"{split}_{metric}"] = {"mean": statistics.mean(values), "sd": statistics.stdev(values),
                                           "min": min(values), "max": max(values)}
    for label in LABELS:
        values = [run[model_name]["test"]["per_class"][label]["recall"] for run in runs]
        output[f"test_recall_{label}"] = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
    output["pooled_test_confusion_matrix"] = np.sum([run[model_name]["test"]["confusion_matrix"] for run in runs], axis=0).tolist()
    return output


def run(output, seeds=SEEDS):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    protocol = {"seeds": list(seeds), "C_grid": C_GRID, "samples_per_family": 20, "labels": LABELS,
                "models": ["unchanged serialized char TF-IDF + LR", "ColumnTransformer char TF-IDF + OHE + imputed/scaled minutes + LR"],
                "selection": "validation macro-F1 only; ties prefer smaller C; all models frozen before test phase",
                "status": "post-hoc diagnostic comparison; existing seed-42 test was already exposed",
                "source_hashes_before": before,
                "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "platform": platform.platform()}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    prepared = []
    for seed in seeds:
        splits = generate(seed)
        result = {"seed": seed, "integrity": check_dataset(splits)}
        models = {}
        for name, model_type in (("serialized", Baseline), ("mixed", MixedBaseline)):
            models[name], result[name] = select(model_type, seed, splits["train"], splits["validation"])
        result["rule"] = {}
        prepared.append((splits, models, result))
    # All hyperparameters for every seed/representation are frozen at this boundary.
    (output / "selection.json").write_text(json.dumps([item[2] for item in prepared], indent=2), encoding="utf-8")
    rule = RuleRouter()
    for splits, models, result in prepared:
        for split in ("validation", "test"):
            rows = splits[split]
            result["rule"][split] = score(rows, [rule.route(RoutingInput(**row["input"])).route.value for row in rows])
            for name, model in models.items():
                if split == "test":
                    result[name][split] = score(rows, model.predict([row["input"] for row in rows]))
                result[name][f"{split}_families"] = family_results(model, rows)
        if result["seed"] == 42:
            result["encoding"] = encoding_diagnostics(splits, models["mixed"])
            result["serialized_evidence"] = {split: baseline_evidence(models["serialized"], splits[split])
                                             for split in ("validation", "test")}
    runs = [item[2] for item in prepared]
    results = {"protocol": protocol, "runs": runs,
               "summary": {name: aggregate(runs, name) for name in ("rule", "serialized", "mixed")},
               "sources_unchanged": source_hashes() == before}
    if not results["sources_unchanged"]:
        raise RuntimeError("Source changed during the audit")
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/router/artifacts/audit/comparison"))
    args = parser.parse_args()
    results = run(args.output)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
