"""Select on validation learners, then evaluate once on held-out test learners and export HLR weights.

    uv run --locked --extra research python -m experiments.forgetting_curve.data
    uv run --locked --extra research python -m experiments.forgetting_curve.run
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .data import ARTIFACTS, SOURCE, load
from .models import (FEATURES, Constant, HalfLifeRegression, HalfLifeRegressionLexeme,
                     Leitner, LogisticRegression, Pimsleur, bootstrap_mae_difference, calibration, evaluate)

RESULTS = Path(__file__).with_name("results")
EXPORT = Path(__file__).resolve().parents[2] / "models" / "recall_model.json"
GRID = {"hlr": {"alpha": [0.0, 0.001, 0.01], "lam": [0.0, 0.001]},
        "hlr_lexeme": {"alpha": [0.0, 0.001, 0.01], "lam": [0.0, 1e-5]},
        "logistic_regression": {"lam": [0.0, 0.001]}}
FACTORIES = {"hlr": HalfLifeRegression, "hlr_lexeme": HalfLifeRegressionLexeme,
             "logistic_regression": LogisticRegression}


def summarize(split):
    return {"rows": int(len(split["p"])), "learners": int(len(np.unique(split["user"]))),
            "mean_p_recall": float(split["p"].mean())}


def select(name, train, validation):
    """Fit every grid point on train, keep the lowest validation MAE. Test data is not touched here."""
    trials = []
    keys = list(GRID[name])
    for values in itertools.product(*(GRID[name][key] for key in keys)):
        params = dict(zip(keys, values))
        start = time.time()
        model = FACTORIES[name](**params).fit(train)
        p_hat, h_hat = model.predict(validation)
        metrics = evaluate(validation["p"], p_hat, h_hat, validation["delta_days"])
        trials.append({"params": params, "validation": metrics, "fit_seconds": round(time.time() - start, 1)})
        print(f"  {name} {params}: validation MAE {metrics['mae']:.4f}")
    best = min(trials, key=lambda trial: trial["validation"]["mae"])
    return FACTORIES[name](**best["params"]).fit(train), best, trials


def export(model, provenance, test_metrics):
    """Write the production recall model: logistic regression with a log-time decay term.

    HLR was the pre-declared candidate, but validation showed it collapses to "always remembered" and
    learns that wrong answers lengthen memory (see REPORT.md), so the bot uses the logistic model.
    """
    payload = {
        "model": "logistic_recall", "version": 1,
        "formula": "p = sigmoid(w . [sqrt(1+right), sqrt(1+wrong), 1, log(1+days_since_last_review)])",
        "features": ["sqrt_right", "sqrt_wrong", "bias", "log1p_days"],
        "weights": [round(float(value), 6) for value in model.w],
        "trained_on": {"source": SOURCE, "sample_sha256": provenance["sha256"], "rows": provenance["rows"]},
        "held_out_test": {key: round(value, 4) for key, value in test_metrics.items()},
        "attribution": "Derived from Duolingo learning traces (Settles & Meeder 2016), CC BY-NC 4.0. "
                       "Noncommercial use only.",
    }
    EXPORT.parent.mkdir(exist_ok=True)
    EXPORT.write_text(json.dumps(payload, indent=2) + "\n")


def export_only(sample):
    """Refit the already-selected logistic model on train and export it. Does not touch test data:
    its test metrics are copied from the single recorded test opening in results.json."""
    provenance = json.loads((sample.parent / "provenance.json").read_text())
    selection = json.loads((RESULTS / "selection.json").read_text())
    results = json.loads((RESULTS / "results.json").read_text())
    params = selection["logistic_regression"]["chosen"]["params"]
    model = LogisticRegression(**params).fit(load(sample)["train"])
    export(model, provenance, results["test"]["logistic_regression"])
    print(f"Exported logistic recall model {params} to {EXPORT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=ARTIFACTS / "sample.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results (a replay, not a new test)")
    parser.add_argument("--export-only", action="store_true", help="Refit the selected model and export it")
    options = parser.parse_args()
    if options.export_only:
        return export_only(options.sample)
    if (RESULTS / "results.json").exists() and not options.force:
        raise SystemExit("Test results already exist. Re-running would re-open the test set; pass --force for a replay.")
    np.random.seed(0)
    provenance = json.loads((options.sample.parent / "provenance.json").read_text())
    splits = load(options.sample)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    assert not set(train["user"]) & set(test["user"]) and not set(validation["user"]) & set(test["user"])

    print("Selecting on validation learners...")
    models, selection = {}, {}
    for baseline in (Constant(), Leitner(), Pimsleur()):
        models[baseline.name] = baseline.fit(train)
    for name in GRID:
        models[name], best, trials = select(name, train, validation)
        selection[name] = {"chosen": best, "trials": trials}

    print("Opening the held-out test learners once...")
    predictions, results = {}, {}
    for name, model in models.items():
        p_hat, h_hat = model.predict(test)
        predictions[name] = p_hat
        results[name] = evaluate(test["p"], p_hat, h_hat, test["delta_days"])
        print(f"  {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in results[name].items()))
    errors = {name: np.abs(test["p"] - p_hat) for name, p_hat in predictions.items()}
    contrasts = {f"hlr_minus_{other}": bootstrap_mae_difference(test["user"], errors["hlr"], errors[other])
                 for other in errors if other != "hlr"}

    RESULTS.mkdir(exist_ok=True)
    report = {"data": {"provenance": provenance, **{k: summarize(v) for k, v in splits.items()}},
              "selection_metric": "validation MAE", "test": results, "mae_contrasts_learner_bootstrap": contrasts,
              "hlr_theta": dict(zip(FEATURES, map(float, models["hlr"].theta))),
              "hlr_calibration": calibration(test["p"], predictions["hlr"])}
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (RESULTS / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    export(models["logistic_regression"], provenance, results["logistic_regression"])
    print(f"Wrote {RESULTS / 'results.json'} and exported production weights to {EXPORT}")


if __name__ == "__main__":
    main()
