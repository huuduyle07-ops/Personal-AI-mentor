"""Course-algorithm baselines for the forgetting curve: KNN, SVM, decision tree and random forest.

Same data, learner split and features as run.py, so the numbers are directly comparable with the logistic
regression the bot uses. Every model predicts p_recall from [sqrt(1+right), sqrt(1+wrong), log(1+days)].

    uv run --locked --extra research python -m experiments.forgetting_curve.classic select   # validation only
    uv run --locked --extra research python -m experiments.forgetting_curve.classic test     # opens test once

How each algorithm turns into a probability of recall:
  KNN            mean p_recall of the k nearest training sessions (KNeighborsRegressor), features standardized
  SVM            support vector regression with an RBF kernel on p_recall, clipped to [0, 1], features
                 standardized; trained on a random subsample because kernel SVMs scale about quadratically
  Decision tree  regression tree on p_recall: each leaf predicts the mean recall of its training sessions
  Random forest  the average of many such trees, each fit on a bootstrap sample with 2 of 3 features per split
  Logistic       the production model (models.LogisticRegression), refit with its already-selected setting

Selection uses validation log loss, a proper scoring rule. The original run selected by MAE, which rewarded
predicting "remembered" for everything (see REPORT.md); that lesson is applied here.

Lecture 4's classification metrics are also reported for the task "detect a forgotten word" (a session with
p_recall < 0.5). Each model's decision threshold is chosen on validation to maximize F1 for that class.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .data import ARTIFACTS, load
from .models import MAX_P, MIN_P, Constant, LogisticRegression, auc, bootstrap_mae_difference

RESULTS = Path(__file__).with_name("results")
SELECTION_FILE, TEST_FILE = RESULTS / "classic_selection.json", RESULTS / "classic_test.json"
SEED = 0
KNN_TRAIN, SVM_TRAIN = 200_000, 5_000  # training subsample sizes for the two distance/kernel methods
GRID = {
    "knn": {"k": [25, 100, 400, 1600]},
    "svm": {"C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [3, 6, 10, 14]},
    "random_forest": {"max_depth": [6, 10, 14]},
}


def design(data):
    return np.column_stack([np.sqrt(1 + data["right"]), np.sqrt(1 + data["wrong"]), np.log1p(data["delta_days"])])


def subsample(data, size, seed=SEED):
    if len(data["p"]) <= size:
        return data
    pick = np.random.default_rng(seed).choice(len(data["p"]), size, replace=False)
    return {key: value[pick] for key, value in data.items()}


class SklearnRecall:
    """A scikit-learn regressor on p_recall, behind the same fit/predict interface as models.py."""

    def __init__(self, name, **params):
        self.name, self.params = name, params

    def _estimator(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor

        if self.name == "knn":
            return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=self.params["k"], n_jobs=-1))
        if self.name == "svm":
            return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=self.params["C"], epsilon=0.05,
                                                       gamma="scale", cache_size=1000))
        if self.name == "decision_tree":
            return DecisionTreeRegressor(max_depth=self.params["max_depth"], min_samples_leaf=200, random_state=SEED)
        if self.name == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=self.params["max_depth"], min_samples_leaf=200,
                                         max_features=2 / 3, n_jobs=-1, random_state=SEED)
        raise ValueError(self.name)

    def fit(self, data):
        size = {"knn": KNN_TRAIN, "svm": SVM_TRAIN}.get(self.name)
        data = subsample(data, size) if size else data
        self.model = self._estimator().fit(design(data), data["p"])
        self.trained_on = len(data["p"])
        return self

    def predict(self, data):
        return np.clip(self.model.predict(design(data)), MIN_P, MAX_P)


def predict(model, data):
    out = model.predict(data)
    return out[0] if isinstance(out, tuple) else out  # models.py models return (p_hat, half_life)


def log_losses(p, q):
    """Per-session cross-entropy for a fractional target p (the share of correct answers)."""
    q = np.clip(q, MIN_P, MAX_P)
    return -(p * np.log(q) + (1 - p) * np.log(1 - q))


def forgot_threshold(p, q):
    """The cut-off on predicted recall that maximizes F1 for detecting forgotten words (validation only)."""
    forgot = p < 0.5
    best = (-1.0, 0.5)
    for threshold in np.unique(np.quantile(q, np.linspace(0.005, 0.5, 100))):
        flagged = q < threshold
        tp = np.sum(flagged & forgot)
        if tp:
            precision, recall = tp / flagged.sum(), tp / forgot.sum()
            best = max(best, (2 * precision * recall / (precision + recall), float(threshold)))
    return best[1]


def classification(p, q, threshold):
    """Confusion matrix and Lecture 4 metrics for 'forgot' (p_recall < 0.5) as the positive class."""
    forgot, flagged = p < 0.5, q < threshold
    tp, fp = int(np.sum(flagged & forgot)), int(np.sum(flagged & ~forgot))
    fn, tn = int(np.sum(~flagged & forgot)), int(np.sum(~flagged & ~forgot))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(p), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def metrics(p, q):
    return {"log_loss": float(log_losses(p, q).mean()), "mse": float(np.mean((p - q) ** 2)),
            "mae": float(np.mean(np.abs(p - q))), "auc": auc(p < 0.5, -q)}  # AUC for detecting forgetting


def logistic_params():
    return json.loads((RESULTS / "selection.json").read_text())["logistic_regression"]["chosen"]["params"]


def fit_all(train, chosen):
    models = {"constant_mean": Constant().fit(train), "logistic_regression": LogisticRegression(**logistic_params()).fit(train)}
    for name, params in chosen.items():
        models[name] = SklearnRecall(name, **params).fit(train)
    return models


def select(sample):
    splits = load(sample)
    train, validation = splits["train"], splits["validation"]
    out = {"selection_metric": "validation log loss", "knn_train_rows": KNN_TRAIN, "svm_train_rows": SVM_TRAIN,
           "models": {}}
    for name, grid in GRID.items():
        trials = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            start = time.time()
            q = SklearnRecall(name, **params).fit(train).predict(validation)
            trials.append({"params": params, "validation": metrics(validation["p"], q),
                           "seconds": round(time.time() - start, 1)})
            print(f"  {name} {params}: validation log loss {trials[-1]['validation']['log_loss']:.4f} "
                  f"({trials[-1]['seconds']} s)")
        best = min(trials, key=lambda trial: trial["validation"]["log_loss"])
        out["models"][name] = {"chosen": best, "trials": trials}
    models = fit_all(train, {name: entry["chosen"]["params"] for name, entry in out["models"].items()})
    out["thresholds"] = {name: forgot_threshold(validation["p"], predict(model, validation))
                         for name, model in models.items()}
    RESULTS.mkdir(exist_ok=True)
    SELECTION_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {SELECTION_FILE}")


def test(sample):
    if TEST_FILE.exists():
        raise SystemExit(f"{TEST_FILE} exists: the test learners were already scored for these models.")
    selection = json.loads(SELECTION_FILE.read_text())
    chosen = {name: entry["chosen"]["params"] for name, entry in selection["models"].items()}
    splits = load(sample)
    train, test_split = splits["train"], splits["test"]
    assert not set(train["user"]) & set(test_split["user"])
    models = fit_all(train, chosen)
    p, results, losses = test_split["p"], {}, {}
    for name, model in models.items():
        start = time.time()
        q = predict(model, test_split)
        losses[name] = log_losses(p, q)
        results[name] = {**metrics(p, q), "forgot_detection": classification(p, q, selection["thresholds"][name]),
                         "predict_seconds": round(time.time() - start, 2)}
        print(f"  {name}: log loss {results[name]['log_loss']:.4f}, AUC {results[name]['auc']:.3f}, "
              f"F1(forgot) {results[name]['forgot_detection']['f1']:.3f}")
    contrasts = {f"{name}_minus_logistic": bootstrap_mae_difference(test_split["user"], losses[name],
                                                                    losses["logistic_regression"])
                 for name in losses if name != "logistic_regression"}
    report = {"chosen": chosen, "logistic_params": logistic_params(), "test_learners": int(len(np.unique(test_split["user"]))),
              "test_rows": int(len(p)), "forgot_rate": float(np.mean(p < 0.5)), "test": results,
              "log_loss_contrasts_learner_bootstrap": contrasts}
    TEST_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {TEST_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["select", "test"])
    parser.add_argument("--sample", type=Path, default=ARTIFACTS / "sample.csv")
    options = parser.parse_args()
    (select if options.phase == "select" else test)(options.sample)


if __name__ == "__main__":
    main()
