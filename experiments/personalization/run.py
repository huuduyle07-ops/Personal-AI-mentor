"""Population vs personalized recall models, scored by next-review (prequential) validation.

Every prediction for review i of a learner uses only that learner's reviews before i. Personalized
weights are refitted after every review for the first 30, then every STEP reviews. Phases:

  select  choose prior strength and the cold-start threshold on synthetic VALIDATION learners
  test    score synthetic TEST learners once, using the constants committed to personalize.py
  real    score your own history in data/accountability.db (read-only)

    uv run --locked --extra research python -m experiments.personalization.run select
    uv run --locked --extra research python -m experiments.personalization.run test
    uv run --locked --extra research python -m experiments.personalization.run real
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

import personalize
from experiments.forgetting_curve.models import auc
from recall_model import RecallModel

from .models import HalfLife, Logistic
from .synthetic import cohort

RESULTS = Path(__file__).with_name("results")
FAMILIES = {"logistic": Logistic, "hlr": HalfLife}
STRENGTHS = [1.0, 3.0, 10.0, 30.0, 100.0]
THRESHOLDS = [0, 5, 10, 20, 30]
BUCKETS = [(0, 10), (10, 30), (30, 100), (100, 10**9)]
STEP = 10
POOL, VALIDATION, TEST = (1, 200, 300), (2, 60, 150), (3, 100, 200)  # (seed, learners, reviews each)


def prequential(events, family, prior, strength, min_events):
    """Predicted recall for each review using only earlier reviews; population and personalized."""
    weights, population, personal = np.asarray(prior, float), [], []
    for i, event in enumerate(events):
        # Refit after every review early on (where the cold-start threshold is decided), then every STEP.
        if i and (i < 30 or i % STEP == 0):
            weights = family.personalize(events[:i], prior, strength)
        population.append(family.predict([event], prior)[0])
        personal.append(family.predict([event], weights if i >= min_events else prior)[0])
    return np.array(population), np.array(personal)


def metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "positive_rate": round(float(y.mean()), 4), "mae": round(float(np.abs(y - p).mean()), 4),
            "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
            "auroc": None if len(set(y)) < 2 else round(auc(y.astype(bool), p), 4)}


def log_losses(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(learners, family, prior, strength, min_events):
    rows = {"y": [], "population": [], "personalized": [], "learner": [], "index": []}
    for learner, events in enumerate(learners):
        population, personal = prequential(events, family, prior, strength, min_events)
        rows["y"] += [e[3] for e in events]
        rows["population"] += list(population)
        rows["personalized"] += list(personal)
        rows["learner"] += [learner] * len(events)
        rows["index"] += list(range(len(events)))
    return {key: np.asarray(value) for key, value in rows.items()}


def summarize(rows, seed=0):
    y = rows["y"]
    out = {name: metrics(y, rows[name]) for name in ("population", "personalized")}
    out["by_history_length"] = {}
    for lo, hi in BUCKETS:
        mask = (rows["index"] >= lo) & (rows["index"] < hi)
        if mask.any():
            out["by_history_length"][f"{lo}-{hi - 1 if hi < 10**9 else '+'}"] = {
                name: metrics(y[mask], rows[name][mask]) for name in ("population", "personalized")}
    # Learner bootstrap of the log-loss difference (personalized minus population; negative favors personalization)
    diff = log_losses(y, rows["personalized"]) - log_losses(y, rows["population"])
    ids = np.unique(rows["learner"])
    sums, counts = np.bincount(rows["learner"], diff), np.bincount(rows["learner"])
    rng = np.random.default_rng(seed)
    boots = [sums[pick].sum() / counts[pick].sum() for pick in (rng.integers(0, len(ids), len(ids)) for _ in range(1000))]
    out["log_loss_difference"] = {"mean": round(float(diff.mean()), 4),
                                  "ci95_learner_bootstrap": [round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5])],
                                  "learners_better": int((sums / counts < 0).sum()), "learners": int(len(ids))}
    return out


def population_priors():
    pool = [event for learner in cohort(*POOL) for event in learner]
    return {name: family.fit_population(pool) for name, family in FAMILIES.items()}


def select():
    start = time.time()
    priors, validation = population_priors(), cohort(*VALIDATION)
    report = {"population_weights": {k: [round(float(x), 4) for x in v] for k, v in priors.items()}, "families": {}}
    for name, family in FAMILIES.items():
        trials = {}
        for strength in STRENGTHS:
            rows = score(validation, family, priors[name], strength, min_events=0)
            trials[strength] = float(log_losses(rows["y"], rows["personalized"]).mean())
            print(f"  {name} strength {strength}: validation log loss {trials[strength]:.4f}")
        best = min(trials, key=trials.get)
        rows = score(validation, family, priors[name], best, min_events=0)
        diff = log_losses(rows["y"], rows["personalized"]) - log_losses(rows["y"], rows["population"])
        by_count = {k: float(diff[rows["index"] >= k].mean()) for k in THRESHOLDS}
        # Smallest history length from which personalizing helps on average for all later reviews,
        # checked separately for each window [k, next k).
        windows = {k: float(diff[(rows["index"] >= k) & (rows["index"] < nxt)].mean())
                   for k, nxt in zip(THRESHOLDS, THRESHOLDS[1:] + [10**9])}
        threshold = next((k for k in THRESHOLDS if all(windows[j] <= 0 for j in THRESHOLDS if j >= k)), THRESHOLDS[-1])
        report["families"][name] = {"validation_log_loss_by_strength": trials, "chosen_strength": best,
                                    "population_validation_log_loss": float(log_losses(rows["y"], rows["population"]).mean()),
                                    "log_loss_difference_by_window": windows, "chosen_min_events": threshold}
        print(f"  {name}: strength {best}, personalize from {threshold} reviews; window differences {windows}")
    report["seconds"] = round(time.time() - start, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selection.json").write_text(json.dumps(report, indent=2) + "\n")


def test():
    selection = json.loads((RESULTS / "selection.json").read_text())
    chosen = selection["families"]["logistic"]
    if (chosen["chosen_strength"], chosen["chosen_min_events"]) != (personalize.PRIOR_STRENGTH, personalize.MIN_EVENTS):
        raise SystemExit("personalize.py constants differ from the validation selection; update them first.")
    if (RESULTS / "test.json").exists():
        raise SystemExit("Test learners were already scored; refusing to reopen them.")
    priors, learners = population_priors(), cohort(*TEST)
    report = {"protocol": {"refit_every": STEP, "test": dict(zip(("seed", "learners", "reviews_each"), TEST))}}
    for name, family in FAMILIES.items():
        settings = selection["families"][name]
        rows = score(learners, family, priors[name], settings["chosen_strength"], settings["chosen_min_events"])
        report[name] = {"strength": settings["chosen_strength"], "min_events": settings["chosen_min_events"],
                        **summarize(rows)}
        print(name, json.dumps({k: report[name][k] for k in ("population", "personalized", "log_loss_difference")}))
    (RESULTS / "test.json").write_text(json.dumps(report, indent=2) + "\n")


def real(db_path, user_id):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    events = personalize.recall_events(db, user_id)
    hlr_prior = list(json.loads((Path(__file__).parents[1] / "forgetting_curve/results/results.json")
                                .read_text())["hlr_theta"].values())
    selection = json.loads((RESULTS / "selection.json").read_text())["families"]
    report = {"review_events": len(events), "correct": int(sum(e[3] for e in events)),
              "hours_since_last_review": {"median": round(float(np.median([e[2] * 24 for e in events])), 2),
                                          "max": round(float(max(e[2] * 24 for e in events)), 2)} if events else None,
              "days_spanned": round((events[-1][4] - events[0][4]).total_seconds() / 86400, 2) if events else 0,
              "note": "Priors are the Duolingo-trained population models. Too few reviews for conclusions."}
    priors = {"logistic": RecallModel().weights, "hlr": hlr_prior}
    for name, family in FAMILIES.items():
        rows = score([events], family, priors[name], selection[name]["chosen_strength"], selection[name]["chosen_min_events"])
        report[name] = {"population": metrics(rows["y"], rows["population"]),
                        "personalized": metrics(rows["y"], rows["personalized"])} if events else None
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "real.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=["select", "test", "real"])
    parser.add_argument("--db", default="data/accountability.db")
    parser.add_argument("--user", type=int)
    options = parser.parse_args()
    if options.phase == "select":
        select()
    elif options.phase == "test":
        test()
    else:
        user = options.user
        if user is None:
            from dotenv import load_dotenv

            from config import Config
            load_dotenv()
            user = Config.from_env().allowed_user_id
        real(options.db, user)


if __name__ == "__main__":
    main()
