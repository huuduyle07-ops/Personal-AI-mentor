from __future__ import annotations

import numpy as np
from sklearn import metrics


def scores(y, p, groups=None, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    prediction = p >= threshold
    two_classes = len(np.unique(y)) == 2
    result = {"n": len(y), "positive_rate": float(y.mean()), "threshold": threshold,
              "brier": float(metrics.brier_score_loss(y, p)),
              "log_loss": float(metrics.log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])),
              "roc_auc": float(metrics.roc_auc_score(y, p)) if two_classes else None,
              "pr_auc_average_precision": float(metrics.average_precision_score(y, p)) if two_classes else None,
              "accuracy": float(metrics.accuracy_score(y, prediction)),
              "precision": float(metrics.precision_score(y, prediction, zero_division=0)),
              "recall": float(metrics.recall_score(y, prediction, zero_division=0)),
              "f1": float(metrics.f1_score(y, prediction, zero_division=0)),
              "confusion_matrix": metrics.confusion_matrix(y, prediction, labels=[0, 1]).tolist()}
    bins, ece = [], 0.0
    for i in range(10):
        selected = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        if selected.any():
            confidence, rate = float(p[selected].mean()), float(y[selected].mean())
            ece += selected.mean() * abs(confidence-rate)
            bins.append({"left": i/10, "right": (i+1)/10, "n": int(selected.sum()),
                         "mean_probability": confidence, "observed_rate": rate})
    result["calibration"] = {"ece_10_equal_width": float(ece), "bins": bins,
                             "mean_probability": float(p.mean()), "observed_rate": float(y.mean())}
    if groups is not None:
        per_learner = [scores(y[groups == g], p[groups == g], threshold=threshold) for g in np.unique(groups)]
        result["evaluated_learners"] = len(per_learner)
        result["learner_macro"] = {}
        for name in ("brier", "log_loss", "roc_auc", "pr_auc_average_precision", "accuracy", "f1"):
            values = [r[name] for r in per_learner if r[name] is not None]
            result["learner_macro"][name] = {"mean": float(np.mean(values)) if values else None,
                                              "defined_learners": len(values)}
    return result


def learner_bootstrap_delta(y, p, reference, groups, seed=42):
    # Paired learner-weighted differences. Negative favors p. No interaction independence claim.
    delta = np.asarray([np.mean((p[groups == g]-y[groups == g])**2) -
                        np.mean((reference[groups == g]-y[groups == g])**2) for g in np.unique(groups)])
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
    return {"learner_macro_brier_delta": float(delta.mean()),
            "paired_learner_bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
            "bootstrap_draws": 2000, "independent_units": len(delta)}
