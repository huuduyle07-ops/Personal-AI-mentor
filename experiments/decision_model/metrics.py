"""Conditional proper scoring rules and descriptive calibration diagnostics."""
import numpy as np
from sklearn.metrics import average_precision_score


def head_metrics(target, probability):
    if len(target) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "pr_auc_ap": None, "ece": None, "bins": []}
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    bins = []
    for index in range(10):
        selected = (probability >= index / 10) & (probability < (index + 1) / 10)
        if selected.any():
            bins.append({"bin": index, "n": int(selected.sum()), "mean_probability": float(probability[selected].mean()),
                         "observed_rate": float(target[selected].mean())})
    return {"n": len(target), "positive_rate": float(target.mean()), "mean_probability": float(probability.mean()),
            "brier": float(np.mean((probability - target)**2)),
            "log_loss": float(-np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))),
            "pr_auc_ap": float(average_precision_score(target, probability)) if len(np.unique(target)) == 2 else None,
            "ece": sum(row["n"] * abs(row["mean_probability"] - row["observed_rate"]) for row in bins) / len(target),
            "bins": bins}


def evaluate_predictions(targets, masks, predictions):
    if predictions.shape != targets.shape or not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
        raise ValueError("Expected finite probabilities shaped like the two targets")
    return {name: head_metrics(targets[masks[:, index].astype(bool), index], predictions[masks[:, index].astype(bool), index])
            for index, name in enumerate(("A_report", "B_completion_given_report"))}


def selection_loss(targets, masks, probabilities):
    report = evaluate_predictions(targets, masks, probabilities)
    values = [head["log_loss"] for head in report.values() if head["n"]]
    return float(np.mean(values))
