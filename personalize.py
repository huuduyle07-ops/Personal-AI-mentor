"""Adapt the population recall model to one learner's own quiz history.

Model:  p(recall) = sigmoid(w . [sqrt(1+right), sqrt(1+wrong), 1, log(1+days since last review)])

Personalization is maximum a posteriori (MAP) estimation with a Gaussian prior centred on the
population (Duolingo-trained) weights, i.e. ridge regression that shrinks toward the prior:

    minimise  sum_i BCE(y_i, p_i(w))  +  (PRIOR_STRENGTH / 2) * ||w - w_population||^2

With few reviews the prior dominates and the weights barely move; as reviews accumulate, the data
takes over. Below MIN_EVENTS reviews the population model is used unchanged (cold start).

Run `uv run python personalize.py` to see what it learns from data/accountability.db.
"""
import argparse
import json
import math
from datetime import datetime, timedelta

from recall_model import FEATURES, RecallModel, features, sigmoid

# Both chosen on synthetic validation learners in experiments/personalization (see REPORT.md).
PRIOR_STRENGTH = 10.0
MIN_EVENTS = 5
RETRY_WINDOW = timedelta(hours=1)  # an answer this soon after a miss was helped by the shown answer


def recall_events(db, user_id):
    """Chronological review events: (right_before, wrong_before, days_since_last_review, correct, when).

    A word's first answer is not a recall event (there was nothing to remember yet), and neither
    is a retry right after a miss, because the correct meaning had just been shown.
    """
    rows = db.execute(
        """SELECT question_id, state, answered_at FROM verification_attempts
        WHERE user_id=? AND state!='pending' AND answered_at IS NOT NULL ORDER BY answered_at, id""",
        (user_id,),
    ).fetchall()
    seen, events = {}, []
    for row in rows:
        correct = row["state"] in {"verified", "passed_after_support"}
        when = datetime.fromisoformat(row["answered_at"])
        past = seen.get(row["question_id"])
        if past is not None:
            gap = when - past["last"]
            if not (past["last_wrong"] and gap < RETRY_WINDOW):
                events.append((past["right"], past["wrong"], gap.total_seconds() / 86400, int(correct), when))
        past = past or {"right": 0, "wrong": 0}
        seen[row["question_id"]] = {"right": past["right"] + correct, "wrong": past["wrong"] + (not correct),
                                    "last": when, "last_wrong": not correct}
    return events


def _solve(matrix, vector):
    """Solve a small dense linear system by Gaussian elimination with partial pivoting."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def fit(events, prior, strength=PRIOR_STRENGTH, iterations=25):
    """MAP weights by Newton's method. `events` are (right, wrong, days, correct, ...) tuples."""
    if strength <= 0:
        raise ValueError("Prior strength must be positive")
    rows = [(features(e[0], e[1], e[2]), e[3]) for e in events]
    w = list(prior)
    k = len(w)
    for _ in range(iterations):
        grad = [strength * (w[j] - prior[j]) for j in range(k)]
        hess = [[strength if i == j else 0.0 for j in range(k)] for i in range(k)]
        for x, y in rows:
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, x)))
            s = p * (1 - p)
            for i in range(k):
                grad[i] += (p - y) * x[i]
                for j in range(k):
                    hess[i][j] += s * x[i] * x[j]
        step = _solve(hess, grad)
        w = [wj - sj for wj, sj in zip(w, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return w


def personal_model(db, user_id, population=None):
    """(model, review_count): personalized once there are MIN_EVENTS reviews, else the population model."""
    population = population or RecallModel()
    events = recall_events(db, user_id)
    if population.weights is None or len(events) < MIN_EVENTS:
        return population, len(events)
    return RecallModel(weights=fit(events, population.weights), source="personalized"), len(events)


def main():
    import sqlite3

    from dotenv import load_dotenv

    from config import Config

    load_dotenv()
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Show how the recall model adapts to your quiz history.")
    parser.add_argument("--db", default=config.database_path)
    parser.add_argument("--user", type=int, default=config.allowed_user_id)
    options = parser.parse_args()
    db = sqlite3.connect(f"file:{options.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    events = recall_events(db, options.user)
    population = RecallModel()
    model, _ = personal_model(db, options.user, population)
    correct = sum(e[3] for e in events)
    print(f"Review events: {len(events)} ({correct} correct, {len(events) - correct} wrong); "
          f"personalization starts at {MIN_EVENTS}.")
    if events:
        gaps = sorted(e[2] for e in events)
        print(f"Time since last review: median {gaps[len(gaps) // 2] * 24:.1f} h, max {gaps[-1] * 24:.1f} h")
    print(f"Model in use: {model.source}")
    weights = fit(events, population.weights) if events else population.weights
    print(json.dumps({"features": FEATURES, "population": [round(w, 4) for w in population.weights],
                      "personalized_fit": [round(w, 4) for w in weights],
                      "prior_strength": PRIOR_STRENGTH}, indent=2))
    if events:
        def logloss(w):
            total = 0.0
            for e in events:
                p = min(max(sigmoid(sum(a * b for a, b in zip(w, features(*e[:3])))), 1e-6), 1 - 1e-6)
                total -= e[3] * math.log(p) + (1 - e[3]) * math.log(1 - p)
            return total / len(events)
        print(f"In-sample log loss (optimistic, not an evaluation): population {logloss(population.weights):.4f}, "
              f"personalized {logloss(weights):.4f}")


if __name__ == "__main__":
    main()
