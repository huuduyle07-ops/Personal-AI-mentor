import math
from datetime import datetime, timedelta

import pytest

import personalize
from recall_model import RecallModel
from time_utils import KST
from verification import Verification, format_recall_report

USER, DAY = 123, "2026-11-22"
START = datetime(2026, 11, 1, 9, tzinfo=KST)
PRIOR = RecallModel().weights


def answer(quiz, task, correct, when):
    quiz.answer(USER, DAY, task["id"], task["meaning"] if correct else "wrong", now=when)


def study(store, schedule):
    """Answer quiz questions at the given (hours from START, correct) points."""
    store.set_goal(USER, "Korean", 30, DAY)
    store.report(USER, DAY, 30)
    quiz = Verification(store)
    for hours, correct in schedule:
        when = START + timedelta(hours=hours)
        answer(quiz, quiz.start(USER, DAY, now=when), correct, when)
    return quiz


def events(n, correct, days):
    return [(1, 0, days, int(correct))] * n


# ---- event extraction ------------------------------------------------------------------------

def test_first_sightings_and_immediate_retries_are_not_recall_events(store):
    # word A (first sight), word B wrong (first sight), B retried after the hint, then reviews later.
    study(store, [(0, True), (0.1, False), (0.2, True), (30, True), (60, False)])
    with store.connect() as db:
        found = personalize.recall_events(db, USER)
    assert len(found) == 2  # only the two later reviews count
    right, wrong, days, correct, _ = found[0]
    assert correct == 1 and 29.7 / 24 <= days <= 30 / 24  # gap since that word's own last answer
    assert (found[1][3], found[1][2] > 1) == (0, True)


# ---- weight updates --------------------------------------------------------------------------

def test_no_reviews_returns_the_prior_exactly():
    assert personalize.fit([], PRIOR) == pytest.approx(PRIOR)


def test_forgetting_after_long_gaps_lowers_predicted_recall():
    forgetful = personalize.fit(events(40, False, 7.0), PRIOR)
    before = RecallModel(weights=PRIOR).recall(1, 0, 7)
    after = RecallModel(weights=forgetful).recall(1, 0, 7)
    assert after < before - 0.2


def test_shrinkage_toward_the_prior_weakens_as_reviews_accumulate():
    def distance(n):
        w = personalize.fit(events(n, False, 7.0), PRIOR)
        return math.dist(w, PRIOR)
    assert distance(3) < distance(30) < distance(300)
    strong_prior = personalize.fit(events(30, False, 7.0), PRIOR, strength=1e6)
    assert strong_prior == pytest.approx(PRIOR, abs=1e-3)


def test_fit_matches_the_ridge_optimality_condition():
    data = events(20, False, 5.0) + events(20, True, 0.5)
    w = personalize.fit(data, PRIOR)
    from recall_model import features, sigmoid
    grad = [personalize.PRIOR_STRENGTH * (a - b) for a, b in zip(w, PRIOR)]
    for e in data:
        x = features(*e[:3])
        p = sigmoid(sum(a * b for a, b in zip(w, x)))
        grad = [g + (p - e[3]) * xi for g, xi in zip(grad, x)]
    assert max(abs(g) for g in grad) < 1e-6


# ---- cold start ------------------------------------------------------------------------------

def test_cold_start_uses_population_then_personalizes(store):
    quiz = study(store, [(0, True)] + [(24 * (i + 1), i % 2 == 0) for i in range(3)])
    with store.connect() as db:
        model, reviews = personalize.personal_model(db, USER)
    assert reviews < personalize.MIN_EVENTS and model.source == "population"
    for i in range(3, 12):
        when = START + timedelta(hours=24 * (i + 1))
        answer(quiz, quiz.start(USER, DAY, now=when), i % 2 == 0, when)
    with store.connect() as db:
        model, reviews = personalize.personal_model(db, USER)
    assert reviews >= personalize.MIN_EVENTS and model.source == "personalized"
    assert model.weights != pytest.approx(PRIOR)


def test_cold_start_without_population_weights_uses_the_rule(store, tmp_path):
    broken = tmp_path / "missing.json"
    with store.connect() as db:
        model, _ = personalize.personal_model(db, USER, RecallModel(broken))
    assert model.source == "fallback_rule"


# ---- half-life and ranking -------------------------------------------------------------------

def test_days_until_inverts_the_recall_curve():
    model = RecallModel(weights=[0.1, -0.4, 3.4, -0.9], source="custom")
    days = model.days_until(2, 0, 0.9)
    assert 0 < days < 1000 and model.recall(2, 0, days) == pytest.approx(0.9, abs=1e-6)
    assert model.half_life(2, 0) > days
    assert RecallModel(weights=[0, 0, 5, 0.1]).days_until(0, 0, 0.9) == math.inf  # no forgetting over time
    assert RecallModel(weights=[0, 0, -5, -1]).days_until(0, 0, 0.9) == 0.0       # already below target


def test_recall_report_ranks_due_words_by_lowest_recall(store):
    quiz = study(store, [(0, True), (0.1, False), (0.2, True), (0.3, True)])
    forgetful = RecallModel(weights=[0.1, -1.5, 2.0, -0.8], source="custom")
    quiz.model = forgetful
    report = quiz.recall_report(USER, now=START + timedelta(days=3))
    statuses = [item["status"] for item in report["words"]]
    assert statuses[0] == "due" and report["words"][0]["word"] == "학교"  # the missed word is least likely recalled
    due = [item["recall"] for item in report["words"] if item["status"] == "due"]
    assert due == sorted(due)
    assert report["next"]["word"] == "학교" and report["next"]["policy"] == "review"
    text = format_recall_report(report)
    assert "학교 (school)" in text and "Next quiz: 학교" in text and "half-life" in text


def test_recall_report_marks_just_answered_words_as_resting(store):
    quiz = study(store, [(0, True)])
    report = quiz.recall_report(USER, now=START + timedelta(minutes=5))
    assert report["words"][0]["status"] == "resting" and report["next"]["policy"] == "new_word"
    assert "personalizes after" in report["label"]
