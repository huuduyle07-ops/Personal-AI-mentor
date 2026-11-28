import asyncio
import json
import math
from datetime import datetime, timedelta

import pytest

import chat
import checkin_policy
from checkin_policy import ACTIONS, FEATURES, LinUCB
from config import Config
from personalize import _solve
from time_utils import KST
from verification import Verification

USER = 123
SATURDAY_EVENING = datetime(2026, 11, 28, 19, 0, tzinfo=KST)
DAY = SATURDAY_EVENING.date().isoformat()


def obstacle(store, reason, when):
    with store.connect() as db:
        db.execute("INSERT INTO events(user_id,day,kind,source,payload,created_at) VALUES(?,?,?,?,?,?)",
                   (USER, when.date().isoformat(), "obstacle_reported", "user_reported",
                    json.dumps({"reason": reason}), when.isoformat()))


def flat(value):
    if isinstance(value, dict):
        return [v for key in sorted(value) for v in flat(value[key])]
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in flat(item)]
    return [value]


def close(a, b, tol=1e-9):
    a, b = flat(a), flat(b)
    return len(a) == len(b) and all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b))


def rows(store):
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in tables}


# ---- feature extraction ----------------------------------------------------------------------

def test_context_features_from_the_database(store):
    store.set_goal(USER, "Korean", 30, "2026-11-27")
    store.report(USER, "2026-11-27", 30)          # yesterday met: streak 1
    store.report(USER, DAY, 15)                   # half of today's goal
    obstacle(store, "tired", SATURDAY_EVENING - timedelta(hours=3))
    features, x = checkin_policy.context(store, Verification(store), USER, SATURDAY_EVENING)
    assert features == {"bias": 1.0, "afternoon": 0.0, "evening": 1.0, "night": 0.0, "weekend": 1.0,
                        "goal_progress": 0.5, "logged_today": 1.0, "streak": 1 / 14, "words_due": 0.0,
                        "tired_or_busy": 1.0}
    assert x == [features[name] for name in FEATURES]


@pytest.mark.parametrize("hours_ago,reason,expected", [(30, "busy", 0.0), (2, "other", 0.0), (2, "busy", 1.0)])
def test_only_recent_tired_or_busy_obstacles_count(store, hours_ago, reason, expected):
    obstacle(store, reason, SATURDAY_EVENING - timedelta(hours=hours_ago))
    features, _ = checkin_policy.context(store, Verification(store), USER, SATURDAY_EVENING)
    assert features["tired_or_busy"] == expected


@pytest.mark.parametrize("hour,bucket", [(8, "morning"), (13, "afternoon"), (18, "evening"), (23, "night"), (3, "night")])
def test_time_buckets(hour, bucket):
    assert checkin_policy.time_bucket(SATURDAY_EVENING.replace(hour=hour)) == bucket


def test_context_without_a_goal_and_with_due_words(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    store.report(USER, "2026-11-01", 30)
    quiz = Verification(store)
    then = datetime(2026, 11, 1, 9, tzinfo=KST)
    task = quiz.start(USER, "2026-11-01", now=then)
    quiz.answer(USER, "2026-11-01", task["id"], "wrong", now=then)  # a missed word stays due
    features, _ = checkin_policy.context(store, quiz, USER, SATURDAY_EVENING)
    assert features["words_due"] == 1 / 20 and features["logged_today"] == 0.0 and features["goal_progress"] == 0.0


# ---- LinUCB ----------------------------------------------------------------------------------

def test_update_matches_the_closed_form():
    policy, x, y = LinUCB(), [1, 0, 1, 0, 0, 0.5, 1, 0.1, 0.2, 0], [1, 1, 0, 0, 1, 0, 0, 0.3, 0, 1]
    policy.update("nudge_quiz", x, 1.0)
    policy.update("nudge_quiz", y, -0.15)
    A = [[float(i == j) + x[i] * x[j] + y[i] * y[j] for j in range(10)] for i in range(10)]
    b = [1.0 * x[i] - 0.15 * y[i] for i in range(10)]
    assert close(policy.A["nudge_quiz"], A) and close(policy.b["nudge_quiz"], b)
    assert policy.updates == {"stay_silent": 0, "nudge_study": 0, "nudge_quiz": 2}
    # Sherman–Morrison inverse equals a direct solve, and the predicted reward is theta . x
    direct = checkin_policy._inverse(A)
    assert close(policy._inverse["nudge_quiz"], direct)
    theta = _solve(A, b)
    scores = policy.scores(x)["nudge_quiz"]
    assert scores["predicted_reward"] == pytest.approx(sum(t * v for t, v in zip(theta, x)))
    bonus = policy.alpha * math.sqrt(sum(v * sum(direct[i][j] * x[j] for j in range(10)) for i, v in enumerate(x)))
    assert scores["exploration_bonus"] == pytest.approx(bonus)


def test_uncertainty_shrinks_with_updates():
    policy, x = LinUCB(), [1.0] + [0.0] * 9
    before = policy.scores(x)["nudge_study"]["exploration_bonus"]
    for _ in range(5):
        policy.update("nudge_study", x, 0.0)
    assert policy.scores(x)["nudge_study"]["exploration_bonus"] < before / 2


def test_cold_start_ties_and_stays_silent():
    action, scores = LinUCB().choose([1.0] * 10)
    assert action == "stay_silent"
    assert len({round(s["ucb"], 12) for s in scores.values()}) == 1


def test_selection_is_deterministic_and_learns():
    policy, x = LinUCB(alpha=0.3), [1, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    for _ in range(20):
        policy.update("nudge_quiz", x, 1.0)
        policy.update("nudge_study", x, -0.15)
        policy.update("stay_silent", x, 0.0)
    assert {policy.choose(x)[0] for _ in range(5)} == {"nudge_quiz"}


@pytest.mark.parametrize("action,x,reward", [("shout", [0] * 10, 1), ("nudge_quiz", [0] * 9, 1), ("nudge_quiz", [0] * 10, math.nan)])
def test_malformed_updates_are_rejected(action, x, reward):
    with pytest.raises(ValueError):
        LinUCB().update(action, x, reward)


# ---- persistence and cold start --------------------------------------------------------------

def test_learn_persists_and_reloads(store):
    x = [1, 0, 0, 1, 0, 1.0, 1, 0, 0, 0]
    checkin_policy.learn(store, USER, "nudge_study", x, 1.0)
    checkin_policy.learn(store, USER, "nudge_study", x, 0.85)
    with store.connect() as db:
        loaded = checkin_policy.load(db, USER)
    fresh = LinUCB()
    fresh.update("nudge_study", x, 1.0)
    fresh.update("nudge_study", x, 0.85)
    assert close(loaded.A, fresh.A) and close(loaded.b, fresh.b) and loaded.updates == fresh.updates
    assert close(loaded.scores(x), fresh.scores(x))
    with store.connect() as db:
        assert checkin_policy.load(db, 999).updates == {a: 0 for a in ACTIONS}  # other users start cold


@pytest.mark.parametrize("change", ["version", "dimension", "json"])
def test_mismatched_or_corrupt_state_falls_back_to_cold_start(store, change):
    checkin_policy.learn(store, USER, "nudge_quiz", [1.0] * 10, 1.0)
    with store.connect() as db:
        if change == "version":
            db.execute("UPDATE bandit_state SET version=0")
        elif change == "dimension":
            db.execute("UPDATE bandit_state SET b='[0, 0]'")
        else:
            db.execute("UPDATE bandit_state SET A='not json'")
    with store.connect() as db:
        assert sum(checkin_policy.load(db, USER).updates.values()) == 0


def test_preview_is_read_only_and_explains_cold_start(store):
    store.set_goal(USER, "Korean", 30, DAY)
    before = rows(store)
    report = checkin_policy.preview(store, Verification(store), USER, SATURDAY_EVENING)
    assert rows(store) == before
    assert report["action"] == "stay_silent" and report["updates"] == 0
    text = checkin_policy.format_preview(report)
    assert "nothing is sent" in text and "Cold start" in text and "Would choose now: stay_silent." in text
    for action in ACTIONS:
        assert f"{action}: predicted reward +0.00 + exploration bonus" in text


def test_policy_command_in_terminal_chat(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(), store, USER)
    checkin_policy.learn(store, USER, "nudge_quiz", checkin_policy.context(store, session.verification, USER)[1], 1.0)
    before = rows(store)
    reply = asyncio.run(session.handle("/policy"))
    assert "nudge_quiz: predicted reward" in reply and "(1 updates)" in reply and "Would choose now:" in reply
    after = rows(store)
    assert {k: v for k, v in after.items() if k != "events"} == {k: v for k, v in before.items() if k != "events"}
    assert len(after["events"]) == len(before["events"]) + 1  # only the usual routing audit of a command
