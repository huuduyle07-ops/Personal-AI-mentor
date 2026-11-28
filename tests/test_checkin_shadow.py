import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

import bot as integration
import checkin_policy
from checkin_policy import FEATURES, ShadowLearner
from config import Config
from core import GoalStore
from scheduler import CheckinScheduler, DeliveryError
from time_utils import KST
from verification import Verification

USER, DAY = 123, "2026-11-23"  # a Monday
START = datetime(2026, 11, 23, 11, 0, tzinfo=KST)
NOON, EVENING = START.replace(hour=12), START.replace(hour=21)


@pytest.fixture
def setup(store):
    store.set_goal(USER, "Korean", 30, DAY)
    send = AsyncMock()
    scheduler = CheckinScheduler(store, USER, send, shadow=ShadowLearner(store, Verification(store)))
    scheduler.configure(USER, now=START)
    return store, scheduler, send


def tick(scheduler, when):
    asyncio.run(scheduler.tick(when))


def shadows(store):
    with store.connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM bandit_shadow ORDER BY id")]


def studied(store, when, minutes=20):
    with store.connect() as db:
        db.execute("INSERT INTO progress_events(user_id, day, minutes, created_at) VALUES(?,?,?,?)",
                   (USER, when.date().isoformat(), minutes, when.isoformat()))


def policy(store):
    with store.connect() as db:
        return checkin_policy.load(db, USER)


def test_delivered_checkin_is_logged_as_a_study_nudge(setup):
    store, scheduler, _ = setup
    tick(scheduler, NOON)
    [row] = shadows(store)
    assert (row["slot"], row["action"], row["policy_choice"], row["resolved_at"]) == ("midday", "nudge_study", "stay_silent", None)
    x = json.loads(row["context"])
    assert len(x) == len(FEATURES) and x[FEATURES.index("afternoon")] == 1.0


def test_failed_delivery_is_not_logged(store):
    store.set_goal(USER, "Korean", 30, DAY)
    scheduler = CheckinScheduler(store, USER, AsyncMock(side_effect=DeliveryError("forbidden")),
                                 shadow=ShadowLearner(store, Verification(store)))
    scheduler.configure(USER, now=START)
    tick(scheduler, NOON)
    assert shadows(store) == []


@pytest.mark.parametrize("study_at,tired,expected", [
    (NOON + timedelta(hours=1, minutes=30), False, 0.85),   # engaged after the nudge
    (None, False, -0.15),                                   # ignored nudge
    (NOON + timedelta(hours=1), True, 0.55),                # engaged, but nudged while tired: higher cost
    (NOON + timedelta(hours=4), False, -0.15),              # studied, but outside the 3-hour window
])
def test_outcome_is_scored_after_the_window_and_updates_the_bandit(setup, study_at, tired, expected):
    store, scheduler, _ = setup
    if tired:
        with store.connect() as db:
            db.execute("INSERT INTO events(user_id,day,kind,source,payload,created_at) VALUES(?,?,?,?,?,?)",
                       (USER, DAY, "obstacle_reported", "user_reported", '{"reason": "tired"}',
                        (NOON - timedelta(hours=1)).isoformat()))
    tick(scheduler, NOON)
    if study_at:
        studied(store, study_at)
    tick(scheduler, NOON + timedelta(hours=2, minutes=59))
    assert shadows(store)[0]["resolved_at"] is None  # window not over yet
    tick(scheduler, NOON + timedelta(hours=3, minutes=1))
    row = shadows(store)[0]
    assert row["reward"] == pytest.approx(expected)
    learned = policy(store)
    assert learned.updates == {"stay_silent": 0, "nudge_study": 1, "nudge_quiz": 0}
    x = json.loads(row["context"])
    assert learned.b["nudge_study"] == pytest.approx([expected * v for v in x])


def test_planned_silence_is_observed_once_at_six(setup):
    store, scheduler, send = setup
    tick(scheduler, START.replace(hour=17, minute=30))
    tick(scheduler, START.replace(hour=18, minute=5))
    tick(scheduler, START.replace(hour=18, minute=35))
    silent = [row for row in shadows(store) if row["slot"] == "silent"]
    assert len(silent) == 1 and silent[0]["action"] == "stay_silent"
    studied(store, START.replace(hour=19))
    tick(scheduler, START.replace(hour=21, minute=10))
    assert [row["reward"] for row in shadows(store) if row["slot"] == "silent"] == [1.0]
    assert policy(store).updates["stay_silent"] == 1


@pytest.mark.parametrize("change", ["disable", "pause"])
def test_no_observations_without_active_opt_in(setup, change):
    store, scheduler, _ = setup
    if change == "disable":
        scheduler.configure(USER, enabled=False, now=START)
    else:
        scheduler.pause(USER, True, START)
    tick(scheduler, NOON)
    tick(scheduler, START.replace(hour=18, minute=5))
    assert shadows(store) == []


def test_shadow_failures_never_affect_checkins(setup, monkeypatch):
    store, scheduler, send = setup
    monkeypatch.setattr(checkin_policy, "context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    tick(scheduler, NOON)
    assert send.await_count == 1 and scheduler.status(USER)["deliveries"][0]["status"] == "delivered"
    assert shadows(store) == []


def test_sends_are_identical_with_and_without_shadow_mode(tmp_path):
    def run(shadow):
        store = GoalStore(tmp_path / f"{shadow}.db")
        store.set_goal(USER, "Korean", 30, DAY)
        send = AsyncMock()
        scheduler = CheckinScheduler(store, USER, send, shadow=ShadowLearner(store, Verification(store)) if shadow else None)
        scheduler.configure(USER, now=START)
        for hour, minute in ((12, 0), (12, 30), (15, 5), (18, 10), (21, 0), (21, 30)):
            tick(scheduler, START.replace(hour=hour, minute=minute))
        return [call.args for call in send.await_args_list], [d["status"] for d in scheduler.status(USER)["deliveries"]]
    assert run(True) == run(False)


def test_policy_preview_reports_shadow_progress(setup):
    store, scheduler, _ = setup
    tick(scheduler, NOON)
    tick(scheduler, NOON + timedelta(hours=3, minutes=5))
    tick(scheduler, START.replace(hour=18, minute=5))
    text = checkin_policy.format_preview(checkin_policy.preview(store, Verification(store), USER, EVENING))
    assert "Shadow mode: 1 real outcome(s) learned, 1 waiting" in text and "(1 updates)" in text


def test_the_bot_runs_shadow_mode(store):
    pm = integration.create_bot(Config(token="synthetic", allowed_user_id=USER), store)
    assert isinstance(pm.scheduler.shadow, ShadowLearner)
