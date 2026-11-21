import sqlite3

import pytest

from core import GoalStore, daily_review
from check import inspect_database
from memory import Memory
from recovery import Recovery
from verification import Verification

DAY = "2026-11-22"
USER = 123


def test_legacy_migration_retains_records_and_is_idempotent(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE goals(user_id INTEGER PRIMARY KEY,title TEXT NOT NULL,target_minutes INTEGER NOT NULL);
            CREATE TABLE reports(user_id INTEGER NOT NULL,day TEXT NOT NULL,minutes INTEGER NOT NULL,
                PRIMARY KEY(user_id,day),FOREIGN KEY(user_id) REFERENCES goals(user_id));
            INSERT INTO goals VALUES(123,'Korean',30);
            INSERT INTO reports VALUES(123,'2026-11-21',35);
        """)
    for _ in range(2):
        store = GoalStore(path)
        data = store.summary(USER, "2026-11-21")
        assert data["minutes"] == 35 and data["target_reached"]
        with store.connect() as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 2
            assert db.execute("SELECT COUNT(*) FROM progress_events").fetchone()[0] == 1
            assert db.execute("SELECT minutes FROM reports").fetchone()[0] == 35


def test_future_database_refused(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=100")
    with pytest.raises(RuntimeError):
        GoalStore(path)


def test_health_check_is_read_only_and_does_not_reveal_records(store, tmp_path):
    store.set_goal(USER, "Synthetic private goal", 30, DAY)
    before = Memory(store).history(USER)
    health = inspect_database(store.path)
    assert health["row_counts"]["goals"] == 1 and health["schema_version"] == 2
    assert "private goal" not in str(health)
    assert Memory(store).history(USER) == before
    missing = tmp_path / "absent.db"
    with pytest.raises(ValueError):
        inspect_database(missing)
    assert not missing.exists()


def test_verification_and_report_revision_history(store):
    store.set_goal(USER, "Korean", 30, DAY)
    quiz = Verification(store)
    assert store.summary(USER, DAY)["verification_state"] == "not_reported"
    with pytest.raises(ValueError):
        quiz.start(USER, DAY)
    store.report(USER, DAY, 30)
    task = quiz.start(USER, DAY)
    assert quiz.start(USER, DAY)["id"] == task["id"]
    assert quiz.answer(USER, DAY, task["id"], "wrong")["state"] == "review_required"
    assert store.summary(USER, DAY)["verification_state"] == "review_required"
    task = quiz.start(USER, DAY)
    assert quiz.answer(USER, DAY, task["id"], " Water ")["state"] == "passed_after_support"
    assert store.summary(USER, DAY)["verification_state"] == "passed_after_support"
    with pytest.raises(ValueError):
        quiz.answer(USER, DAY, task["id"], "water")
    # Repeating a supported question cannot erase the support history.
    repeated = quiz.start(USER, DAY)
    assert quiz.answer(USER, DAY, repeated["id"], "water")["state"] == "passed_after_support"
    store.report(USER, DAY, 40)
    assert store.summary(USER, DAY)["verification_state"] == "reported"
    task = quiz.start(USER, DAY)
    assert quiz.answer(USER, DAY, task["id"], "water")["state"] == "verified"
    data = store.summary(USER, DAY)
    assert data["status"] == "reported" and data["verification_state"] == "verified"
    results = Memory(store).history(USER, day=DAY, kind="verification_result")
    assert len(results) == 4 and all(e["source"] == "quiz_result" for e in results)
    assert "does not prove study duration" in daily_review(data)


def test_quiz_ownership_and_stale_attempt(store):
    store.set_goal(USER, "Korean", 30, DAY)
    store.report(USER, DAY, 30)
    quiz = Verification(store)
    task = quiz.start(USER, DAY)
    with pytest.raises(ValueError):
        quiz.answer(999, DAY, task["id"], "water")
    store.report(USER, DAY, 31)
    with pytest.raises(ValueError, match="Progress changed"):
        quiz.answer(USER, DAY, task["id"], "water")


@pytest.mark.parametrize("decision,status", [("approve", "approved"), ("reject", "rejected"), ("cancel", "cancelled")])
def test_recovery_explicit_approval_and_daily_scope(store, decision, status):
    store.set_goal(USER, "Korean", 30, DAY)
    recovery = Recovery(store)
    proposal = recovery.propose(USER, DAY, "Unexpected work shift")
    assert store.summary(USER, DAY)["target"] == 30
    assert recovery.resolve(USER, DAY, proposal["id"], decision) == status
    data = store.summary(USER, DAY)
    assert data["original_target"] == 30
    assert data["target"] == (15 if decision == "approve" else 30)
    assert store.summary(USER, "2026-11-23")["target"] == 30
    with pytest.raises(ValueError):
        recovery.resolve(USER, DAY, proposal["id"], decision)
    assert Memory(store).history(USER, kind="recovery_" + status)


def test_recovery_stale_and_unauthorized(store):
    store.set_goal(USER, "Korean", 30, DAY)
    recovery = Recovery(store)
    first = recovery.propose(USER, DAY, "Work")
    second = recovery.propose(USER, DAY, "New obligation")
    with pytest.raises(ValueError):
        recovery.resolve(USER, DAY, first["id"], "approve")
    with pytest.raises(ValueError):
        recovery.resolve(999, DAY, second["id"], "approve")
    with pytest.raises(ValueError):
        recovery.resolve(USER, "2026-11-23", second["id"], "approve")
    store.set_goal(USER, "Korean", 40, DAY)
    with pytest.raises(ValueError, match="plan changed"):
        recovery.resolve(USER, DAY, second["id"], "approve")
    assert store.summary(USER, DAY)["target"] == 40


def test_history_preserves_old_daily_target_and_all_reports(store):
    store.set_goal(USER, "Korean", 30, DAY)
    store.report(USER, DAY, 30)
    store.report(USER, DAY, 40)
    store.set_goal(USER, "Reading", 60, "2026-11-23")
    data = GoalStore(store.path).summary(USER, DAY)
    assert (data["title"], data["target"], data["minutes"]) == ("Korean", 30, 40)
    events = Memory(store).history(USER, DAY, "progress_reported")
    assert len(events) == 2
    assert Memory(store).history(999) == []
    assert "no conclusion" in daily_review(store.summary(USER, "2026-11-23"))


@pytest.mark.parametrize("minutes", [-1, 1441, True, 1.5, "30", None])
def test_invalid_reports(store, minutes):
    store.set_goal(USER, "Korean", 30, DAY)
    with pytest.raises(ValueError):
        store.report(USER, DAY, minutes)


@pytest.mark.parametrize("day", ["2026-02-30", "22-11-2026", "2026-1-1", "garbage"])
def test_invalid_days(store, day):
    with pytest.raises(ValueError):
        store.report(USER, day, 30)
