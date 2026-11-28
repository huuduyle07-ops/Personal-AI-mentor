import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

import chat
import focus
from config import Config
from dm_intents import CHAT, QUERY, REPORT, Interpretation, action
from scheduler import CheckinScheduler
from time_utils import KST
from verification import Verification

USER = 123
NOW = datetime(2026, 11, 22, 19, 0, tzinfo=KST)
DAY = NOW.date().isoformat()


def rows(store):
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in tables}


def summary(store, now=NOW):
    return focus.format_focus(focus.snapshot(store, Verification(store), USER, now))


def answer_words(store, schedule):
    """Quiz at (hours after a start time, correct?) points; returns the Verification used."""
    quiz = Verification(store)
    start = NOW - timedelta(days=10)
    for hours, correct in schedule:
        when = start + timedelta(hours=hours)
        day = when.date().isoformat()
        if not store.summary(USER, day)["minutes"]:
            store.report(USER, day, 30)
        task = quiz.start(USER, day, now=when)
        quiz.answer(USER, day, task["id"], task["meaning"] if correct else "wrong", now=when)
    return quiz


# ---- read-only -------------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["empty", "goal_without_todays_plan", "busy"])
def test_focus_creates_no_rows(store, state):
    if state != "empty":
        store.set_goal(USER, "Korean", 30, "2026-11-20")  # no daily_goals row for today
        store.report(USER, "2026-11-21", 30)
    if state == "busy":
        answer_words(store, [(0, True), (1, False), (30, True)])
        CheckinScheduler(store, USER, AsyncMock()).configure(USER, now=NOW - timedelta(days=1))
    before = rows(store)
    summary(store)
    assert rows(store) == before


def test_natural_question_uses_focus_not_the_writing_progress_path(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    session.conversation.adapter.interpret = AsyncMock(return_value=Interpretation([action(QUERY, day="today")], "ollama"))
    store.set_goal(USER, "Korean", 30, "2026-11-20")
    with store.connect() as db:
        plans = db.execute("SELECT COUNT(*) FROM daily_goals").fetchone()[0]
    reply = asyncio.run(session.handle("what should I focus on today?"))
    assert reply.startswith("Today's focus") and "Suggestion:" in reply
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM daily_goals").fetchone()[0] == plans  # QUERY would have added one


@pytest.mark.parametrize("text,intent", [("what should I focus on today?", CHAT), ("오늘 뭐부터 할까?", QUERY),
                                         ("hôm nay nên học gì", QUERY)])
def test_ways_to_ask(store, monkeypatch, text, intent):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    session.conversation.adapter.interpret = AsyncMock(return_value=Interpretation(
        [action(intent, **({"day": "today"} if intent == QUERY else {}))], "ollama"))
    assert asyncio.run(session.handle(text)).startswith("Today's focus")


def test_messages_that_record_something_are_not_hijacked(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    store.set_goal(USER, "Korean", 30, DAY)
    session.conversation.adapter.interpret = AsyncMock(return_value=Interpretation([action(REPORT, minutes=20)], "ollama"))
    reply = asyncio.run(session.handle("I focused on grammar for 20 minutes"))
    assert not reply.startswith("Today's focus") and "Did I understand" in reply


def test_focus_command_in_terminal_chat(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(), store, USER)
    store.set_goal(USER, "Korean", 30, "2026-11-20")
    before = rows(store)
    reply = asyncio.run(session.handle("/focus"))
    after = rows(store)
    assert reply.startswith("Today's focus")
    assert {k: v for k, v in after.items() if k != "events"} == {k: v for k, v in before.items() if k != "events"}
    assert len(after["events"]) == len(before["events"]) + 1  # only the routing audit every command records


# ---- formatting ------------------------------------------------------------------------------

def test_cold_start_formatting(store):
    text = summary(store)
    assert "• Study: no daily goal yet." in text
    assert "• Words: 0 due · 0 resting · 0 of 40 seen · next up: 물 (new word)" in text
    assert "Duolingo population model (0 of 5 genuine reviews needed to personalize)" in text
    assert "• Check-ins: off (shadow-mode learning starts" in text
    assert "Suggestion: Whenever you like, set a daily goal" in text


def test_personalized_formatting_with_due_and_resting_words(store):
    store.set_goal(USER, "Korean", 30, (NOW - timedelta(days=10)).date().isoformat())
    # first sightings, then 6 genuine reviews a day apart (one wrong), then a fresh answer just now
    schedule = [(0, True), (0.5, True)] + [(24 * d, d % 3 != 0) for d in range(1, 7)]
    quiz = answer_words(store, schedule)
    store.report(USER, DAY, 30)
    task = quiz.start(USER, DAY, now=NOW - timedelta(minutes=10))
    quiz.answer(USER, DAY, task["id"], task["meaning"], now=NOW - timedelta(minutes=10))
    f = focus.snapshot(store, quiz, USER, NOW)
    assert f["model"] == "personalized" and f["reviews"] >= 5
    assert f["resting"] == 1 and f["due"] + f["resting"] + f["not_due"] == f["seen"]
    text = focus.format_focus(f)
    assert f"personalized from your {f['reviews']} reviews" in text
    assert f"{f['due']} due · 1 resting" in text and "30 / 30 min logged" in text


def test_checkin_and_shadow_status(store):
    scheduler = CheckinScheduler(store, USER, AsyncMock())
    scheduler.configure(USER, now=NOW - timedelta(days=1))
    with store.connect() as db:
        for slot, resolved, engaged in (("midday", True, 1), ("silent", True, 0), ("evening", False, None)):
            db.execute("INSERT INTO bandit_shadow(user_id, day, slot, action, policy_choice, context, decided_at, "
                       "reward, engaged, resolved_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (USER, DAY, slot, "nudge_study", "stay_silent", json.dumps([0.0] * 10), NOW.isoformat(),
                        0.85 if resolved else None, engaged, NOW.isoformat() if resolved else None))
    assert "on at 12:00 and 21:00 KST · shadow mode: 2 scored, 1 waiting, last window: no activity" in summary(store)
    scheduler.pause(USER, True, NOW)
    assert "• Check-ins: paused · shadow mode: 2 scored, 1 waiting" in summary(store)


@pytest.mark.parametrize("minutes,obstacle,due,expected", [
    (None, None, False, "No pressure: if you get a moment"),
    (10, None, False, "You're 20 minutes from today's goal, only if it fits your day."),
    (30, None, False, "Goal met and nothing is due."),
    (30, None, True, "Goal met. If you feel like it, a quick quiz would review 1 due word"),
    (10, "busy", False, "You mentioned being busy recently, so go easy"),
])
def test_suggestions_are_low_pressure_and_match_state(store, minutes, obstacle, due, expected):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    if due:
        answer_words(store, [(0, False)])  # a missed word stays due
    if minutes is not None:
        with store.connect() as db:
            db.execute("INSERT INTO progress_events(user_id, day, minutes, created_at) VALUES(?,?,?,?)",
                       (USER, DAY, minutes, NOW.isoformat()))
    if obstacle:
        with store.connect() as db:
            db.execute("INSERT INTO events(user_id,day,kind,source,payload,created_at) VALUES(?,?,?,?,?,?)",
                       (USER, DAY, "obstacle_reported", "user_reported", json.dumps({"reason": obstacle}),
                        (NOW - timedelta(hours=2)).isoformat()))
    assert f"Suggestion: {expected}" in summary(store)
