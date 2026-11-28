import asyncio
import hashlib
from unittest.mock import AsyncMock

import pytest

import chat
import chat_context
import llm_adapter
from config import Config
from dm_intents import CHAT, CLARIFY, Interpretation, action

USER, DAY = 123, "2026-11-22"


def interpretation(intent):
    return Interpretation([action(intent)], "ollama")


@pytest.fixture
def local(store, monkeypatch):
    """Terminal chat with the AI enabled and both model endpoints mocked."""
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    adapter = session.conversation.adapter
    adapter.interpret = AsyncMock(return_value=interpretation(CHAT))
    adapter.chat_transport = AsyncMock(side_effect=lambda payload: {
        "message": {"content": f"reply {len(payload['messages'])}"}})
    return session


def say(local, *lines):
    async def run():
        return [await local.handle(line) for line in lines]
    return asyncio.run(run())


def sent(local, call=-1):
    return local.conversation.adapter.chat_transport.call_args_list[call].args[0]["messages"]


def snapshot_db(store):
    """Every table's rows, except replay receipts and the interpretation audit that every message gets."""
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: db.execute(f"SELECT * FROM {table}" + (" WHERE kind!='dm_interpretation'" if table == "events" else ""))
                .fetchall() for table in tables if table != "dm_receipts"}


def test_multi_turn_history_is_passed_as_real_turns_and_capped(local):
    replies = say(local, "hey", "how was your day", "I watched a drama", "it was sad", "you there?")
    assert replies[0] == "reply 2"  # system + this message
    messages = sent(local)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user",
                                             "assistant", "user"]
    assert [m["content"] for m in messages[1:]] == [
        "how was your day", replies[1], "I watched a drama", replies[2], "it was sad", replies[3],
        "you there?\n\n(Reply briefly in English.)"]


def test_history_includes_non_chat_turns_for_context(local):
    say(local, "/setgoal Korean 30")
    local.conversation.adapter.interpret = AsyncMock(side_effect=[interpretation(CHAT)])
    say(local, "today's total is 20 minutes")  # handled by the rules, not the chat model
    say(local, "nice")
    contents = [m["content"] for m in sent(local)]
    assert "today's total is 20 minutes" in contents and any("Recorded 20" in c for c in contents)


def test_reply_is_grounded_in_read_only_status(local):
    say(local, "/setgoal Korean 30", "/report 20")
    say(local, "how's it going")
    system = sent(local)[0]["content"]
    assert "Daily goal: Korean, 30 minutes" in system and "20 of 30 minutes logged today" in system
    assert "Streak: 0 day(s)" in system and "word(s) due for review" in system
    assert "do not tell them to study or practise Korean unless they ask" in system
    assert "Reply in English only" in system


def test_reply_language_follows_each_message(local):
    say(local, "오늘 날씨 좋다", "thanks!")
    assert "Reply in Korean only" in sent(local, 0)[0]["content"]
    assert sent(local, 0)[-1]["content"].endswith("(한국어로 짧게 대답해 주세요.)")
    assert sent(local, 1)[1]["content"] == "오늘 날씨 좋다"  # the nudge is not kept in history
    assert "Reply in English only" in sent(local, 1)[0]["content"]


def test_streak_counts_consecutive_days_meeting_the_goal(store):
    store.set_goal(USER, "Korean", 30, "2026-11-19")
    for day, minutes in (("2026-11-19", 10), ("2026-11-20", 30), ("2026-11-21", 45)):
        store.report(USER, day, minutes)
    with store.connect() as db:
        assert chat_context.streak(db, USER, DAY, 30) == 2       # today not done yet: counts back from yesterday
    store.report(USER, DAY, 30)
    with store.connect() as db:
        assert chat_context.streak(db, USER, DAY, 30) == 3


def test_chit_chat_never_modifies_the_database(local):
    say(local, "/setgoal Korean 30", "/report 20")
    before = snapshot_db(local.store)
    local.conversation.adapter.interpret = AsyncMock(side_effect=[interpretation(CHAT), interpretation(CLARIFY),
                                                                  interpretation(CHAT)])
    say(local, "lol thanks", "what should I watch tonight", "ok bye")
    assert snapshot_db(local.store) == before


def test_status_read_does_not_create_todays_plan(local):
    say(local, "/setgoal Korean 30")
    with local.store.connect() as db:
        db.execute("DELETE FROM daily_goals")
    say(local, "hello there")
    with local.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM daily_goals").fetchone()[0] == 0
    assert "nothing logged yet today" in sent(local)[0]["content"]


def test_reply_claiming_an_action_falls_back_to_the_template(local):
    local.conversation.adapter.chat_transport = AsyncMock(return_value={
        "message": {"content": "Great job! I've logged 30 minutes for you."}})
    reply = say(local, "studied a bunch today")[0]
    assert "logged" not in reply and "Would you like" in reply


def test_study_logs_and_quizzes_still_use_the_router(local):
    say(local, "/setgoal Korean 30")
    assert "Recorded 20" in say(local, "today's total is 20 minutes")[0]
    assert "물" in say(local, "quiz me")[0]
    local.conversation.adapter.chat_transport.assert_not_called()


def test_numbers_keep_the_clarification_template(local):
    local.conversation.adapter.interpret = AsyncMock(return_value=interpretation(CLARIFY))
    assert "Would you like to record" in say(local, "maybe 30 or 40 idk")[0]
    local.conversation.adapter.chat_transport.assert_not_called()


def test_without_the_model_the_old_reply_is_unchanged(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=False), store, USER)
    assert say(session, "hello")[0].startswith("Would you like to log study time")


def test_intent_prompt_is_the_benchmarked_version():
    """The chat feature must not change classification: this is the prompt scored in HELDOUT.md (v2)."""
    assert hashlib.sha256(llm_adapter.INTENT_PROMPT.encode()).hexdigest()[:12] == "d6f2449c8b8e"
