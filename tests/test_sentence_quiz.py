import asyncio
import json
from unittest.mock import AsyncMock

import pytest

import chat
import llm_adapter
from config import Config
from dm_intents import QUERY, QUIZ, Interpretation, action
from memory import Memory

USER, DAY = 123, "2026-11-22"
GOOD = {"sentence": "아침에 물을 마셔요.", "translation": "I drink water in the morning."}


def interpretation(intent, **slots):
    return Interpretation([action(intent, **slots)], "ollama")


@pytest.fixture
def local(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    adapter = session.conversation.adapter
    adapter.interpret = AsyncMock(return_value=interpretation(QUIZ))
    adapter.transport = AsyncMock(return_value={"response": json.dumps(GOOD)})
    adapter.chat_transport = AsyncMock(return_value={"message": {"content": "hi!"}})
    say(session, "/setgoal Korean 30", "/report 30")
    return session


def say(local, *lines):
    async def run():
        return [await local.handle(line) for line in lines]
    return asyncio.run(run())


def offered(local):
    return Memory(local.store).history(USER, kind="quiz_offered")[0]["payload"]


def all_text(store):
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return " ".join(str(tuple(row)) for table in tables for row in db.execute(f"SELECT * FROM {table}"))


# ---- generation ------------------------------------------------------------------------------

def test_sentence_quiz_uses_the_scheduled_word_and_hides_the_translation(local):
    reply = say(local, "quiz me with a sentence")[0]
    assert reply.startswith("아침에 물을 마셔요.") and "What does 물 mean in this sentence?" in reply
    assert "morning" not in reply  # the translation would give the answer away
    prompt = local.conversation.adapter.transport.call_args.args[0]
    assert '"물"' in prompt["prompt"] and prompt["format"]["required"] == ["sentence", "translation"]
    event = offered(local)
    assert (event["style"], event["sentence_model"], event["question_id"]) == ("sentence", "qwen3:0.6b", "ko_water")


def test_answer_feedback_includes_the_translation_once(local):
    say(local, "quiz me with a sentence")
    reply = say(local, "water")[0]
    assert reply.startswith("Correct! 물 means water.") and "“I drink water in the morning.”" in reply
    results = Memory(local.store).history(USER, kind="verification_result")
    assert results[0]["payload"]["state"] == "verified" and results[0]["source"] == "quiz_result"
    say(local, "quiz me")
    assert "The sentence means" not in say(local, "school")[0]


@pytest.mark.parametrize("request_text", ["/quiz context", "/quiz sentence", "quiz me with a sentence",
                                          "can I get a context quiz", "문장으로 퀴즈 내줘", "đố mình bằng câu"])
def test_ways_to_ask_for_a_sentence_quiz(local, request_text):
    assert "아침에 물을 마셔요." in say(local, request_text)[0]


def test_plain_quiz_requests_do_not_call_the_generator(local):
    reply = say(local, "quiz me")[0]
    assert reply.startswith("What does 물 mean in English?")
    local.conversation.adapter.transport.assert_not_called()
    assert offered(local)["style"] == "word"


# ---- validation and fallback -----------------------------------------------------------------

@pytest.mark.parametrize("output,reason", [
    ("not json", "unavailable_or_invalid"),
    ({"sentence": "아침에 물을 마셔요."}, "malformed"),
    ({**GOOD, "note": "extra"}, "malformed"),
    ({"sentence": "I drink 물 every day.", "translation": "I drink water."}, "sentence_not_korean"),
    ({"sentence": "아침에 3잔 마셔요.", "translation": "I drink water."}, "sentence_not_korean"),
    ({"sentence": "아침에 커피를 마셔요.", "translation": "I drink water in the morning."}, "word_missing"),
    ({"sentence": "아침에 물을 마셔요.", "translation": "I drink coffee in the morning."}, "meaning_missing"),
    ({"sentence": "아침에 물을 마셔요.", "translation": "아침"}, "translation_invalid"),
    ({"sentence": "물 " * 20, "translation": "water"}, "sentence_not_korean"),
])
def test_malformed_output_falls_back_to_the_flashcard(local, output, reason):
    raw = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    local.conversation.adapter.transport = AsyncMock(return_value={"response": raw})
    reply = say(local, "quiz me with a sentence")[0]
    assert reply.startswith("I couldn't write a good sentence this time") and "What does 물 mean in English?" in reply
    event = offered(local)
    assert (event["style"], event["sentence_fallback"]) == ("word", reason)


def test_generation_errors_and_disabled_model_fall_back(local, store, monkeypatch):
    local.conversation.adapter.transport = AsyncMock(side_effect=OSError("down"))
    assert "couldn't write a good sentence" in say(local, "/quiz context")[0]
    monkeypatch.setattr(chat, "today", lambda: DAY)
    offline = chat.LocalChat(Config(ollama_enabled=False), store, USER)
    say(offline, "water")
    assert "couldn't write a good sentence" in say(offline, "/quiz context")[0]


def test_verb_stems_count_as_using_the_word():
    assert llm_adapter.validate_sentence({"sentence": "밥을 먹어요.", "translation": "I eat rice."},
                                         "먹다", {"to eat", "eat"}) == ("밥을 먹어요.", "I eat rice.")


# ---- database safety -------------------------------------------------------------------------

def test_generated_text_is_never_stored(local):
    say(local, "quiz me with a sentence", "water")
    stored = all_text(local.store)
    assert GOOD["sentence"] not in stored and GOOD["translation"] not in stored


def test_quiz_is_logged_like_a_standard_quiz(local, store, monkeypatch):
    say(local, "quiz me with a sentence")
    sentence_event = offered(local)
    monkeypatch.setattr(chat, "today", lambda: "2026-11-23")
    other = chat.LocalChat(Config(), store, USER)
    say(other, "/report 30", "/quiz")
    plain_event = Memory(store).history(USER, "2026-11-23", "quiz_offered")[0]["payload"]
    shared = {"attempt_id", "question_id", "policy", "predicted_recall", "recall_model", "review_events", "style"}
    assert shared <= set(sentence_event) and shared <= set(plain_event)


def test_nothing_is_written_when_the_quiz_cannot_start(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    session = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    session.conversation.adapter.transport = AsyncMock(return_value={"response": json.dumps(GOOD)})
    say(session, "/setgoal Korean 30")
    with store.connect() as db:
        before = db.execute("SELECT COUNT(*) FROM verification_attempts").fetchone()[0]
    assert "tell me today's total first" in say(session, "/quiz context")[0]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM verification_attempts").fetchone()[0] == before
    session.conversation.adapter.transport.assert_not_called()


def test_a_waiting_quiz_is_shown_again_without_generating(local):
    say(local, "quiz me")
    assert "What does 물 mean in English?" in say(local, "quiz me with a sentence")[0]
    local.conversation.adapter.transport.assert_not_called()


# ---- streak ----------------------------------------------------------------------------------

def test_streak_appears_in_progress_answers_and_summary(local):
    for day, minutes in (("2026-11-20", 30), ("2026-11-21", 40)):
        local.store.report(USER, day, minutes)
    local.conversation.adapter.interpret = AsyncMock(return_value=interpretation(QUERY, day="today"))
    reply = say(local, "how's my streak?")[0]
    assert "Streak: 3 days in a row meeting your goal." in reply  # 20th, 21st and today (30 of 30)
    assert "Streak: 3 days in a row meeting your goal" in say(local, "/summary")[0]


def test_streak_reporting_is_read_only(local):
    local.conversation.adapter.interpret = AsyncMock(return_value=interpretation(QUERY, day="today"))
    with local.store.connect() as db:
        before = db.execute("SELECT COUNT(*) FROM daily_goals").fetchone()[0]
    say(local, "how's my streak?")
    with local.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM daily_goals").fetchone()[0] == before
