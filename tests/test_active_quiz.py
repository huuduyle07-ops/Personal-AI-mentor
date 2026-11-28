import asyncio
import hashlib
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import active_quiz
import chat
import confusion
import focus
import personalize
from active_quiz import ActiveQuiz
from config import Config
from dm_intents import CLARIFY, QUIZ, Interpretation, action
from time_utils import KST
from verification import Verification, format_recall_report

USER = 123
NOW = datetime(2026, 11, 22, 19, 0, tzinfo=KST)
DAY = NOW.date().isoformat()
ROOT = Path(__file__).resolve().parents[1]


def rows(store):
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in tables}


def flashcards(store, answers, start=NOW - timedelta(days=3)):
    """Answer flashcards in order: each item is the typed answer, or True for the right meaning."""
    quiz = Verification(store)
    for i, typed in enumerate(answers):
        when = start + timedelta(minutes=10 * i)
        day = when.date().isoformat()
        if not store.summary(USER, day)["minutes"]:
            store.report(USER, day, 30)
        task = quiz.start(USER, day, now=when)
        quiz.answer(USER, day, task["id"], task["meaning"] if typed is True else typed, now=when)
    return quiz


def ask_active(store, quiz, now=NOW):
    return ActiveQuiz(store, quiz).start(USER, DAY, now=now)


@pytest.fixture
def seen(store):
    """Goal set and the first three deck words (물, 학교, 사람) answered correctly, three days ago."""
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    return flashcards(store, [True, True, True])


# ---- jamo similarity and Hangul grading ------------------------------------------------------

def test_jamo_similarity():
    assert confusion.jamo("학교") == "학교" and len(confusion.jamo("학교")) == 5
    assert confusion.similarity("학교", "학교") == 1.0
    assert confusion.similarity("나무", "나라") == pytest.approx(0.5)   # ㄴㅏㅁㅜ / ㄴㅏㄹㅏ
    assert confusion.similarity("문", "눈") == pytest.approx(2 / 3)
    assert confusion.similarity("학교", "고양이") < confusion.LOOKALIKE_SIMILARITY


@pytest.mark.parametrize("typed", ["학교", " 학교 ", "학 교", "학교.", "학교!", "학교?", "학교 。",
                                   unicodedata.normalize("NFD", "학교")])
def test_hangul_answers_are_normalized(store, seen, typed):
    task = ask_active(store, seen)
    assert task["word"] == "물" or task["word"] in {"학교", "사람"}
    with store.connect() as db:
        db.execute("UPDATE active_attempts SET question_id='ko_school' WHERE id=?", (task["id"],))
    result = ActiveQuiz(store, seen).answer(USER, DAY, task["id"], typed, now=NOW)
    assert result["correct"] and result["state"] == "verified"


@pytest.mark.parametrize("typed,given", [("학생", "학생"), ("hakgyo", "hakgyo"), (active_quiz.DONT_KNOW, None)])
def test_wrong_answers_are_recorded_with_what_was_typed(store, seen, typed, given):
    task = ask_active(store, seen)
    result = ActiveQuiz(store, seen).answer(USER, DAY, task["id"], typed, now=NOW)
    assert not result["correct"] and result["given"] == given
    with store.connect() as db:
        row = db.execute("SELECT mode, state, given FROM active_attempts WHERE id=?", (task["id"],)).fetchone()
    assert tuple(row) == ("active", "review_required", given)


def test_close_spelling_gets_close_feedback(store, seen):
    task = ask_active(store, seen)
    near = {"물": "뭍", "학교": "핵교", "사람": "사랑"}[task["word"]]
    text = active_quiz.feedback(ActiveQuiz(store, seen).answer(USER, DAY, task["id"], near, now=NOW))
    assert text.startswith("Close:") and task["word"] in text


def test_only_words_already_seen_are_asked(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    with pytest.raises(ValueError, match="already met"):
        ask_active(store, Verification(store))
    assert rows(store)["active_attempts"] == []


def test_due_words_first_and_recent_practice_is_skipped(store, seen):
    quiz = ActiveQuiz(store, seen)
    first = quiz.start(USER, DAY, now=NOW)
    assert quiz.start(USER, DAY, now=NOW)["id"] == first["id"]  # a waiting quiz is shown again
    quiz.answer(USER, DAY, first["id"], first["word"], now=NOW)
    second = quiz.start(USER, DAY, now=NOW + timedelta(minutes=1))
    assert second["question_id"] != first["question_id"]


# ---- confusion pairs -------------------------------------------------------------------------

def test_meaning_confusion_from_a_wrong_flashcard_answer(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    # deck order: 물 water, 학교 school, 사람 person; answer 학교 with "water"
    flashcards(store, [True, "Water"])
    [pair] = confusion.pairs(store, USER)
    assert (pair["word"], pair["other"], pair["kind"], pair["count"]) == ("학교", "물", "meaning", 1)
    assert confusion.contrast(pair) == "학교 = school, 물 = water"


def test_production_confusion_from_an_active_quiz(store, seen):
    quiz = ActiveQuiz(store, seen)
    task = quiz.start(USER, DAY, now=NOW)
    other = next(word for word in ("물", "학교", "사람") if word != task["word"])
    result = quiz.answer(USER, DAY, task["id"], other, now=NOW)
    [pair] = confusion.pairs(store, USER)
    assert (pair["word"], pair["other"], pair["kind"]) == (task["word"], other, "production")
    # the very miss that created the pair already gets the side-by-side contrast
    assert result["contrast"] == confusion.contrast(pair)
    assert f"Don't mix them up: {task['word']} = " in active_quiz.feedback(result)


def test_lookalike_pair_needs_repeated_misses(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    # 학교 (school) and 학생 (student) are 50% similar in jamo
    with store.connect() as db:
        for i, (question_id, state) in enumerate([("ko_student", "verified"), ("ko_school", "review_required")]):
            db.execute("INSERT INTO active_attempts(user_id,day,question_id,state,given,created_at,answered_at) "
                       "VALUES(?,?,?,?,?,?,?)", (USER, DAY, question_id, state, None, NOW.isoformat(), NOW.isoformat()))
    assert confusion.pairs(store, USER) == []  # one miss is not a pattern
    with store.connect() as db:
        db.execute("INSERT INTO active_attempts(user_id,day,question_id,state,given,created_at,answered_at) "
                   "VALUES(?,?,?,?,?,?,?)", (USER, DAY, "ko_school", "review_required", "학셍", NOW.isoformat(), NOW.isoformat()))
    [pair] = confusion.pairs(store, USER)
    assert (pair["word"], pair["other"], pair["kind"], pair["count"]) == ("학교", "학생", "look-alike", 2)
    assert "missed 2 times and looks 50% like 학생" in confusion.describe(pair)


def test_pairs_are_ranked_and_listed_once(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    flashcards(store, [True, "water", True, "school"])  # 학교 -> water (then retried), 사람 -> school
    with store.connect() as db:
        db.execute("INSERT INTO events(user_id,day,kind,source,payload,created_at) VALUES(?,?,?,?,?,?)",
                   (USER, DAY, "verification_result", "quiz_result",
                    '{"question_id": "ko_school", "state": "review_required", "given": "water"}', NOW.isoformat()))
    found = confusion.pairs(store, USER)
    assert [(p["word"], p["other"], p["count"]) for p in found] == [("학교", "물", 2), ("사람", "학교", 1)]


def test_reading_pairs_writes_nothing(store, seen):
    flashcards(store, ["water", "school"], start=NOW - timedelta(days=1))
    before = rows(store)
    confusion.pairs(store, USER)
    focus.format_focus(focus.snapshot(store, seen, USER, NOW))
    format_recall_report(seen.recall_report(USER, now=NOW), confusion.format_pairs(confusion.pairs(store, USER)))
    assert rows(store) == before


def test_pairs_show_in_recall_and_focus(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    quiz = flashcards(store, [True, "water"])
    text = format_recall_report(quiz.recall_report(USER, now=NOW), confusion.format_pairs(confusion.pairs(store, USER)))
    assert "Often mixed up:\n• 학교 = school, 물 = water: you answered 학교 with the meaning of 물" in text
    assert "• Mix-ups: 학교 = school, 물 = water (try “quiz me in Korean”)" in focus.format_focus(
        focus.snapshot(store, quiz, USER, NOW))


def test_no_pairs_means_no_extra_lines(store, seen):
    assert "Often mixed up" not in format_recall_report(seen.recall_report(USER, now=NOW),
                                                        confusion.format_pairs(confusion.pairs(store, USER)))
    assert "Mix-ups" not in focus.format_focus(focus.snapshot(store, seen, USER, NOW))


# ---- isolation from the recognition history and personalization -----------------------------

def test_personalize_module_is_unchanged():
    assert hashlib.sha256((ROOT / "personalize.py").read_bytes()).hexdigest() == \
        "c084bdad7617703a79a613f35cef936319ed44a09501444220ec8481c4446107"


def test_active_quizzes_never_reach_recall_events_or_scheduling(store):
    store.set_goal(USER, "Korean", 30, "2026-11-01")
    quiz = flashcards(store, [True, True, True] + [True, "wrong", True, True, True, "wrong"],
                      start=NOW - timedelta(days=6))
    with store.connect() as db:
        events = personalize.recall_events(db, USER)
        weights = personalize.personal_model(db, USER)[0].weights
        history = {k: tuple(v) for k, v in quiz._history(db, USER, quiz_deck(db)).items()}
    report = quiz.recall_report(USER, now=NOW)
    summary = store.summary(USER, DAY)
    active = ActiveQuiz(store, quiz)
    for i in range(6):  # a mix of right, wrong and "don't know"
        task = active.start(USER, DAY, now=NOW + timedelta(hours=2 * i))
        typed = [task["word"], "틀림", active_quiz.DONT_KNOW][i % 3]
        active.answer(USER, DAY, task["id"], typed, now=NOW + timedelta(hours=2 * i))
    with store.connect() as db:
        assert personalize.recall_events(db, USER) == events
        assert personalize.personal_model(db, USER)[0].weights == weights
        assert {k: tuple(v) for k, v in quiz._history(db, USER, quiz_deck(db)).items()} == history
        modes = {row[0] for row in db.execute("SELECT DISTINCT mode FROM active_attempts")}
        kinds = [row[0] for row in db.execute("SELECT kind FROM events WHERE json_extract(payload,'$.mode')='active'")]
    assert quiz.recall_report(USER, now=NOW) == report
    assert store.summary(USER, DAY)["verification_state"] == summary["verification_state"]
    assert modes == {"active"} and set(kinds) == {"active_quiz_offered", "active_quiz_result"} and len(kinds) == 12


def test_mode_column_only_accepts_active(store):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        with store.connect() as db:
            db.execute("INSERT INTO active_attempts(user_id,day,question_id,mode,created_at) VALUES(?,?,?,?,?)",
                       (USER, DAY, "ko_water", "passive", NOW.isoformat()))


def quiz_deck(db):
    from verification import user_deck
    return user_deck(db, USER)


# ---- conversation, terminal chat -------------------------------------------------------------

def session(store, monkeypatch, intent=QUIZ):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    local = chat.LocalChat(Config(ollama_enabled=True), store, USER)
    local.conversation.adapter.interpret = AsyncMock(return_value=Interpretation([action(intent)], "ollama"))
    return local


@pytest.mark.parametrize("text,intent", [("quiz me in Korean", QUIZ), ("spelling quiz please", CLARIFY),
                                         ("한글로 퀴즈 내줘", QUIZ), ("kiểm tra chính tả", QUIZ)])
def test_natural_requests_start_an_active_quiz(store, seen, monkeypatch, text, intent):
    reply = asyncio.run(session(store, monkeypatch, intent).handle(text))
    assert "Hangul" in reply or "한글로" in reply or "chữ Hàn" in reply
    assert len(rows(store)["active_attempts"]) == 1


def test_typed_hangul_is_graded_and_flashcards_still_take_english(store, seen, monkeypatch):
    local = session(store, monkeypatch)
    local.conversation.today = lambda: DAY
    monkeypatch.setattr("conversation.now_kst", lambda: NOW)
    monkeypatch.setattr("active_quiz.now_kst", lambda: NOW)
    asyncio.run(local.handle("/quiz active"))
    [attempt] = rows(store)["active_attempts"]
    deck = {"ko_water": "물", "ko_school": "학교", "ko_person": "사람"}
    reply = asyncio.run(local.handle(deck[attempt["question_id"]] + "."))
    assert reply.startswith("Correct!")
    # a flashcard waiting for English is still answered in English, even with an active quiz started later
    store.report(USER, DAY, 30)
    asyncio.run(local.handle("/quiz"))
    asyncio.run(local.handle("/quiz active"))
    reply = asyncio.run(local.handle("water"))
    assert "water" in reply.lower() and "Hangul" not in reply


def test_terminal_quiz_active_command(store, seen, monkeypatch):
    reply = asyncio.run(session(store, monkeypatch).handle("/quiz active"))
    assert reply.startswith("How do you write “") and "in Korean? Type it in Hangul." in reply


def test_quiz_without_a_logged_total_explains_and_offers_active_practice(store, seen, monkeypatch):
    local = session(store, monkeypatch)
    before = rows(store)
    reply = asyncio.run(local.handle("quiz me"))
    assert "tell me today's total first" in reply and "quiz me in Korean" in reply
    after = rows(store)
    assert after["verification_attempts"] == before["verification_attempts"]
    # the suggested active quiz works without a log
    assert "Type it in Hangul" in asyncio.run(local.handle("quiz me in Korean"))
