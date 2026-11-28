import json
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import personalize
from active_quiz import DONT_KNOW
from dashboard_data import Dashboard
from scheduler import CheckinScheduler
from time_utils import KST
from verification import Verification, user_deck

USER = 123
NOW = datetime(2026, 11, 22, 19, 0, tzinfo=KST)
DAY = NOW.date().isoformat()
ROOT = Path(__file__).resolve().parents[1]


def rows(store):
    with store.connect() as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {table: db.execute(f"SELECT * FROM {table}").fetchall() for table in tables}


def flashcards(store, answers, start):
    """answers: (hours after start, typed answer or True for the right meaning)."""
    quiz = Verification(store)
    for hours, typed in answers:
        when = start + timedelta(hours=hours)
        day = when.date().isoformat()
        if not store.summary(USER, day)["minutes"]:
            store.report(USER, day, 30)
        task = quiz.start(USER, day, now=when)
        quiz.answer(USER, day, task["id"], task["meaning"] if typed is True else typed, now=when)


@pytest.fixture
def board(tmp_path):
    return Dashboard(tmp_path / "dashboard.db", USER)


def populate(board):
    store = board.store
    store.set_goal(USER, "Korean", 30, "2026-11-10")
    # 물 then 학교 (answered "water"), then reviews spread over the next days
    reviews = [(12 * k, True if k % 4 else "wrong") for k in range(1, 9)]
    flashcards(store, [(0, True), (0.2, "water"), (0.3, True)] + reviews, NOW - timedelta(days=5))
    CheckinScheduler(store, USER, AsyncMock()).configure(USER, now=NOW - timedelta(days=2))
    with store.connect() as db:
        db.execute("INSERT INTO bandit_shadow(user_id, day, slot, action, policy_choice, context, decided_at, reward, "
                   "engaged, resolved_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (USER, DAY, "midday", "nudge_study", "stay_silent", json.dumps([0.0] * 10),
                    NOW.replace(hour=12).isoformat(), 0.85, 1, NOW.replace(hour=15).isoformat()))
        db.execute("INSERT INTO bandit_shadow(user_id, day, slot, action, policy_choice, context, decided_at) "
                   "VALUES(?,?,?,?,?,?,?)", (USER, DAY, "silent", "stay_silent", "stay_silent",
                                             json.dumps([0.0] * 10), NOW.replace(hour=18).isoformat()))
        db.execute("INSERT INTO events(user_id,day,kind,source,payload,created_at) VALUES(?,?,?,?,?,?)",
                   (USER, DAY, "obstacle_reported", "user_reported", '{"reason": "busy"}', NOW.isoformat()))


# ---- read-only helpers -----------------------------------------------------------------------

@pytest.mark.parametrize("state", ["empty", "goal_without_todays_plan", "populated"])
def test_read_only_helpers_create_no_rows(board, state):
    if state == "goal_without_todays_plan":
        board.store.set_goal(USER, "Korean", 30, "2026-11-10")
    if state == "populated":
        populate(board)
    before = rows(board.store)
    board.overview(NOW)
    board.memory(NOW)
    board.policy(NOW)
    board.pending_quizzes(DAY)
    board.preview_import([("사과", ["apple"]), ("물", ["water"])])
    assert rows(board.store) == before


def test_overview_kpis_and_week(board):
    populate(board)
    o = board.overview(NOW)
    assert (o["goal"], o["target"], o["min_events"]) == ("Korean", 30, personalize.MIN_EVENTS)
    assert (o["shadow"]["scored"], o["shadow"]["waiting"], o["obstacle"]) == (1, 1, "busy")
    assert [d["day"] for d in o["week"]] == [(NOW.date() - timedelta(days=k)).isoformat() for k in range(6, -1, -1)]
    assert all(d["minutes"] in (0, 30) and d["target"] == 30 for d in o["week"])
    assert o["text"].startswith("Today's focus")


def test_memory_reports_model_mode_words_and_pairs(board):
    populate(board)
    m = board.memory(NOW)
    assert m["review_events"] >= personalize.MIN_EVENTS and m["personalized"]
    assert [w["feature"] for w in m["weights"]] == ["sqrt_right", "sqrt_wrong", "bias", "log1p_days"]
    assert all(w["population"] is not None and w["in_use"] is not None for w in m["weights"])
    assert {w["status"] for w in m["words"]} <= {"due", "resting", "not due"}
    assert all(0 <= w["recall"] <= 1 for w in m["words"])
    assert [(p["word"], p["other"], p["kind"]) for p in m["pairs"]] == [("학교", "물", "meaning")]


def test_policy_shows_context_three_actions_and_history(board):
    populate(board)
    p = board.policy(NOW)
    assert [s["action"] for s in p["scores"]] == ["stay_silent", "nudge_study", "nudge_quiz"]
    assert all({"predicted_reward", "exploration_bonus", "ucb"} <= set(s) for s in p["scores"])
    assert p["features"]["tired_or_busy"] == 1.0 and p["features"]["evening"] == 1.0
    assert [(h["slot"], h["engaged"]) for h in p["history"]] == [("silent", None), ("midday", 1)]


# ---- practice isolation ----------------------------------------------------------------------

def test_active_quizzes_from_the_dashboard_stay_isolated_from_personalize(board):
    populate(board)
    store = board.store
    with store.connect() as db:
        events = personalize.recall_events(db, USER)
        weights = personalize.personal_model(db, USER)[0].weights
        history = {k: tuple(v) for k, v in board.verification._history(db, USER, user_deck(db, USER)).items()}
    flashcard_rows = rows(store)["verification_attempts"]
    memory = board.memory(NOW)
    for typed in ("correct", "틀림", DONT_KNOW, "correct"):
        task = board.start_active(DAY)
        board.answer_active(DAY, task["id"], task["word"] if typed == "correct" else typed)
    with store.connect() as db:
        assert personalize.recall_events(db, USER) == events
        assert personalize.personal_model(db, USER)[0].weights == weights
        assert {k: tuple(v) for k, v in board.verification._history(db, USER, user_deck(db, USER)).items()} == history
    after = rows(store)
    assert after["verification_attempts"] == flashcard_rows
    assert len(after["active_attempts"]) == 4 and {r["mode"] for r in after["active_attempts"]} == {"active"}
    fresh = board.memory(NOW)
    assert (fresh["weights"], fresh["review_events"], fresh["words"]) == (memory["weights"], memory["review_events"],
                                                                         memory["words"])


def test_passive_quiz_from_the_dashboard_is_the_normal_flashcard(board):
    board.store.set_goal(USER, "Korean", 30, "2026-11-10")
    with pytest.raises(ValueError, match="Report progress first"):
        board.start_passive(DAY)
    board.store.report(USER, DAY, 20)
    task = board.start_passive(DAY)
    assert board.pending_quizzes(DAY)["passive"]["id"] == task["id"]
    assert board.answer_passive(DAY, task["id"], task["meaning"])["correct"]
    assert board.pending_quizzes(DAY) == {"passive": None, "active": None}


def test_note_import_writes_only_after_confirmation(board):
    board.store.set_goal(USER, "Korean", 30, "2026-11-10")
    before = rows(board.store)
    items, skipped = board.preview_import([("사과", ["apple"]), ("포도", ["grape"]), ("물", ["water"])])
    assert items == [("사과", ["apple"]), ("포도", ["grape"])] and skipped == ["물"]
    assert rows(board.store) == before
    assert board.confirm_import([items[0]], "photo notes.jpg", DAY).startswith("Added 1 word")
    after = rows(board.store)
    assert [(r["word"], json.loads(r["answers"])) for r in after["user_vocab"]] == [("사과", ["apple"])]
    assert after["vocab_imports"] == []  # no half-finished proposal is left behind
    assert board.confirm_import([], "photo notes.jpg", DAY).startswith("Nothing selected")


# ---- packaging and the Streamlit app ---------------------------------------------------------

def test_streamlit_is_an_optional_extra():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any(dep.startswith("streamlit") for dep in project["dependencies"])
    assert any(dep.startswith("streamlit") for dep in project["optional-dependencies"]["dashboard"])


def test_streamlit_app_renders_all_tabs_without_writing(tmp_path, monkeypatch):
    pytest.importorskip("streamlit", reason="Optional dashboard extra is not installed")
    from streamlit.testing.v1 import AppTest

    board = Dashboard(tmp_path / "app.db", USER)
    populate(board)
    before = rows(board.store)
    monkeypatch.setenv("DASHBOARD_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("DASHBOARD_USER", str(USER))
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    assert not app.exception  # includes the history table with a waiting (unscored) window
    assert [tab.label for tab in app.tabs] == ["Chat", "Overview & Focus", "Memory & Confusion Pairs",
                                                "Check-in Policy (LinUCB)", "Practice & Note Import"]
    assert [m.label for m in app.metric][:3] == ["Today's minutes", "Streak", "Due words"]
    assert rows(board.store) == before
    # pressing "Start an active quiz" writes one active attempt and nothing to the flashcard history
    next(b for b in app.button if b.key == "start_active").click().run()
    after = rows(board.store)
    assert len(after["active_attempts"]) == 1 and after["verification_attempts"] == before["verification_attempts"]


def test_chat_tab_talks_to_the_bot_through_the_normal_conversation(tmp_path, monkeypatch):
    pytest.importorskip("streamlit", reason="Optional dashboard extra is not installed")
    from streamlit.testing.v1 import AppTest

    board = Dashboard(tmp_path / "chat.db", USER)
    board.store.set_goal(USER, "Korean", 30, "2026-11-10")
    monkeypatch.setenv("DASHBOARD_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("DASHBOARD_USER", str(USER))
    monkeypatch.setenv("OLLAMA_ENABLED", "false")  # rule-based replies only, no local model needed
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    app.chat_input[0].set_value("today's total is 20 minutes").run()
    assert not app.exception
    said = [m.markdown[0].value for m in app.chat_message]
    assert said[0] == "today's total is 20 minutes" and "Recorded 20 self-reported minutes" in said[1]
    with board.store.connect() as db:
        assert [r["minutes"] for r in db.execute("SELECT minutes FROM progress_events")] == [20]
    # the Overview tab, rendered after the chat in the same run, already shows the new total
    assert app.metric[0].value == "20"
    next(b for b in app.button if b.key == "shortcut_how am I doing?").click().run()
    assert "You reported 20 minutes" in app.chat_message[-1].markdown[0].value
    assert len(app.chat_message) == 4


# ---- pictures in the chat --------------------------------------------------------------------

def test_chat_photo_is_imported_only_after_yes(tmp_path):
    from config import Config
    from dashboard_data import ChatSession
    from study_notes import NoteImporter

    session = ChatSession(Config(ollama_enabled=True), tmp_path / "photo.db", USER)
    session.local.store.set_goal(USER, "Korean", 30, "2026-11-10")
    pairs = {"pairs": [{"korean": "사과", "english": "apple"}, {"korean": "물", "english": "water"}]}
    session.local.conversation.importer = NoteImporter(
        Config(ollama_enabled=True), transport=AsyncMock(return_value={"response": json.dumps(pairs)}),
        installed=AsyncMock(return_value={"qwen2.5vl:3b"}))
    reply = session.send_file("notes.jpg", b"\xff\xd8 fake jpeg bytes")
    assert "1. 사과 = apple" in reply and "Already in your deck, skipped: 물" in reply
    assert rows(session.local.store)["user_vocab"] == []  # nothing saved before the confirmation
    assert "Added 1 word" in session.send("yes")
    assert [r["word"] for r in rows(session.local.store)["user_vocab"]] == ["사과"]


def test_chat_text_notes_and_missing_vision_model(tmp_path):
    from config import Config
    from dashboard_data import ChatSession

    session = ChatSession(Config(), tmp_path / "notes.db", USER)  # local AI off
    session.local.store.set_goal(USER, "Korean", 30, "2026-11-10")
    assert "1. 포도 = grape" in session.send_file("notes.txt", "포도 - grape\n".encode())
    assert "Cancelled" in session.send("no")
    assert "needs the local AI" in session.send_file("notes.png", b"\x89PNG fake")
    assert rows(session.local.store)["user_vocab"] == []


# ---- microphone ------------------------------------------------------------------------------

class FakeWhisper:
    """Stand-in for faster_whisper.WhisperModel. `segments` is a list of (start, end, text) used for every
    language, or a dict {language: [(start, end, text, avg_logprob), ...]}."""

    def __init__(self, segments, language="en", probabilities=None):
        self.segments, self.calls = segments, []
        self.probabilities = probabilities or [(language, 0.95)]

    def detect_language(self, audio):
        return self.probabilities[0][0], self.probabilities[0][1], self.probabilities

    def transcribe(self, audio, **options):
        from types import SimpleNamespace
        self.calls.append((audio, options))
        found = self.segments.get(options["language"], []) if isinstance(self.segments, dict) else \
            [(*segment, -0.5) for segment in self.segments]
        return iter(SimpleNamespace(start=a, end=b, text=t, avg_logprob=p, compression_ratio=1.2 if len(t) < 40 else 3.0)
                    for a, b, t, p in found), None


def transcriber(model, downloaded):
    from speech import Transcriber
    state = {"downloaded": downloaded, "downloads": 0}

    def loader(name, download):
        if download:
            state["downloads"] += 1
            state["downloaded"] = True
        return model if state["downloaded"] else None
    return Transcriber("small", loader=loader, decoder=lambda wav: wav), state


def test_speech_model_is_never_downloaded_without_the_button():
    from speech import SpeechUnavailable
    voice, state = transcriber(FakeWhisper([]), downloaded=False)
    assert not voice.ready()
    with pytest.raises(SpeechUnavailable, match="480 MB"):
        voice.transcribe(b"RIFF")
    assert state["downloads"] == 0
    assert voice.download() and voice.ready() and state["downloads"] == 1


def test_transcription_joins_segments_and_stops_after_a_minute():
    model = FakeWhisper([(0.0, 1.2, " today's total is "), (1.2, 2.0, "20 minutes. "), (61.0, 62.0, "ignored")])
    voice, _ = transcriber(model, downloaded=True)
    assert voice.transcribe(b"RIFF wav") == ("today's total is 20 minutes.", "en")
    audio, options = model.calls[0]
    assert audio == b"RIFF wav" and options["vad_filter"] is True


def test_spoken_message_reaches_the_bot(tmp_path):
    from config import Config
    from dashboard_data import ChatSession, voice_message

    voice, _ = transcriber(FakeWhisper([(0.0, 1.0, "today's total is 20 minutes")]), downloaded=True)
    message, shown, note = voice_message(voice, "", b"RIFF")
    assert (message, shown, note) == ("today's total is 20 minutes", "🎤 today's total is 20 minutes", None)
    session = ChatSession(Config(), tmp_path / "voice.db", USER)
    session.local.store.set_goal(USER, "Korean", 30, "2026-11-10")
    assert "Recorded 20 self-reported minutes" in session.send(message)
    silent, _ = transcriber(FakeWhisper([]), downloaded=True)
    assert voice_message(silent, "", b"RIFF")[2].startswith("I didn't catch anything")
    missing, _ = transcriber(FakeWhisper([]), downloaded=False)
    assert voice_message(missing, "hi", b"RIFF") == ("hi", "hi", "The speech model isn't downloaded yet (480 MB). "
                                                     "Press the download button above to turn on voice input.")


def test_confident_english_is_transcribed_once_with_english_hints():
    model = FakeWhisper({"en": [(0.0, 1.0, "quiz me", -0.6)]}, probabilities=[("en", 0.9), ("ko", 0.01)])
    voice, _ = transcriber(model, downloaded=True)
    assert voice.transcribe(b"RIFF", korean_hints="학교") == ("quiz me", "en")
    assert [call[1]["language"] for call in model.calls] == ["en"]
    assert "quiz me" in model.calls[0][1]["hotwords"]


@pytest.mark.parametrize("english,korean,expected", [
    ((-0.9, "with me"), (-0.7, "퀴즈 미"), ("퀴즈 미", "ko")),       # Korean transcript is more confident
    ((-0.6, "quiz me"), (-1.2, "꾸즈 미"), ("quiz me", "en")),       # English transcript is more confident
    ((None, ""), (-1.2, "모르겠어요"), ("모르겠어요", "ko")),            # an empty transcript never wins
    ((-0.2, "学教 学教"), (-0.6, "학교"), ("학교", "ko")),              # wrong script on the English pass
    ((-0.1, "quiz me, " * 8), (-0.7, "퀴즈 내줘"), ("퀴즈 내줘", "ko")),  # a repetition loop never wins
    ((-0.6, "I don't know"), (-0.3, "I don't know"), ("I don't know", "en")),  # no Hangul: not Korean
])
def test_unsure_clips_are_tried_both_ways_and_the_confident_one_wins(english, korean, expected):
    segments = {"en": [(0.0, 1.0, english[1], english[0])] if english[1] else [],
                "ko": [(0.0, 1.0, korean[1], korean[0])]}
    model = FakeWhisper(segments, probabilities=[("ja", 0.4), ("en", 0.3), ("ko", 0.2), ("vi", 0.1)])
    voice, _ = transcriber(model, downloaded=True)
    assert voice.transcribe(b"RIFF", korean_hints="학교 나무") == expected
    assert sorted(call[1]["language"] for call in model.calls) == ["en", "ko"]
    assert next(c for c in model.calls if c[1]["language"] == "ko")[1]["hotwords"] == "학교 나무"


def test_vietnamese_and_a_chosen_language():
    vietnamese = {"vi": [(0.0, 1.0, "hôm nay tôi học 20 phút", -0.3)], "ko": [(0.0, 1.0, "나무", -0.9)],
                  "en": [(0.0, 1.0, "nam", -1.5)]}
    sure = [("vi", 0.6), ("en", 0.05)]
    model = FakeWhisper(vietnamese, probabilities=sure)
    voice, _ = transcriber(model, downloaded=True)
    assert voice.transcribe(b"RIFF") == ("hôm nay tôi học 20 phút", "vi")
    assert sorted(call[1]["language"] for call in model.calls) == ["en", "ko", "vi"]
    assert all(call[1]["max_new_tokens"] == 100 for call in model.calls)
    # while an active quiz waits Vietnamese is not considered, so the clip is tried as English and Korean
    assert voice.transcribe(b"RIFF", allow_vietnamese=False) == ("나무", "ko")
    # the dashboard's language switch skips detection entirely
    model = FakeWhisper(vietnamese, probabilities=sure)
    voice, _ = transcriber(model, downloaded=True)
    assert voice.transcribe(b"RIFF", language="ko") == ("나무", "ko") and len(model.calls) == 1


def test_voice_only_message_has_no_none_prefix():
    from dashboard_data import voice_message
    voice, _ = transcriber(FakeWhisper([(0.0, 1.0, "quiz me")]), downloaded=True)
    assert voice_message(voice, None, b"RIFF")[:2] == ("quiz me", "🎤 quiz me")


def test_listening_hints_follow_the_waiting_quiz(board):
    populate(board)
    allow_vietnamese, hints = board.listening_hints(DAY)
    assert allow_vietnamese and "한글로 퀴즈" in hints and "학교" not in hints
    task = board.start_active(DAY)
    allow_vietnamese, hints = board.listening_hints(DAY)
    assert allow_vietnamese is False and "학교" in hints
    board.answer_active(DAY, task["id"], task["word"])
    assert board.listening_hints(DAY)[0] is True
