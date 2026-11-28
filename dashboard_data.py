"""Data for the Streamlit dashboard (dashboard.py), in plain Python so it is testable without Streamlit.

Read-only helpers (overview, memory, policy) only SELECT, through the same modules the bot uses: focus,
chat_context, verification.recall_report, personalize, confusion and checkin_policy.preview. None of them
calls GoalStore.summary()/plan(), which would create today's plan row.

The writes the dashboard can make are the explicit actions at the bottom, each triggered by a button:
starting/answering a quiz (the same Verification / ActiveQuiz code as the chat) and adding confirmed
note words. Active answers stay in active_attempts, never in the recognition history personalize.py reads.
"""
import asyncio
import threading
from datetime import date, datetime, timedelta

import checkin_policy
import confusion
import focus
import personalize
import study_notes
from active_quiz import ActiveQuiz
from core import GoalStore
from recall_model import FEATURES as RECALL_FEATURES, RecallModel
from time_utils import now_kst
from verification import Verification, duration, user_deck


class Dashboard:
    def __init__(self, database_path, user_id):
        self.store = GoalStore(database_path)  # opening runs the additive CREATE TABLE IF NOT EXISTS migrations only
        self.user_id = user_id
        self.verification = Verification(self.store)
        self.active = ActiveQuiz(self.store, self.verification)

    # ---- read-only ---------------------------------------------------------------------------

    def overview(self, now=None):
        now = now or now_kst()
        f = focus.snapshot(self.store, self.verification, self.user_id, now)
        return {**f, "min_events": personalize.MIN_EVENTS, "suggestion": focus.suggestion(f),
                "text": focus.format_focus(f), "week": self.week_minutes(now)}

    def week_minutes(self, now=None, days=7):
        """Latest reported total (minutes) and target for each of the last `days` days, oldest first."""
        today = (now or now_kst()).date()
        with self.store.connect() as db:
            goal = db.execute("SELECT target_minutes FROM goals WHERE user_id=?", (self.user_id,)).fetchone()
            out = []
            for offset in range(days - 1, -1, -1):
                day = (today - timedelta(days=offset)).isoformat()
                latest = db.execute("SELECT minutes FROM progress_events WHERE user_id=? AND day=? "
                                    "ORDER BY id DESC LIMIT 1", (self.user_id, day)).fetchone()
                plan = db.execute("SELECT target FROM daily_goals WHERE user_id=? AND day=?",
                                  (self.user_id, day)).fetchone()
                target = plan["target"] if plan else (goal["target_minutes"] if goal else None)
                out.append({"day": day, "minutes": latest["minutes"] if latest else 0, "target": target})
        return out

    def memory(self, now=None):
        now = now or now_kst()
        report = self.verification.recall_report(self.user_id, now=now, limit=10_000)
        population = RecallModel()
        with self.store.connect() as db:
            events = personalize.recall_events(db, self.user_id)
            model, reviews = personalize.personal_model(db, self.user_id, population)
        weights = [{"feature": name, "population": population.weights[i] if population.weights else None,
                    "in_use": model.weights[i] if model.weights else None}
                   for i, name in enumerate(RECALL_FEATURES)]
        words = [{"word": w["word"], "meaning": w["meaning"], "recall": w["recall"], "status": w["status"],
                  "right": w["right"], "wrong": w["wrong"], "half_life_days": w["half_life_days"], "half_life": duration(w["half_life_days"]), "why": w["why"]}
                 for w in report["words"]]
        return {"model": report["model"], "label": report["label"], "reviews": reviews,
                "min_events": personalize.MIN_EVENTS, "personalized": model.source == "personalized",
                "review_events": len(events), "correct_events": sum(e[3] for e in events),
                "weights": weights, "words": words, "seen": report["seen"], "deck": report["deck"],
                "next": report["next"], "pairs": confusion.pairs(self.store, self.user_id, limit=None)}

    def policy(self, now=None):
        now = now or now_kst()
        report = checkin_policy.preview(self.store, self.verification, self.user_id, now)
        with self.store.connect() as db:
            history = [dict(row) for row in db.execute(
                "SELECT day, slot, action, policy_choice, decided_at, engaged, reward, resolved_at FROM bandit_shadow "
                "WHERE user_id=? ORDER BY id DESC LIMIT 100", (self.user_id,))]
            settings = db.execute("SELECT enabled, paused, midday, evening FROM checkin_settings WHERE user_id=?",
                                  (self.user_id,)).fetchone()
        scores = [{"action": action, **report["scores"][action]} for action in checkin_policy.ACTIONS]
        return {"features": report["features"], "scores": scores, "action": report["action"],
                "updates": report["updates"], "alpha": report["alpha"], "shadow": report["shadow"],
                "history": history, "checkins": dict(settings) if settings else None}

    def pending_quizzes(self, day):
        """Quizzes waiting for an answer today: {'passive': task or None, 'active': task or None}. Read-only."""
        with self.store.connect() as db:
            deck = user_deck(db, self.user_id)
            passive = db.execute("SELECT id, question_id FROM verification_attempts WHERE user_id=? AND day=? "
                                 "AND state='pending' ORDER BY id DESC LIMIT 1", (self.user_id, day)).fetchone()
            active = db.execute("SELECT id, question_id FROM active_attempts WHERE user_id=? AND day=? "
                                "AND state='pending' ORDER BY id DESC LIMIT 1", (self.user_id, day)).fetchone()
        out = {"passive": None, "active": None}
        if passive and passive["question_id"] in deck:
            out["passive"] = {"id": passive["id"], **deck[passive["question_id"]]}
        if active and active["question_id"] in deck:
            q = deck[active["question_id"]]
            out["active"] = {"id": active["id"], "word": q["word"], "meaning": q["meaning"],
                             "prompt": f"How do you write “{q['meaning']}” in Korean? Type it in Hangul."}
        return out

    def listening_hints(self, day):
        """(allow_vietnamese, Korean hints) for the next recording. While an active quiz waits, the answer is
        Korean or an English request, never Vietnamese, and the deck words are added to the hints. Read-only."""
        from speech import hotwords

        # Deck words only help while a Korean answer is expected; otherwise they dilute the request hints.
        quiz = self.pending_quizzes(day)["active"] is not None
        return not quiz, hotwords(self.known_words() if quiz else ())

    # ---- explicit actions (each one is a button in the dashboard) ----------------------------

    def start_passive(self, day):
        """Korean -> English flashcard; needs today's reported minutes, like /quiz."""
        return self.verification.start(self.user_id, day)

    def answer_passive(self, day, attempt_id, text):
        return self.verification.answer(self.user_id, day, attempt_id, text)

    def start_active(self, day):
        """English -> Korean; stored in active_attempts only."""
        return self.active.start(self.user_id, day)

    def answer_active(self, day, attempt_id, text):
        return self.active.answer(self.user_id, day, attempt_id, text)

    def known_words(self):
        with self.store.connect() as db:
            return [q["word"] for q in user_deck(db, self.user_id).values()]

    def preview_import(self, pairs):
        """(new items, words already in the deck) for extracted pairs. Read-only: nothing is stored yet."""
        return study_notes.deduplicate(pairs, self.known_words())

    def confirm_import(self, items, source, day=None):
        """Add the words the user confirmed with the button. One transaction, through the bot's import code."""
        if not items:
            return "Nothing selected, so nothing was added."
        day = day or now_kst().date().isoformat()
        with self.store.transaction() as db:
            study_notes.propose(db, self.user_id, items, source, [q["word"] for q in user_deck(db, self.user_id).values()])
            return study_notes.confirm(db, self.user_id, day) or "Nothing new to add."


class ChatSession:
    """The terminal chat (chat.LocalChat: same conversation logic, confirmations and database as the bot),
    driven from Streamlit. Its coroutines run on one long-lived event loop in a background thread, because
    Streamlit reruns the script for every message and the conversation's lock must stay on a single loop."""

    def __init__(self, config, database_path, user_id):
        import chat  # imported here so the read-only helpers above don't load the chat stack

        self.local = chat.LocalChat(config, GoalStore(database_path), user_id)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="dashboard-chat", daemon=True).start()

    def send(self, text, timeout=180):
        """The bot's reply to one message (or a /command), as text."""
        future = asyncio.run_coroutine_threadsafe(self.local.handle(text), self.loop)
        reply = future.result(timeout=timeout)
        return reply if reply else "(no reply)"

    def send_file(self, filename, data, timeout=180):
        """Import words from a notes photo or text file, exactly like an attachment in the Discord DM: the bot
        lists what it found and nothing is saved until you reply yes. The local vision model reads photos."""
        import chat

        coroutine = self.local.conversation.import_notes(self.local.user_id, self.local.next_message_id(), filename,
                                                         data, chat.today())
        reply = asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=timeout)
        return reply if reply else "(no reply)"


def voice_message(voice, typed, wav, language=None, korean_hints="", allow_vietnamese=True):
    """Combine typed text with a recording: (message for the bot, text for the user's bubble, note or None).
    `voice` is a speech.Transcriber; a missing model or silence gives a note instead of a message."""
    from speech import SpeechUnavailable

    try:
        heard, _ = voice.transcribe(wav, language=language, korean_hints=korean_hints,
                                    allow_vietnamese=allow_vietnamese)
    except SpeechUnavailable as error:
        return typed, typed, f"{error} Press the download button above to turn on voice input."
    typed = typed or ""
    if not heard:
        return typed, typed, "I didn't catch anything. Try again a little closer to the mic."
    return f"{typed} {heard}".strip(), f"{typed} 🎤 {heard}".strip(), None


def short_day(day):
    return date.fromisoformat(day).strftime("%a %d").replace(" 0", " ")


def when(iso):
    """'Nov 27 12:00', or '' for a missing value (None, or NaN once it has been through pandas)."""
    return datetime.fromisoformat(iso).strftime("%b %d %H:%M") if isinstance(iso, str) and iso else ""


def context_rows(features):
    return [{"feature": name, "value": features[name]} for name in checkin_policy.FEATURES]
