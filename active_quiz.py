"""Active production practice: show the English meaning, the learner types the Korean word in Hangul.

Producing a word is a different (harder) task from recognising it, so these answers are stored in their
own table, `active_attempts` (mode='active'), and in events of their own kind with style 'active'. The
recognition history in `verification_attempts`, which schedules flashcards and personalizes the recall
model (personalize.recall_events), never contains them.

Which word: chosen from the read-only recall report, so asking never writes anything until the quiz starts.
  1. the due word with the lowest predicted recall,
  2. otherwise the seen word with the lowest predicted recall,
skipping words practiced actively in the last REST period. Unseen words are never asked: you can't
produce a word you haven't met yet.
"""
from datetime import datetime

import confusion
from memory import record
from time_utils import now_kst, valid_day
from verification import REST, user_deck

DONT_KNOW = "(don't know)"


class ActiveQuiz:
    def __init__(self, store, verification):
        self.store, self.verification = store, verification

    def _recent(self, db, user_id, now):
        since = (now - REST).isoformat()
        return {row["question_id"] for row in db.execute(
            "SELECT question_id FROM active_attempts WHERE user_id=? AND created_at>=?", (user_id, since))}

    def choose(self, user_id, now=None):
        """(question_id, reason) for the next active quiz, or None if no word has been seen yet. Read-only."""
        now = now or now_kst()
        report = self.verification.recall_report(user_id, now=now, limit=10_000)
        with self.store.connect() as db:
            recent = self._recent(db, user_id, now)
        words = [item for item in report["words"] if item["question_id"] not in recent]
        due = sorted((item for item in words if item["status"] == "due"), key=lambda item: item["recall"])
        if due:
            return due[0]["question_id"], "due"
        if words:
            return min(words, key=lambda item: item["recall"])["question_id"], "practice"
        if report["seen"]:  # every seen word was practiced in the last hour: take the least recent anyway
            return min(report["words"], key=lambda item: item["recall"])["question_id"], "practice"
        return None

    def pending(self, user_id, day, *, window=None):
        """Id of the active quiz waiting for an answer today (within `window`, if given), else None."""
        with self.store.connect() as db:
            row = db.execute("SELECT id, created_at FROM active_attempts WHERE user_id=? AND day=? AND state='pending' "
                             "ORDER BY id DESC LIMIT 1", (user_id, day)).fetchone()
        if row and (window is None or now_kst() - datetime.fromisoformat(row["created_at"]) <= window):
            return row["id"]
        return None

    def start(self, user_id, day, *, now=None):
        valid_day(day)
        now = now or now_kst()
        with self.store.connect() as db:
            deck = user_deck(db, user_id)
            waiting = db.execute("SELECT * FROM active_attempts WHERE user_id=? AND day=? AND state='pending' "
                                 "ORDER BY id DESC LIMIT 1", (user_id, day)).fetchone()
            if waiting:
                return self._task(waiting["id"], waiting["question_id"], "pending", deck)
        choice = self.choose(user_id, now)
        if choice is None:
            raise ValueError("Active practice uses words you've already met. Try a few flashcards with “quiz me” first.")
        question_id, reason = choice
        with self.store.transaction() as db:
            deck = user_deck(db, user_id)
            attempt = db.execute("INSERT INTO active_attempts(user_id,day,question_id,created_at) VALUES(?,?,?,?)",
                                 (user_id, day, question_id, now.isoformat())).lastrowid
            record(db, user_id, day, "active_quiz_offered",
                   {"attempt_id": attempt, "question_id": question_id, "reason": reason, "style": "active",
                    "mode": "active"}, "system")
        return self._task(attempt, question_id, reason, deck)

    @staticmethod
    def _task(attempt, question_id, reason, deck):
        question = deck[question_id]
        return {"id": attempt, "question_id": question_id, "reason": reason, "style": "active",
                "word": question["word"], "meaning": question["meaning"],
                "prompt": f"How do you write “{question['meaning']}” in Korean? Type it in Hangul."}

    def answer(self, user_id, day, attempt_id, text, *, now=None, db=None):
        valid_day(day)
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 100:
            raise ValueError("Answer must contain 1 to 100 characters")
        now = now or now_kst()
        with self.store.transaction(db) as db:
            attempt = db.execute("SELECT * FROM active_attempts WHERE id=? AND user_id=? AND day=?",
                                 (attempt_id, user_id, day)).fetchone()
            if attempt is None or attempt["state"] != "pending":
                raise ValueError("No active quiz is waiting today; say “quiz me in Korean” to start one")
            question = user_deck(db, user_id)[attempt["question_id"]]
            given = None if text == DONT_KNOW else confusion.normalize_hangul(text)
            correct = given is not None and given == confusion.normalize_hangul(question["word"])
            state = "verified" if correct else "review_required"
            db.execute("UPDATE active_attempts SET state=?, given=?, answered_at=? WHERE id=?",
                       (state, None if correct else given, now.isoformat(), attempt_id))
            record(db, user_id, day, "active_quiz_result",
                   {"attempt_id": attempt_id, "question_id": attempt["question_id"], "state": state,
                    "style": "active", "mode": "active"}, "quiz_result")
            result = {"correct": correct, "state": state, "word": question["word"], "meaning": question["meaning"],
                      "given": given, "contrast": None, "similarity": None}
            if not correct:
                if given:
                    result["similarity"] = confusion.similarity(given, question["word"])
                # Read in the same transaction, so this miss already counts.
                for pair in confusion.pairs(self.store, user_id, limit=None, db=db):
                    if attempt["question_id"] in (pair["question_id"], pair["other_id"]):
                        result["contrast"] = confusion.contrast(pair)
                        break
        return result


def feedback(result):
    word, meaning = result["word"], result["meaning"]
    if result["correct"]:
        return f"Correct! {meaning} = {word}. Say “quiz me in Korean” for another word."
    if result["given"] is None:
        text = f"{meaning} is {word}."
    elif result["similarity"] is not None and result["similarity"] >= 0.6:
        text = f"Close: {meaning} is {word} (you wrote {result['given']})."
    else:
        text = f"Not quite: {meaning} is {word} (you wrote {result['given']})."
    if result["contrast"]:
        text += f" Don't mix them up: {result['contrast']}."
    return text + " Say “quiz me in Korean” to keep practicing."
