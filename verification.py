"""A deliberately small demonstration quiz; never evidence of study duration."""
import unicodedata

from memory import record
from time_utils import now_kst, valid_day

QUESTIONS = {
    "ko_water": {"prompt": "What does 물 mean in English?", "answers": {"water"},
                 "support": "물 means water. Read it once, then try the quiz again."},
    "ko_school": {"prompt": "What does 학교 mean in English?", "answers": {"school"},
                  "support": "학교 means school. Read it once, then try the quiz again."},
}


class Verification:
    def __init__(self, store):
        self.store = store

    def start(self, user_id, day):
        valid_day(day)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            report = db.execute(
                "SELECT id FROM progress_events WHERE user_id=? AND day=? ORDER BY id DESC LIMIT 1",
                (user_id, day),
            ).fetchone()
            if report is None:
                raise ValueError("Report progress first with /report")
            pending = db.execute(
                "SELECT * FROM verification_attempts WHERE progress_id=? AND state='pending' ORDER BY id DESC LIMIT 1",
                (report["id"],),
            ).fetchone()
            if pending:
                return {"id": pending["id"], **QUESTIONS[pending["question_id"]]}
            previous = db.execute(
                "SELECT * FROM verification_attempts WHERE progress_id=? ORDER BY id DESC LIMIT 1",
                (report["id"],),
            ).fetchone()
            supported = previous is not None and db.execute(
                "SELECT 1 FROM verification_attempts WHERE progress_id=? AND question_id=? AND state='review_required'",
                (report["id"], previous["question_id"]),
            ).fetchone() is not None
            question_id = previous["question_id"] if supported else "ko_water"
            result = db.execute(
                """INSERT INTO verification_attempts(user_id,day,progress_id,question_id,supported,created_at)
                VALUES(?,?,?,?,?,?)""",
                (user_id, day, report["id"], question_id, int(supported), now_kst().isoformat()),
            )
            record(db, user_id, day, "quiz_offered", {"attempt_id": result.lastrowid, "question_id": question_id}, "system")
            return {"id": result.lastrowid, **QUESTIONS[question_id]}

    def answer(self, user_id, day, attempt_id, answer):
        valid_day(day)
        if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 100:
            raise ValueError("Answer must contain 1 to 100 characters")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            attempt = db.execute(
                "SELECT * FROM verification_attempts WHERE id=? AND user_id=? AND day=?",
                (attempt_id, user_id, day),
            ).fetchone()
            if attempt is None or attempt["state"] != "pending":
                raise ValueError("No pending quiz found for you today; use /quiz")
            latest = db.execute(
                "SELECT MAX(id) FROM progress_events WHERE user_id=? AND day=?", (user_id, day),
            ).fetchone()[0]
            if latest != attempt["progress_id"]:
                raise ValueError("Progress changed since this quiz; use /quiz for the latest report")
            question = QUESTIONS[attempt["question_id"]]
            correct = unicodedata.normalize("NFKC", answer).strip().casefold() in question["answers"]
            state = ("passed_after_support" if attempt["supported"] else "verified") if correct else "review_required"
            db.execute("UPDATE verification_attempts SET state=?,answered_at=? WHERE id=?",
                       (state, now_kst().isoformat(), attempt_id))
            record(db, user_id, day, "verification_result",
                   {"attempt_id": attempt_id, "question_id": attempt["question_id"], "state": state}, "quiz_result")
        return {"state": state, "message": "Tested vocabulary passed." if correct else question["support"]}
