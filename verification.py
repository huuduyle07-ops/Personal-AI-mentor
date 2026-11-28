"""Vocabulary quiz scheduled by a learned forgetting curve; never evidence of study duration.

Which word to ask next:
1. Right after a wrong answer, retry that word once (the support hint was just shown).
2. Otherwise, review the known word with the lowest predicted recall, if it is below TARGET_RECALL.
   Words answered in the last REST period are skipped, so spacing needs real time to pass.
3. Otherwise, introduce the next new word from the deck.
4. With no new words left, review the word with the lowest predicted recall.
"""
import csv
import json
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

from memory import record
from personalize import MIN_EVENTS, personal_model
from recall_model import FOREVER, RecallModel
from time_utils import now_kst, valid_day

DECK = Path(__file__).with_name("vocab") / "korean_basic.csv"
TARGET_RECALL = 0.9  # review a word once its predicted recall drops below 90%
# The learned model rewards each correct answer only slightly, so a word with one mistake stays
# just under the target even right after a correct answer. Resting it makes the quiz move on.
REST = timedelta(hours=1)


def question(word, answers):
    answers = [answer.strip().casefold() for answer in answers if answer.strip()]
    return {"prompt": f"What does {word} mean in English?", "answers": set(answers), "word": word,
            "meaning": answers[0], "support": f"{word} means {answers[0]}. Read it once, then try the quiz again."}


def load_deck(path=DECK):
    questions = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            answers = [answer for answer in row["answers"].split("|") if answer.strip()]
            if not row["id"] or not row["word"] or not answers or row["id"] in questions:
                raise ValueError(f"Invalid vocabulary row: {row}")
            questions[row["id"]] = question(row["word"], answers)
    return questions


QUESTIONS = load_deck()  # the built-in deck shared by everyone


def user_deck(db, user_id):
    """The user's deck: words they imported from their notes (oldest first), then the built-in words.

    Imported words come first so they are introduced as soon as no seen word is due.
    """
    deck = {f"user_{row['id']}": question(row["word"], json.loads(row["answers"]))
            for row in db.execute("SELECT id, word, answers FROM user_vocab WHERE user_id=? ORDER BY id", (user_id,))}
    return {**deck, **QUESTIONS}


def duration(days):
    """Human-readable length of time."""
    if days == FOREVER or days > 365:
        return "over a year"
    if days < 1 / 24:
        return f"{max(1, round(days * 1440))} min"
    if days < 2:
        return f"{days * 24:.0f} h"
    return f"{days:.0f} days"


def model_label(source, reviews):
    if source == "personalized":
        return f"personalized from your {reviews} reviews"
    if source == "population":
        return f"Duolingo population model (personalizes after {MIN_EVENTS} reviews; you have {reviews})"
    return "fixed Leitner rule (model weights unavailable)"


def explain(task):
    """One line telling the user why this word was chosen."""
    if task["policy"] == "review":
        who = "your personalized" if task.get("recall_model") == "personalized" else "the"
        return f"Review: {who} recall model estimates a {task['predicted_recall']:.0%} chance you still remember it."
    if task["policy"] == "retry_after_support":
        return "Let's try this one again."
    return "New word."


class Verification:
    def __init__(self, store, model=None):
        self.store, self.model = store, model or RecallModel()

    def _history(self, db, user_id, deck):
        rows = db.execute(
            """SELECT question_id, SUM(state IN ('verified','passed_after_support')) AS right,
            SUM(state='review_required') AS wrong, MAX(answered_at) AS last
            FROM verification_attempts WHERE user_id=? AND state!='pending' GROUP BY question_id""",
            (user_id,),
        ).fetchall()
        return {row["question_id"]: row for row in rows if row["question_id"] in deck}

    def choose(self, history, now, model=None, deck=None):
        """Return (question_id, policy, predicted_recall) using the learned recall model."""
        model, deck = model or self.model, deck or QUESTIONS
        recall, resting = {}, set()
        for question_id, row in history.items():
            elapsed = now - datetime.fromisoformat(row["last"])
            recall[question_id] = model.recall(row["right"], row["wrong"], elapsed.total_seconds() / 86400)
            if elapsed < REST:
                resting.add(question_id)
        due = [(p, question_id) for question_id, p in recall.items() if p < TARGET_RECALL and question_id not in resting]
        if due:
            p, question_id = min(due)
            return question_id, "review", p
        unseen = [question_id for question_id in deck if question_id not in history]
        if unseen:
            return unseen[0], "new_word", None
        candidates = [(p, question_id) for question_id, p in recall.items() if question_id not in resting]
        p, question_id = min(candidates or [(p, question_id) for question_id, p in recall.items()])
        return question_id, "review", p

    def _select(self, db, user_id, day, now):
        """What the next quiz would be, without writing anything: (deck, report, pending, choice)."""
        deck = user_deck(db, user_id)
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
            return deck, report, pending, None
        previous = db.execute(
            "SELECT * FROM verification_attempts WHERE progress_id=? ORDER BY id DESC LIMIT 1",
            (report["id"],),
        ).fetchone()
        model, reviews = personal_model(db, user_id, self.model)
        if previous is not None and previous["state"] == "review_required":
            question_id, policy, predicted = previous["question_id"], "retry_after_support", None
        else:
            question_id, policy, predicted = self.choose(self._history(db, user_id, deck), now, model, deck)
        return deck, report, None, (question_id, policy, predicted, model, reviews)

    def next_up(self, user_id, day, *, now=None):
        """Read-only preview of the word the next quiz would ask (None while a quiz is waiting)."""
        valid_day(day)
        with self.store.connect() as db:
            deck, _, pending, choice = self._select(db, user_id, day, now or now_kst())
        if pending:
            return None
        return {"question_id": choice[0], **deck[choice[0]]}

    def start(self, user_id, day, *, db=None, now=None, sentence=None):
        """Record and return the next quiz. `sentence` is an optional validated sentence prompt for the
        previewed word ({question_id, sentence, translation, model}) or a failed attempt ({failure})."""
        valid_day(day)
        now = now or now_kst()
        with self.store.transaction(db) as db:
            deck, report, pending, choice = self._select(db, user_id, day, now)
            if pending:
                return {"id": pending["id"], "question_id": pending["question_id"], "policy": "pending",
                        "predicted_recall": None, "style": "word", **deck[pending["question_id"]]}
            question_id, policy, predicted, model, reviews = choice
            # A word missed earlier on this report keeps its support history, whenever it comes back.
            supported = db.execute(
                "SELECT 1 FROM verification_attempts WHERE progress_id=? AND question_id=? AND state='review_required'",
                (report["id"], question_id),
            ).fetchone() is not None
            result = db.execute(
                """INSERT INTO verification_attempts(user_id,day,progress_id,question_id,supported,created_at)
                VALUES(?,?,?,?,?,?)""",
                (user_id, day, report["id"], question_id, int(supported), now.isoformat()),
            )
            style, presentation = "word", {}
            if sentence is not None:
                if sentence.get("question_id") == question_id and sentence.get("sentence"):
                    style, presentation = "sentence", {"sentence_model": sentence["model"]}
                else:
                    presentation = {"sentence_fallback": sentence.get("failure", "scheduler_changed")}
            # Same event as every quiz, plus how it was presented; the generated text itself is not stored.
            record(db, user_id, day, "quiz_offered",
                   {"attempt_id": result.lastrowid, "question_id": question_id, "policy": policy,
                    "predicted_recall": None if predicted is None else round(predicted, 4),
                    "recall_model": model.source, "review_events": reviews, "style": style, **presentation}, "system")
            task = {"id": result.lastrowid, "question_id": question_id, "policy": policy, "style": style,
                    "predicted_recall": predicted, "recall_model": model.source, **deck[question_id]}
            if style == "sentence":
                task.update(sentence=sentence["sentence"], translation=sentence["translation"])
            return task

    def recall_report(self, user_id, *, now=None, limit=5):
        """Seen words ranked by what the scheduler will do next, with the reason for each."""
        now = now or now_kst()
        with self.store.connect() as db:
            model, reviews = personal_model(db, user_id, self.model)
            deck = user_deck(db, user_id)
            history = self._history(db, user_id, deck)
        rows = []
        for question_id, row in history.items():
            elapsed = (now - datetime.fromisoformat(row["last"])).total_seconds() / 86400
            p = model.recall(row["right"], row["wrong"], elapsed)
            due_in = max(0.0, model.days_until(row["right"], row["wrong"], TARGET_RECALL) - elapsed)
            rest_left = REST.total_seconds() / 86400 - elapsed
            if rest_left > 0:
                status, why, rank = "resting", f"answered {duration(elapsed)} ago; can come back in {duration(rest_left)}", 1
            elif p < TARGET_RECALL:
                status, why, rank = "due", f"predicted recall {p:.0%} is below the {TARGET_RECALL:.0%} target", 0
            else:
                status, why, rank = "not due", f"stays above {TARGET_RECALL:.0%} for about {duration(due_in)}", 2
            rows.append({"question_id": question_id, "word": deck[question_id]["word"], "meaning": deck[question_id]["meaning"],
                         "right": row["right"], "wrong": row["wrong"], "recall": p, "status": status, "why": why,
                         "half_life_days": model.half_life(row["right"], row["wrong"]), "rank": (rank, p)})
        rows.sort(key=lambda item: item["rank"])
        upcoming = None
        if deck:
            question_id, policy, predicted = self.choose(history, now, model, deck)
            upcoming = {"question_id": question_id, "word": deck[question_id]["word"], "policy": policy,
                        "predicted_recall": predicted}
        return {"model": model.source, "reviews": reviews, "label": model_label(model.source, reviews),
                "seen": len(history), "deck": len(deck), "words": rows[:limit], "next": upcoming}

    def answer(self, user_id, day, attempt_id, answer, *, db=None, now=None):
        valid_day(day)
        if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 100:
            raise ValueError("Answer must contain 1 to 100 characters")
        with self.store.transaction(db) as db:
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
            question = user_deck(db, user_id)[attempt["question_id"]]
            correct = unicodedata.normalize("NFKC", answer).strip().casefold() in question["answers"]
            state = ("passed_after_support" if attempt["supported"] else "verified") if correct else "review_required"
            db.execute("UPDATE verification_attempts SET state=?,answered_at=? WHERE id=?",
                       (state, (now or now_kst()).isoformat(), attempt_id))
            result = {"attempt_id": attempt_id, "question_id": attempt["question_id"], "state": state}
            typed = unicodedata.normalize("NFKC", answer).strip()
            if not correct and typed != "(don't know)":
                result["given"] = typed  # only wrong answers, so confusion pairs can be found
            record(db, user_id, day, "verification_result", result, "quiz_result")
        return {"state": state, "correct": correct, "word": question["word"], "meaning": question["meaning"],
                "message": "Tested vocabulary passed." if correct else question["support"]}


def format_recall_report(report, confusions=""):
    lines = [f"Recall model: {report['label']}.", f"Words seen: {report['seen']} of {report['deck']}."]
    if not report["words"]:
        lines.append("No quiz answers yet. Say “quiz me” to start.")
    for item in report["words"]:
        lines.append(f"{item['word']} ({item['meaning']}): recall {item['recall']:.0%}, half-life "
                     f"{duration(item['half_life_days'])}, {item['right']} right / {item['wrong']} wrong. "
                     f"{item['status'].capitalize()}: {item['why']}.")
    upcoming = report["next"]
    if upcoming:
        if upcoming["policy"] == "review":
            reason = f"lowest predicted recall ({upcoming['predicted_recall']:.0%}) among words available"
        else:
            reason = "no seen word is due, so a new word comes next"
        lines.append(f"Next quiz: {upcoming['word']} ({reason}).")
    if report["words"] and report["model"] != "personalized" and all(
            item["half_life_days"] > 365 for item in report["words"]):
        lines.append("Half-life = time until predicted recall falls to 50%. The population model learned from "
                     "Duolingo learners who rarely forget between reviews, so it predicts very slow forgetting; "
                     "personalization adjusts this once you have reviews spread over several days.")
    if confusions:
        lines.append(confusions)
    return "\n".join(lines)
