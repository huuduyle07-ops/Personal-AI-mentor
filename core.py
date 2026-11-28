"""Goal domain and SQLite unit of work; no Discord or inference dependencies."""
import sqlite3
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from typing import NamedTuple

import chat_context
from memory import migrate, record
from time_utils import now_kst, valid_day, valid_minutes


class ReportResult(NamedTuple):
    report_id: int
    changed: bool


class GoalStore:
    def __init__(self, path="data/accountability.db"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            migrate(db)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db

    @staticmethod
    def plan(db, user_id, day):
        goal = db.execute("SELECT * FROM goals WHERE user_id=?", (user_id,)).fetchone()
        if goal is None:
            raise ValueError("Create a goal first with /setgoal")
        db.execute(
            "INSERT OR IGNORE INTO daily_goals(user_id,day,title,original_target,target) VALUES(?,?,?,?,?)",
            (user_id, day, goal["title"], goal["target_minutes"], goal["target_minutes"]),
        )
        return db.execute("SELECT * FROM daily_goals WHERE user_id=? AND day=?", (user_id, day)).fetchone()

    @contextmanager
    def transaction(self, db=None):
        """Join the caller's atomic operation, or own a new write transaction."""
        if db is not None:
            yield db
            return
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def get_state(self, key):
        with self.connect() as db:
            row = db.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_state(self, key, value):
        with self.connect() as db:
            db.execute("INSERT INTO app_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (key, value))

    def set_goal(self, user_id, title, target_minutes, day=None):
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
            raise ValueError("Goal title must contain 1 to 200 characters")
        valid_minutes(target_minutes, 1, 600)
        day, title = valid_day(day or now_kst().date().isoformat()), title.strip()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM goals WHERE user_id=?", (user_id,)).fetchone()
            db.execute(
                """INSERT INTO goals VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                title=excluded.title,target_minutes=excluded.target_minutes""",
                (user_id, title, target_minutes),
            )
            db.execute(
                """INSERT INTO daily_goals(user_id,day,title,original_target,target) VALUES(?,?,?,?,?)
                ON CONFLICT(user_id,day) DO UPDATE SET title=excluded.title,
                original_target=excluded.original_target,target=excluded.target,revision=revision+1""",
                (user_id, day, title, target_minutes, target_minutes),
            )
            record(db, user_id, day, "goal_set", {"previous": dict(old) if old else None,
                   "title": title, "target_minutes": target_minutes})

    @staticmethod
    def latest_report(db, user_id, day):
        return db.execute("SELECT id,minutes FROM progress_events WHERE user_id=? AND day=? "
                          "ORDER BY id DESC LIMIT 1", (user_id, day)).fetchone()

    def report(self, user_id, day, minutes, *, db=None):
        """Record today's total. Repeating the current total is a no-op (changed=False),
        so it doesn't add a duplicate report or detach a quiz result from the existing one."""
        valid_day(day)
        valid_minutes(minutes)
        with self.transaction(db) as db:
            self.plan(db, user_id, day)
            latest = self.latest_report(db, user_id, day)
            if latest is not None and latest["minutes"] == minutes:
                return ReportResult(latest["id"], changed=False)
            db.execute(
                """INSERT INTO reports VALUES(?,?,?) ON CONFLICT(user_id,day)
                DO UPDATE SET minutes=excluded.minutes""", (user_id, day, minutes),
            )
            result = db.execute(
                "INSERT INTO progress_events(user_id,day,minutes,created_at) VALUES(?,?,?,?)",
                (user_id, day, minutes, now_kst().isoformat()),
            )
            record(db, user_id, day, "progress_reported", {"minutes": minutes, "report_id": result.lastrowid})
            return ReportResult(result.lastrowid, changed=True)

    def summary(self, user_id, day, *, db=None):
        valid_day(day)
        with (self.connect() if db is None else nullcontext(db)) as db:
            if not db.execute("SELECT 1 FROM goals WHERE user_id=?", (user_id,)).fetchone():
                return None
            plan = self.plan(db, user_id, day)
            report = db.execute(
                "SELECT * FROM progress_events WHERE user_id=? AND day=? ORDER BY id DESC LIMIT 1",
                (user_id, day),
            ).fetchone()
            verification = None
            if report:
                verification = db.execute(
                    """SELECT state FROM verification_attempts WHERE progress_id=? AND state!='pending'
                    ORDER BY id DESC LIMIT 1""", (report["id"],),
                ).fetchone()
            goal = db.execute("SELECT target_minutes FROM goals WHERE user_id=?", (user_id,)).fetchone()
            streak = chat_context.streak(db, user_id, day, goal["target_minutes"])
            recoveries = [dict(row) for row in db.execute(
                "SELECT id,reason,status,proposed_target FROM recovery_proposals WHERE user_id=? AND day=? ORDER BY id",
                (user_id, day),
            )]
        minutes = report["minutes"] if report else None
        return {"title": plan["title"], "target": plan["target"],
                "original_target": plan["original_target"], "minutes": minutes,
                "status": "reported" if report else "not_reported",
                "verification_state": verification["state"] if verification else ("reported" if report else "not_reported"),
                "target_reached": minutes is not None and minutes >= plan["target"],
                "recoveries": recoveries, "streak": streak}


def daily_review(data):
    if data is None:
        return "No goal found. Use /setgoal first."
    progress = "not reported" if data["minutes"] is None else str(data["minutes"])
    lines = [f"{data['title']}", f"Reported: {progress} / {data['target']} min",
             f"Original daily target: {data['original_target']} min",
             f"Activity status: {data['status']}", f"Learning verification: {data['verification_state']}",
             f"Target reached (self-reported): {data['target_reached']}",
             f"Streak: {data['streak']} day{'s' if data['streak'] != 1 else ''} in a row meeting your goal",
             "A quiz checks only the tested vocabulary; it does not prove study duration."]
    if data["minutes"] is None:
        lines.append("No response is recorded; no conclusion about effort or failure is made.")
    for proposal in data["recoveries"][-3:]:
        lines.append(f"Obstacle: {proposal['reason']} | adjustment #{proposal['id']}: {proposal['status']}")
    return "\n".join(lines)
