"""Versioned, additive SQLite migrations and minimal append-only event history."""
import json

from time_utils import now_kst


MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS goals (
        user_id INTEGER PRIMARY KEY, title TEXT NOT NULL, target_minutes INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS reports (
        user_id INTEGER NOT NULL, day TEXT NOT NULL, minutes INTEGER NOT NULL,
        PRIMARY KEY(user_id, day), FOREIGN KEY(user_id) REFERENCES goals(user_id)
    );
    """,
    """
    CREATE TABLE events (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, day TEXT NOT NULL,
        kind TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
    );
    CREATE INDEX events_history ON events(user_id, day, id);
    CREATE TABLE daily_goals (
        user_id INTEGER NOT NULL, day TEXT NOT NULL, title TEXT NOT NULL,
        original_target INTEGER NOT NULL, target INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY(user_id, day), FOREIGN KEY(user_id) REFERENCES goals(user_id)
    );
    INSERT INTO daily_goals(user_id,day,title,original_target,target)
        SELECT r.user_id,r.day,g.title,g.target_minutes,g.target_minutes
        FROM reports r JOIN goals g ON g.user_id=r.user_id;
    CREATE TABLE progress_events (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, day TEXT NOT NULL,
        minutes INTEGER NOT NULL, created_at TEXT NOT NULL
    );
    INSERT INTO progress_events(user_id,day,minutes,created_at)
        SELECT user_id,day,minutes,'legacy_timestamp_unknown' FROM reports;
    CREATE TABLE verification_attempts (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, day TEXT NOT NULL,
        progress_id INTEGER NOT NULL REFERENCES progress_events(id),
        question_id TEXT NOT NULL, supported INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, answered_at TEXT
    );
    CREATE TABLE recovery_proposals (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, day TEXT NOT NULL,
        reason TEXT NOT NULL, original_target INTEGER NOT NULL, proposed_target INTEGER NOT NULL,
        plan_revision INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL, resolved_at TEXT
    );
    CREATE TABLE checkin_settings (
        user_id INTEGER PRIMARY KEY, enabled INTEGER NOT NULL, paused INTEGER NOT NULL DEFAULT 0,
        midday TEXT NOT NULL, evening TEXT NOT NULL, active_since TEXT NOT NULL
    );
    CREATE TABLE checkin_deliveries (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, day TEXT NOT NULL, slot TEXT NOT NULL,
        status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL, retry_at TEXT, error_code TEXT, UNIQUE(user_id,day,slot)
    );
    """,
]


def migrate(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > len(MIGRATIONS):
            raise RuntimeError("Database is newer than this application; refusing to downgrade")
        for index in range(version, len(MIGRATIONS)):
            for statement in MIGRATIONS[index].split(";"):
                if statement.strip():
                    db.execute(statement)
            db.execute(f"PRAGMA user_version = {index + 1}")
        db.commit()
    except BaseException:
        db.rollback()
        raise


def record(db, user_id, day, kind, payload, source="user_reported"):
    if source not in {"user_reported", "system", "quiz_result", "model_interpretation"}:
        raise ValueError("Unknown evidence source")
    db.execute(
        "INSERT INTO events(user_id,day,kind,source,payload,created_at) VALUES(?,?,?,?,?,?)",
        (user_id, day, kind, source, json.dumps(payload, ensure_ascii=False), now_kst().isoformat()),
    )


class Memory:
    def __init__(self, store):
        self.store = store

    def history(self, user_id, day=None, kind=None, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("History limit must be between 1 and 100")
        query, args = "SELECT * FROM events WHERE user_id=?", [user_id]
        for key, value in (("day", day), ("kind", kind)):
            if value is not None:
                query += f" AND {key}=?"
                args.append(value)
        with self.store.connect() as db:
            rows = db.execute(query + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]
