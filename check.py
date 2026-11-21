"""Read-only database health check; never print personal goal or report rows."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv

from config import Config


def inspect_database(path):
    location = Path(path).resolve()
    if not location.is_file():
        raise ValueError("Database not found; run the bot to initialize it")
    with closing(sqlite3.connect(location.as_uri() + "?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {}
        for table in ("goals", "reports", "events", "verification_attempts", "recovery_proposals", "checkin_deliveries"):
            if table in tables:
                counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"schema_version": db.execute("PRAGMA user_version").fetchone()[0], "row_counts": counts}


def main():
    load_dotenv()
    print(json.dumps(inspect_database(Config.from_env().database_path), indent=2))


if __name__ == "__main__":
    main()
